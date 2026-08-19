"""
Общая retry-механика ОДНОЙ стадии пайплайна Router->Analyzer->Planner->
Coder->Verifier (mcp_agent/pipeline.py, mcp_agent/roles.py) — извлечена из
self-heal цикла mcp_agent/agent.py:stream_chat (recursion-limit-обработка,
восстановление после ollama.ResponseError, разбор утёкшей tool-call
разметки, punt-to-user rescue через настоящий ask_user, дайджест-ретраи по
образцу _start_next_attempt) в переиспользуемую run_stage(...).

В отличие от легаси stream_chat, здесь НЕТ ни одной строки того огромного
if/elif вердикт-дерева (~150 строк в agent.py, "wrote code but didn't
verify", "diff not read" и т.п.) — та логика имела смысл только для
монолитного агента, который делал всё сразу. Каждая стадия пайплайна
передаёт СВОЙ verdict_fn/guidance_fn (собранные из тех же чистых функций
self_heal.py, просто по-другому распределённых — см. docstring
mcp_agent/roles.py и план в .claude/plans/), а run_stage остаётся общей
"дорогой" между попытками одной стадии, не знающей, что именно проверяет
конкретный verdict_fn.

Копия механики — по точным диапазонам из mcp_agent/agent.py, та же
дисциплина, что описана в его собственном докстринге: пересочинять код по
памяти вместо копирования точных диапазонов рискует перепутать границы
stream_chat при вырезании.
"""
import time
from dataclasses import dataclass, field

import ollama
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from tools.confirm import ask_user_question
from mcp_agent.debug_log import log_event
from mcp_agent.delegate_tool import _SUBAGENT_TOOLS, _suppress_during_subagent_tools  # noqa: F401 (re-exported, see below)
from mcp_agent.model_config import MAX_SELF_HEAL_ASKS
from mcp_agent.self_heal import (
    _execute_leaked_tool_call,
    _extract_ask_user_shape,
    _leaked_tool_call_syntax,
    _parse_leaked_tool_calls,
)

_thread_counter = 0

# _SUBAGENT_TOOLS/_suppress_during_subagent_tools жили здесь, переехали в
# mcp_agent/delegate_tool.py (естественный дом — они защищают ЛЮБОГО
# вызывающего delegate от его же вложенного token-leak, а не только
# пайплайн; см. их докстринг там) и переимпортированы выше, чтобы не
# менять вызывающий код в этом файле.


def _next_thread_id() -> str:
    global _thread_counter
    _thread_counter += 1
    return f"stage-{_thread_counter}"


@dataclass
class StageResult:
    """Что стадия успела произвести к моменту, когда verdict_fn признал
    раунд подходящим, или попытки кончились. final_text — то, что стадия
    должна передать следующей (саммари Analyzer'а, план Planner'а, отчёт
    Coder'а/Verifier'а) — см. _seed_next_stage в mcp_agent/pipeline.py."""
    final_text: str
    round_msgs: list = field(default_factory=list)
    all_round_msgs: list = field(default_factory=list)  # по ВСЕМ попыткам, не только последней
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    # Суммарное время реальной генерации токенов (см. _stream_round's gen_ms
    # docstring) по ВСЕМ попыткам — no pipeline.py stage currently reports
    # this in its own "stats" event, но mcp_agent/agent.py's legacy caller
    # does (tok/s в UI), так что run_stage не должен его тихо отбрасывать.
    gen_duration_ms: int = 0
    attempts_used: int = 0
    verdict: dict | None = None  # None = ни разу не проверялось (нечего проверять), иначе последний verdict
    hit_recursion_limit: bool = False
    hit_generation_error: bool = False
    hit_context_overflow: bool = False


async def run_stage(
    agent, payload: dict, on_event, *,
    judge_model, tools_by_name: dict, read_history: dict,
    verdict_fn, guidance_fn, max_attempts: int, recursion_limit: int,
    stage_name: str, mid_turn_queue=None,
) -> StageResult:
    """Крутит agent.astream(...) с self-heal ретраями до max_attempts, как
    внутренний while-цикл mcp_agent/agent.py:stream_chat, но без завязки на
    какой-то один конкретный вердикт-набор.

    verdict_fn(round_msgs, new_tool_msgs, round_final_text) -> dict — тот же
    формат {"relevant": bool, "reason": str, "kind": str|None (опц.)}, что
    self_heal.py уже возвращает. Может быть async или sync (оборачивается
    ниже).
    guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text) -> str
    — текст коррекции для следующей попытки, вызывается только если verdict
    не relevant и попытки ещё остались.

    on_event получает ту же трубу событий, что и весь пайплайн — добавляет
    stage=stage_name к tool_start/tool_end/verdict, чтобы debug_log
    (mcp_agent/debug_log.py) и UI (ui/stream.py:"stage_changed") могли
    отличить, какая стадия сейчас говорит.

    mid_turn_queue — see _stream_round's own docstring (mcp_agent/agent.py);
    forwarded through as-is, default None means no mid-turn injection, same
    as before this parameter existed here."""
    from mcp_agent.agent import _stream_round  # локальный импорт — общий низкоуровневый стример, без цикла импортов на уровне модуля

    on_event = _suppress_during_subagent_tools(on_event)
    original_messages = list(payload["messages"])
    round_digests: list[str] = []
    all_round_msgs: list = []

    # read_history is shared per repo_path across EVERY role/stage/round for
    # the life of the process (_get_tools's cache, see agent_builder.py) —
    # but this call is about to start a brand-new thread with no memory of
    # any earlier conversation. Without this clear, a leftover "you already
    # read `path`" entry from a DIFFERENT thread (a previous stage, a
    # previous Coder<->Verifier round, even a previous turn) makes
    # _dedupe_read_tool hand this fresh thread a stub pointing at an
    # "earlier result" it never actually saw — the real file content never
    # reaches it. _seed_retry already clears this for retries WITHIN one
    # run_stage call (see below) — this extends the same invariant to the
    # call boundary itself.
    read_history.clear()
    config = {"configurable": {"thread_id": _next_thread_id()}, "recursion_limit": recursion_limit}
    tokens_in = tokens_out = llm_calls = 0
    gen_duration_ms = 0
    emitted = 0
    result = None
    attempt = 0

    round_msgs: list = []
    verdict: dict | None = None
    generation_error_bonus_used = False
    self_heal_asks_used = 0

    while attempt < max_attempts:
        log_event("stage_attempt", stage=stage_name, n=attempt + 1, max=max_attempts)
        attempt_start = emitted
        try:
            result, emitted, tin, tout, calls, hit_limit_this_round, hit_overflow_this_round, round_gen_ms = await _stream_round(
                agent, payload, config, on_event, emitted, mid_turn_queue=mid_turn_queue,
            )
            tokens_in += tin
            tokens_out += tout
            llm_calls += calls
            gen_duration_ms += round_gen_ms
        except ollama.ResponseError as e:
            if attempt == max_attempts - 1:
                # A generation failure on the LAST attempt can happen right
                # after a real, correct edit was already applied — giving up
                # immediately here would return final_text="", and
                # pipeline.py treats an empty final_text exactly like
                # "nothing was done" (touched_paths gets reverted via
                # _revert_turn_paths), throwing away a genuinely successful
                # edit over a transient generation hiccup unrelated to
                # whether the work was correct. One bonus attempt (same
                # pattern as the punt-to-user rescue below) gives the model
                # a real chance to just report what it already did, without
                # masking a SECOND consecutive generation failure — that
                # still gives up for real.
                if not generation_error_bonus_used:
                    generation_error_bonus_used = True
                    max_attempts += 1
                else:
                    return StageResult(
                        final_text="", all_round_msgs=all_round_msgs,
                        tokens_in=tokens_in, tokens_out=tokens_out, llm_calls=llm_calls,
                        gen_duration_ms=gen_duration_ms, attempts_used=attempt + 1,
                        hit_generation_error=True,
                    )
            round_digests.append(f"- rejected: the model's last response failed to generate/parse ({e})")
            payload, config, emitted = _seed_retry(
                original_messages, round_digests,
                "Your last response failed to generate correctly — the tool-call "
                "markup was malformed and the underlying client couldn't parse it. "
                "Try again, making sure any tool call uses the real tool-calling "
                "mechanism cleanly.",
                read_history, recursion_limit,
            )
            attempt += 1
            continue

        round_msgs = result["messages"][attempt_start:]
        all_round_msgs.extend(round_msgs)
        new_tool_msgs = [m for m in round_msgs if isinstance(m, ToolMessage)]
        round_final = round_msgs[-1] if round_msgs else None
        round_final_text = str(round_final.content) if isinstance(round_final, AIMessage) else ""

        if hit_limit_this_round:
            if attempt == max_attempts - 1:
                return StageResult(
                    final_text=round_final_text, round_msgs=round_msgs, all_round_msgs=all_round_msgs,
                    tokens_in=tokens_in, tokens_out=tokens_out, llm_calls=llm_calls,
                    gen_duration_ms=gen_duration_ms, attempts_used=attempt + 1,
                    hit_recursion_limit=True,
                )
            verdict = {"relevant": False, "reason": f"ran out of its {recursion_limit}-step budget mid-investigation"}
            log_event("stage_verdict", stage=stage_name, **verdict)
            round_digests.append(_stage_digest(round_msgs, verdict))
            payload, config, emitted = _seed_retry(
                original_messages, round_digests,
                f"You ran out of your {recursion_limit}-step budget before finishing. "
                "The 'explored' line above lists the EXACT files AND line ranges "
                "you already saw (e.g. 'file.php:170-190') — not just file names. "
                "Do NOT call read_file/grep_search again on a range you "
                "already have; if you need to re-check something, use exactly what "
                "you already learned about it from the digest instead of "
                "re-fetching it. Spend this attempt on genuinely NEW files/ranges, "
                "or — if you already saw enough to answer — stop investigating "
                "entirely and produce your final answer now.",
                read_history, recursion_limit,
            )
            attempt += 1
            continue

        if hit_overflow_this_round:
            # Тот же приём, что hit_limit_this_round выше — _stream_round
            # перехватил реальный 400 "exceeds the available context size"
            # (compaction.py's проактивный гейт — chars//4 эвристика — не
            # сработал заранее, например для больших лог-дампов) ровно как
            # GraphRecursionError, так что round_msgs всё ещё содержит все
            # успешно завершённые шаги до отказавшего вызова модели.
            # Дайджестим и ретраим с чистого, маленького payload вместо
            # того, чтобы терять весь ход целиком.
            if attempt == max_attempts - 1:
                return StageResult(
                    final_text=round_final_text, round_msgs=round_msgs, all_round_msgs=all_round_msgs,
                    tokens_in=tokens_in, tokens_out=tokens_out, llm_calls=llm_calls,
                    gen_duration_ms=gen_duration_ms, attempts_used=attempt + 1,
                    hit_context_overflow=True,
                )
            verdict = {"relevant": False, "reason": "the request grew too large for the model's context window mid-investigation"}
            log_event("stage_verdict", stage=stage_name, **verdict)
            round_digests.append(_stage_digest(round_msgs, verdict))
            payload, config, emitted = _seed_retry(
                original_messages, round_digests,
                "Your last request grew too large for the model's context window "
                "and was rejected before it could even run. The digest above "
                "lists WHERE you already looked — only re-call one of them if "
                "you actually need to see its content again. Be more selective "
                "this time: prefer narrower/scoped tool calls (grep a specific "
                "pattern instead of dumping a whole log, read a line range "
                "instead of a whole file) over broad dumps.",
                read_history, recursion_limit,
            )
            attempt += 1
            continue

        if not new_tool_msgs and _leaked_tool_call_syntax(round_final_text):
            # Модель иногда генерирует невалидную разметку вызова тула
            # вместо настоящего structured tool_calls (см. delegate_tool.py/
            # agent.py) — парсим её сами и выполняем напрямую вместо того,
            # чтобы жечь оставшиеся попытки на одной и той же генерации.
            leaked_calls = _parse_leaked_tool_calls(round_final_text)
            if leaked_calls:
                result_parts = []
                for call in leaked_calls:
                    if on_event:
                        await on_event({"type": "tool_start", "name": call["name"], "args": call["args"], "stage": stage_name})
                    call_result = await _execute_leaked_tool_call(tools_by_name, call["name"], call["args"])
                    if on_event:
                        await on_event({"type": "tool_end", "name": call["name"], "result": call_result[:2000], "stage": stage_name})
                    result_parts.append(f"`{call['name']}` result:\n{call_result}")
                verdict = {
                    "relevant": False,
                    "reason": "the model's tool-call markup didn't parse as a real tool call, so it was run directly instead of retried",
                }
                log_event("stage_verdict", stage=stage_name, **verdict)
                if attempt == max_attempts - 1:
                    break
                round_digests.append(_stage_digest(round_msgs, verdict))
                payload, config, emitted = _seed_retry(
                    original_messages, round_digests,
                    "Your tool-call markup didn't parse as a real tool call, so it "
                    "was executed directly instead — here are the real results. "
                    "Continue from them, don't repeat the same call:\n\n"
                    + "\n\n".join(result_parts),
                    read_history, recursion_limit,
                )
                attempt += 1
                continue
            verdict = {
                "relevant": False,
                "reason": "the model generated malformed tool-call markup as plain text instead of a real structured tool call — no tool actually ran this round",
            }
            log_event("stage_verdict", stage=stage_name, **verdict)
            if attempt == max_attempts - 1:
                break
            round_digests.append(_stage_digest(round_msgs, verdict))
            payload, config, emitted = _seed_retry(
                original_messages, round_digests,
                "Your last message contained malformed tool-call markup as part of "
                "its text instead of an actual tool call — no tool ran. Call the "
                "tool using the real tool-calling mechanism, not text in your "
                "message content.",
                read_history, recursion_limit,
            )
            attempt += 1
            continue

        verdict = await _call_verdict_fn(verdict_fn, round_msgs, new_tool_msgs, round_final_text)
        log_event("stage_verdict", stage=stage_name, **verdict)
        if on_event and not verdict["relevant"]:
            await on_event({"type": "self_heal_reject", "reason": verdict["reason"], "kind": verdict.get("kind"), "stage": stage_name})
        if verdict["relevant"]:
            break

        # Punt-to-user rescue: раунд закончился текстовым вопросом вместо
        # настоящего ask_user — открываем настоящий диалог с этим же
        # вопросом вместо того, чтобы жечь попытку на "вызови тул как надо".
        # MAX_SELF_HEAL_ASKS cap — without it a model that keeps punting to
        # the user forever would loop indefinitely: each punt does
        # `max_attempts += 1; attempt += 1`, so the gap between attempt and
        # max_attempts never closes and `attempt == max_attempts - 1` never
        # fires. mcp_agent/agent.py's legacy self-heal loop already caps
        # this the same way (self_heal_asks_used < MAX_SELF_HEAL_ASKS) —
        # this was missing here since run_stage was extracted from it.
        if (
            ("?" in round_final_text or "？" in round_final_text)
            and not _called_ask_user(new_tool_msgs)
            and self_heal_asks_used < MAX_SELF_HEAL_ASKS
        ):
            self_heal_asks_used += 1
            shape = await _extract_ask_user_shape(judge_model, round_final_text)
            if on_event:
                await on_event({"type": "tool_start", "name": "ask_user", "args": shape, "stage": stage_name})
            answer = await ask_user_question(shape["question"], shape["options"], shape["recommended"])
            if on_event:
                await on_event({"type": "tool_end", "name": "ask_user", "result": answer[:2000], "stage": stage_name})
            round_digests.append(f"- asked user: {shape['question']!r}\n- user answered: {answer!r}")
            payload = {"messages": [HumanMessage(content=f"The user's answer: {answer}")]}
            max_attempts += 1  # гарантированный ход на использование ответа, не засчитывается как провал
            attempt += 1
            continue

        if attempt == max_attempts - 1:
            break

        guidance = await _call_guidance_fn(guidance_fn, verdict, round_msgs, new_tool_msgs, round_final_text)
        round_digests.append(_stage_digest(round_msgs, verdict))
        payload, config, emitted = _seed_retry(
            original_messages, round_digests,
            f"The previous tool results don't answer the task (reason: {verdict['reason']}). {guidance}",
            read_history, recursion_limit,
        )
        attempt += 1

    final = result["messages"][-1] if result and result.get("messages") else None
    final_text = str(final.content) if isinstance(final, AIMessage) else ""
    return StageResult(
        final_text=final_text, round_msgs=round_msgs,
        all_round_msgs=all_round_msgs,
        tokens_in=tokens_in, tokens_out=tokens_out, llm_calls=llm_calls,
        gen_duration_ms=gen_duration_ms, attempts_used=attempt + 1,
        verdict=verdict,
    )


def _called_ask_user(new_tool_msgs: list) -> bool:
    return any(m.name == "ask_user" for m in new_tool_msgs)


async def _call_verdict_fn(verdict_fn, round_msgs, new_tool_msgs, round_final_text) -> dict:
    result = verdict_fn(round_msgs, new_tool_msgs, round_final_text)
    if hasattr(result, "__await__"):
        result = await result
    return result


async def _call_guidance_fn(guidance_fn, verdict, round_msgs, new_tool_msgs, round_final_text) -> str:
    result = guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text)
    if hasattr(result, "__await__"):
        result = await result
    return result


def _stage_digest(round_msgs: list, verdict: dict) -> str:
    """Тот же формат, что _summarize_round в agent.py (explored/changed/ran/
    diffed + rejected reason) — сюда не импортирован напрямую, потому что
    agent.py импортирует stage_runner.py в перспективе (легаси cutover, см.
    план), а не наоборот; логика достаточно короткая, чтобы держать копию
    здесь не было накладно, но при правке ОБЕИХ копий синхронно проверять
    вторую (agent.py:_summarize_round)."""
    from mcp_agent.agent import _summarize_round
    return _summarize_round(round_msgs, verdict)


def _seed_retry(
    original_messages: list, round_digests: list[str], correction_text: str,
    read_history: dict, recursion_limit: int,
) -> tuple[dict, dict, int]:
    """Аналог agent.py:_start_next_attempt, обобщённый на произвольный
    recursion_limit (у каждой роли свой, см. mcp_agent/roles.py) — та же
    дайджест-вместо-полной-истории механика, тот же сброс read_history по
    той же причине (см. докстринг оригинала)."""
    read_history.clear()
    new_config = {"configurable": {"thread_id": _next_thread_id()}, "recursion_limit": recursion_limit}
    digest_block = "\n\n".join(f"Attempt {i + 1} summary:\n{d}" for i, d in enumerate(round_digests))
    payload = {"messages": [
        *original_messages,
        HumanMessage(content=(
            f"(Continuing after {len(round_digests)} earlier attempt(s) — summarized below "
            "instead of replayed in full; the underlying files/repo state are unchanged, "
            "re-read a file/diff if you need its exact current content.)\n\n"
            f"{digest_block}\n\n{correction_text}"
        )),
    ]}
    return payload, new_config, 0
