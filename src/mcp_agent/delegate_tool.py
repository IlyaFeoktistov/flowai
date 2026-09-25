"""
agent / agent_result — sub-agents in the Claude Code style (types and custom
agents: mcp_agent/subagents.py), replacing the old read-only `delegate`.

Every sub-agent runs on the SAME resident model object as the main agent —
never a second model instance: this machine's RAM/VRAM holds one copy of the
chat model, not two. What varies is how many sub-agents may generate at the
same time — settings.parallel_slots (default 1), which is also the number of
llama-server slots (expert_streaming.py's -np): with 1, runs are strictly
sequential and a background run simply queues behind the main agent's own
requests on the one model (slower, never more memory); with N, up to N run
at once and the model's throughput is shared between them.

A foreground call blocks the calling agent until the report is back, like a
normal tool. run_in_background=True returns an agent_id immediately — the
main agent keeps working and collects the report with agent_result().

Writing sub-agents (general, or a custom agent whose tools include writes)
use the main agent's already-wrapped tools (snapshot before write, fresh-read
check) and get the permission dialogs via writer_middleware — the same
approvals the user would see for the main agent.

compact_research (_CompactResearchMiddleware) per run keeps a long
investigation from overflowing the window; is_context_overflow_error is the
fallback when even that isn't enough (raw read_file/grep_search results are
never compacted).
"""
import asyncio
import contextvars
import uuid
from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from mcp_agent.ask_user_tool import _ToolErrorGuardMiddleware
from mcp_agent.compaction import _CompactResearchMiddleware, _summarize_research, is_context_overflow_error
from mcp_agent.message_utils import _DedupeToolResultsMiddleware, _tool_artifact_diagnostics, _tool_artifact_diff, _tool_text
from mcp_agent.model_config import DEBUG, DELEGATE_RECURSION_LIMIT, TOOL_OUTPUT_CHAR_CAP
from mcp_agent.roles import MAIN_INVESTIGATION_TOOL_NAMES
from mcp_agent.self_heal import (
    _execute_leaked_tool_call,
    _leaked_tool_call_syntax,
    _parse_leaked_tool_calls,
)
from mcp_agent.tool_wrappers import _dedupe_read_tool
from mcp_agent.web_read_tool import build_web_read_tool
from mcp_agent import work_mode
from mcp_agent.subagents import NEVER_FOR_SUBAGENTS, discover_agents
import settings
from ui.console import debug_print

# С достаточно большим набором read-only тулов (например 25) модель может
# не сформировать настоящий structured tool_calls, а слить его текстом в
# content ("<function=read_text_file><parameter=path>...") — тот же баг, из-за
# которого в agent.py/self_heal.py вообще есть _leaked_tool_call_syntax.
# create_agent сам не отличает эту утечку от настоящего финального ответа
# (AIMessage без tool_calls — для графа это всегда "конец хода"), так что
# без этой обработки delegate в половине случаев возвращал бы наружу кусок
# псевдокода вместо реального ответа. Число попыток восстановления
# ОГРАНИЧЕНО (в отличие от MAX_ATTEMPTS в agent.py) — это дешёвый прямой
# ретрай, а не полноценный self-heal с judge/ask_user, ему не нужна такая
# же щедрость.
_MAX_LEAK_RECOVERIES = 2

# Allowlist по ИМЕНИ тула, а не "всё, что не в TOOLS_REQUIRING_APPROVAL" — так
# подмножество не меняется молча, если кто-то добавит новый read-only-с-виду
# тул в approval-неймспейс не подумав про delegate. mcp_agent/roles.py:
# MAIN_INVESTIGATION_TOOL_NAMES — фиксированный (без per-turn флагов
# router.py, которых у основного агента нет) read-only+web набор, БЕЗ shell —
# держит верным собственный системный промпт этого файла ниже ("no shell").
_ALLOWED_TOOLS = frozenset(MAIN_INVESTIGATION_TOOL_NAMES)

# Промпт может дважды явно советовать звать delegate для многофайлового
# расследования, а модель всё равно не вызывает его ни разу за несколько
# попыток, просто читая файлы вручную до потери всякого бюджета (наблюдался
# ход на 37 минут, 106 вызовов тулов, 2 258 084 входных токена, 0 правок).
# Совет текстом в системном промпте на длинном ходу не работает — нужна
# детерминированная проверка, а не ещё одна просьба.
#
# _DelegateNudgeMiddleware считает read-only "разведочные" tool-вызовы
# ПРЯМО В ХОДЕ раунда (awrap_model_call — перед КАЖДЫМ обращением к модели,
# не только по итогам всего хода, как self_heal) и, если их накопилось
# больше порога, а delegate так и не позвали, вставляет напоминание в
# историю ОДИН раз (метка _NUDGE_MARKER не даёт повторить на следующем же
# вызове). Это не запрет — модель всё ещё может проигнорировать и читать
# дальше, но теперь это явное решение, а не то, что до неё вообще не
# доходило само по себе среди остального текста системного промпта.
_EXPLORATION_TOOL_NAMES = {"read_file", "grep_search", "glob_search", "search_code_semantic"}
_DELEGATE_NUDGE_THRESHOLD = 12
_NUDGE_MARKER = "you've made a lot of read/search calls in this investigation"


class _DelegateNudgeMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request, handler):
        if not settings.get("delegate_nudge_enabled"):
            return await handler(request)
        messages = request.messages
        already_nudged = any(
            isinstance(m, HumanMessage) and _NUDGE_MARKER in str(m.content).lower()
            for m in messages
        )
        if already_nudged:
            return await handler(request)

        used_delegate = any(isinstance(m, ToolMessage) and m.name in _SUBAGENT_TOOLS for m in messages)
        if used_delegate:
            return await handler(request)

        explore_count = sum(
            1 for m in messages if isinstance(m, ToolMessage) and m.name in _EXPLORATION_TOOL_NAMES
        )
        if explore_count < _DELEGATE_NUDGE_THRESHOLD:
            return await handler(request)

        if DEBUG:
            debug_print(f"[dim][MCP-AGENT] delegate nudge injected after {explore_count} exploration calls[/]")

        nudge = HumanMessage(content=(
            f"(System note: you've made a lot of read/search calls in this "
            f"investigation — {explore_count} so far — without ever using "
            "a sub-agent. If there's still more ground to cover, STOP reading "
            "files yourself: call agent(subagent_type=\"explore\") with a "
            "complete, self-contained description of what's left to "
            "investigate, and continue from its report instead of reading "
            "more files manually.)"
        ))
        return await handler(request.override(messages=list(messages) + [nudge]))


# Позволяет delegate() отдавать СВОИ tool_start/tool_end наружу в ЖИВОМ
# режиме, хотя сам сабагент собирается один раз на всю сессию (см.
# build_delegate_tool ниже) и не получает on_event как параметр — on_event
# у каждого хода СВОЙ (agent.py:stream_chat), а delegate-тул кешируется
# вместе с остальным агентом (agent_builder.py:_agent_cache). ContextVar,
# не глобальная переменная — agent.py выставляет её ПЕРЕД стартом хода и
# сбрасывает после (contextvars переживают await и копируются в дочерние
# asyncio.Task, так что значение доживает до реального вызова delegate()
# глубоко внутри astream()-цикла LangGraph, даже через несколько уровней
# await). default=None — вызывающий код без активного хода (тесты,
# run_cli.py без стрима) просто не получает живых событий, тул всё равно
# работает и возвращает финальный текст как раньше.
current_on_event: ContextVar = ContextVar("delegate_on_event", default=None)

# Тулы, чья работа идёт ВНУТРИ отдельного sub_agent.astream()/ainvoke() на
# том же объекте `model`, что и внешняя роль/основной агент — delegate
# единственный такой сейчас. Пока delegate работает, внешний
# agent.astream(stream_mode=["values", "messages"]) может выдавать ДЕСЯТКИ
# answer_start подряд без единого answer_chunk/answer_end — похоже, токен-
# стрим вложенного sub_agent просачивается в
# общий "messages" канал LangGraph, раз оба используют один и тот же
# ChatOllama-инстанс в одном async-контексте (точная причина на стороне
# LangGraph не установлена — это защитный фикс симптома, не первопричины).
# Пока delegate не закрылся своим tool_end, answer_*/thinking_* от ВНЕШНЕГО
# потока глушим — они не принадлежат ему, а tool_start/tool_end самого
# delegate (плюс его собственные "delegate → ..." из _run_subagent_streaming
# выше) НЕ глушатся, так что пользователь всё равно видит, что происходит,
# просто без рваных чужих текстовых фрагментов поверх. Общее для основного
# stream_chat (agent.py) и пайплайна (stage_runner.py) — оба реально зовут
# delegate/делят один `model`.
_SUBAGENT_TOOLS = frozenset({"agent", "agent_result"})


def _suppress_during_subagent_tools(on_event):
    if on_event is None:
        return None
    state = {"pending": False}

    async def wrapped(event: dict):
        t = event.get("type")
        name = event.get("name")
        if t == "tool_start" and name in _SUBAGENT_TOOLS:
            state["pending"] = True
        if state["pending"] and t in (
            "answer_start", "answer_chunk", "answer_end",
            "thinking_start", "thinking_chunk", "thinking_end",
        ):
            return  # чужой токен-стрим — не принадлежит внешнему потоку
        await on_event(event)
        if t == "tool_end" and name in _SUBAGENT_TOOLS:
            state["pending"] = False

    return wrapped


async def _run_subagent_streaming(sub_agent, conversation: list, config: dict, label: str = "") -> tuple[dict, int, int, bool, int]:
    """Замена sub_agent.ainvoke(...) с тем же возвращаемым значением (dict с
    ключом "messages"), но эмитящая tool_start/tool_end наружу через
    current_on_event ПО МЕРЕ того, как сабагент реально вызывает свои тулы —
    "delegate → grep_search" и т.п., а не молчание на весь срок ainvoke()
    (без этого пользователь не видит вообще ничего, что происходит, пока
    delegate работает). stream_mode="values" — снимок полного состояния
    графа после каждого шага, а не токен-стрим модели (тот канал стриминга
    ДРУГОЙ, см. stage_runner.py:_SUBAGENT_TOOLS про утечку чужих
    answer_start во внешний stream_mode=["values","messages"] при *любом*
    вызове сабагента на том же shared model — использование astream здесь
    для ЗНАЧЕНИЙ графа, не для токенов, не трогает эту утечку сильнее, чем
    раньше делал ainvoke.

    Возвращает (final_state, tokens_in, tokens_out, hit_recursion_limit,
    peak_context) — peak_context: максимальный prompt+reply ОДНОГО вызова,
    то есть насколько реально заполнялось окно сабагента (tokens_in — сумма
    по всем вызовам, каждый из которых заново шлёт всю историю).
    GraphRecursionError ловится ЗДЕСЬ, а не в delegate() (как раньше) —
    LangGraph поднимает его НА СТАРТЕ шага, который превысил бы лимит (тот
    же факт, что agent.py:_stream_round уже использует для внешнего
    агента), то есть final_state к этому моменту уже содержит ПОЛНОЕ
    состояние всех успешно завершённых шагов — если ловить исключение
    снаружи этой функции, эти накопленные messages терялись бы вместе с
    ним, и delegate() не смог бы попросить модель подвести итог того, что
    уже нашлось (см. delegate() ниже).

    Возвращает (final_state, tokens_in, tokens_out) — the sub-agent's OWN
    model calls happen entirely inside this astream loop, invisible to the
    outer stream_chat's own tokens_in/tokens_out accounting (that scans ITS
    OWN agent.astream() messages — delegate's internal AIMessages live in a
    completely separate graph/state, so a delegate call that burned real
    tokens investigating showed up in the visible running token count as
    zero). Every AIMessage's usage_metadata (populated by the underlying
    ChatOllama call, same field the outer loop already reads) is summed
    here; delegate() below forwards the total to the outer loop via a
    "tokens_add" event on the same current_on_event channel already used
    for tool_start/tool_end."""
    on_event = current_on_event.get()
    prev_len = 0
    final_state: dict = {}
    tokens_in = tokens_out = 0
    peak_context = 0
    hit_recursion_limit = False
    try:
        async for state in sub_agent.astream({"messages": conversation}, config, stream_mode="values"):
            final_state = state
            msgs = state.get("messages") or []
            for m in msgs[prev_len:]:
                if isinstance(m, AIMessage) and m.usage_metadata:
                    call_in = m.usage_metadata.get("input_tokens", 0) or 0
                    call_out = m.usage_metadata.get("output_tokens", 0) or 0
                    tokens_in += call_in
                    tokens_out += call_out
                    peak_context = max(peak_context, call_in + call_out)
                if on_event is None:
                    continue
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        await on_event({
                            "type": "tool_start",
                            "name": f"agent → {tc['name']}",
                            "args": tc.get("args", {}),
                            "id": tc.get("id"),
                            "agent": label,
                        })
                elif isinstance(m, ToolMessage):
                    # Match the model's own output cap instead of a smaller flat cutoff — see agent.py's tool_end.
                    event = {
                        "type": "tool_end",
                        "name": f"agent → {m.name}",
                        "result": _tool_text(m.content)[:TOOL_OUTPUT_CHAR_CAP],
                        "id": m.tool_call_id,
                        "agent": label,
                    }
                    # A writing sub-agent's edits show their diff too.
                    diff = _tool_artifact_diff(getattr(m, "artifact", None))
                    if diff is not None:
                        event["diff"] = diff
                    diagnostics = _tool_artifact_diagnostics(getattr(m, "artifact", None))
                    if diagnostics is not None:
                        event["diagnostics"] = diagnostics
                    await on_event(event)
            prev_len = len(msgs)
    except GraphRecursionError:
        hit_recursion_limit = True
    return final_state, tokens_in, tokens_out, hit_recursion_limit, peak_context


def build_delegate_tool(
    model, tools: list, raw_read_file_tool=None, judge_model=None, *,
    repo_path: str | None = None, tracked_read_file_tool=None, writer_middleware=None,
):
    """Builds the `agent` + `agent_result` tools as closures over the model
    and tools this session already has — no second weight load, no new MCP
    server. Returns [agent, agent_result].

    tools — the main agent's tool objects (already wrapped: snapshot before
    write, fresh-read check, read-cache invalidation — so a writing
    sub-agent's edits get exactly the same safety net as the main agent's).
    read_file is the one exception: every sub-agent run gets its OWN
    dedupe history (raw_read_file_tool / tracked_read_file_tool re-wrapped
    per run) — sharing the main agent's would answer the sub-agent's first
    read of a file with "you already read this", pointing at content it
    never saw (a different conversation). tracked_read_file_tool also
    records the read in the fresh-read tracker, which write_file/edit_file
    require — used for agents that may write.

    writer_middleware — factory for the middleware a sub-agent that can
    mutate things needs (permission prompts, out-of-project write check,
    plugin hooks, plan-mode guard); supplied by agent_builder, which owns
    those classes (importing them here would be circular).

    judge_model — the same one the main agent judges its rounds with, used
    only for _CompactResearchMiddleware."""
    available = {t.name for t in tools} | {"web_read"}
    background: dict[str, dict] = {}

    def _tools_for(spec) -> list:
        names = (available if spec.tools is None else spec.tools & available) - NEVER_FOR_SUBAGENTS
        own_read_history: dict = {}
        selected = [t for t in tools if t.name in names and t.name != "read_file"]
        if "read_file" in names:
            base = tracked_read_file_tool if (spec.can_mutate(available) and tracked_read_file_tool is not None) else raw_read_file_tool
            if base is None:
                base = next((t for t in tools if t.name == "read_file"), None)
            if base is not None:
                selected.append(_dedupe_read_tool(base, own_read_history))
        if "web_read" in names:
            selected.append(build_web_read_tool(model))
        return selected

    def _build_sub_agent(spec):
        # Built per run (cheap: no model load, tools already exist) — each
        # run needs its own compaction cache and read history, and runs may
        # now overlap (parallel_slots > 1, or background runs).
        sub_tools = _tools_for(spec)
        compact_research = _CompactResearchMiddleware(judge_model)
        middleware = [_ToolErrorGuardMiddleware()]
        if spec.can_mutate(available) and writer_middleware is not None:
            middleware += writer_middleware()
        middleware += [_DedupeToolResultsMiddleware(), compact_research]
        sub_agent = create_agent(
            model, sub_tools, system_prompt=spec.system_prompt,
            middleware=middleware, checkpointer=InMemorySaver(),
        )
        return sub_agent, {t.name: t for t in sub_tools}

    async def _run_agent(spec, task: str, label: str) -> str:
        sub_agent, sub_tools_by_name = _build_sub_agent(spec)
        conversation = [HumanMessage(content=task)]
        final_text = ""
        tokens_in_total = tokens_out_total = 0
        peak_context = 0

        async def _emit_token_usage() -> None:
            # Reported once, right before the run returns, so the outer
            # counter jumps by this run's real total when its result lands.
            on_event = current_on_event.get()
            if on_event and (tokens_in_total or tokens_out_total):
                await on_event({
                    "type": "tokens_add",
                    "tokens_in": tokens_in_total, "tokens_out": tokens_out_total,
                    "peak_context": peak_context,
                })

        # До _MAX_LEAK_RECOVERIES+1 попыток: каждая — свежий invoke графа со
        # своим ПОЛНЫМ recursion_limit'ом (не растягиваем один и тот же
        # исчерпанный лимит на ретраи). thread_id новый каждый раз — мы сами
        # несём всю историю в conversation, а не полагаемся на checkpointer
        # между попытками.
        for _ in range(_MAX_LEAK_RECOVERIES + 1):
            config = {
                "configurable": {"thread_id": f"agent-{uuid.uuid4().hex[:8]}"},
                "recursion_limit": DELEGATE_RECURSION_LIMIT,
            }
            try:
                result, round_tokens_in, round_tokens_out, hit_recursion_limit, round_peak = await _run_subagent_streaming(
                    sub_agent, conversation, config, label
                )
                tokens_in_total += round_tokens_in
                tokens_out_total += round_tokens_out
                peak_context = max(peak_context, round_peak)
            except Exception as e:
                # Same detector agent.py:_stream_round already uses for the
                # outer agent — compact_research above shrinks the odds of
                # this a lot, but doesn't eliminate them (raw sticky content
                # — read_file/grep_search/glob_search, see compaction.py:
                # STICKY_TOOL_NAMES — is never compacted, so several
                # genuinely large files alone can still do it). Everything
                # ELSE (a real network/model failure) must keep propagating
                # as before, not get silently swallowed as if it were this.
                if not is_context_overflow_error(e):
                    raise
                await _emit_token_usage()
                return (
                    "Sub-agent's own investigation grew too large for the "
                    "model's context window before it could finish (too many "
                    "large files/wide searches in one investigation). Split "
                    "the task into narrower agent() calls covering one "
                    "part of the investigation each, or investigate the "
                    "remainder yourself."
                )

            messages = result.get("messages") or []

            if hit_recursion_limit:
                # Steps ran out mid-investigation — the sub-agent never got
                # to write its own considered final answer (see
                # the sub-agent's system prompt). _summarize_research (same
                # digest-writer compact_research above uses periodically)
                # gets ONE extra shot at turning whatever raw
                # read_file/grep_search/... results DID accumulate into a
                # dense, file:line-cited digest instead of handing back
                # nothing — this is the ONLY path that calls it directly
                # rather than through the middleware, exactly because
                # there's no next round left for the middleware to run in.
                digest = await _summarize_research(judge_model, messages) if messages else ""
                await _emit_token_usage()
                if digest:
                    return (
                        f"Sub-agent used its full {DELEGATE_RECURSION_LIMIT}-step "
                        "budget without reaching its own considered final answer "
                        "— but here's a dense summary of what it found before "
                        f"running out:\n\n{digest}\n\nTreat this as partial and "
                        "unverified (the sub-agent itself never confirmed these "
                        "are its real conclusions). Call agent again with a "
                        "narrower task to fill the gaps, or investigate the "
                        "remainder yourself."
                    )
                # _summarize_research failed too (fail-open, see its own
                # except-clause) — no digest to fall back on, same bare
                # message as before this existed.
                return (
                    f"Sub-agent used its full {DELEGATE_RECURSION_LIMIT}-step "
                    "budget without reaching a final answer. Either delegate "
                    "again with a narrower task, or investigate the remainder "
                    "yourself — don't hand the exact same task over again."
                )

            final = messages[-1] if messages else None
            final_text = _tool_text(final.content) if isinstance(final, AIMessage) else ""

            if not final_text:
                await _emit_token_usage()
                return "Sub-agent finished without producing a final answer."
            if not _leaked_tool_call_syntax(final_text):
                await _emit_token_usage()
                return final_text

            # Модель слила вызов тула текстом вместо настоящего structured
            # tool_calls (см. модульный docstring) — create_agent видит в
            # этом обычный финальный ответ без вызовов и останавливается.
            # Выполняем разобранный вызов напрямую и продолжаем с реальным
            # результатом вместо того, чтобы отдать наружу псевдокод.
            leaked_calls = _parse_leaked_tool_calls(final_text)
            if not leaked_calls:
                conversation = conversation + [
                    AIMessage(content=final_text),
                    HumanMessage(content=(
                        "Your last message contained malformed tool-call "
                        "markup (like '<function=...>' or '<tool_call>...') "
                        "as plain text instead of an actual tool call — no "
                        "tool ran. Call the tool using the real tool-calling "
                        "mechanism, not text written into your message "
                        "content."
                    )),
                ]
                continue

            result_parts = []
            for call in leaked_calls:
                call_result = await _execute_leaked_tool_call(sub_tools_by_name, call["name"], call["args"])
                result_parts.append(f"`{call['name']}` result:\n{call_result}")
            conversation = conversation + [
                AIMessage(content=final_text),
                HumanMessage(content=(
                    "Your tool-call markup didn't parse as a real tool call, "
                    "so it was executed directly instead — here are the real "
                    "results. Continue from them, don't repeat the same "
                    "call:\n\n" + "\n\n".join(result_parts)
                )),
            ]

        await _emit_token_usage()
        return (
            "Sub-agent kept generating malformed tool-call markup instead of "
            "real tool calls after multiple recovery attempts — giving up. "
            f"Its last output was:\n\n{final_text}"
        )

    async def _run_limited(spec, task: str, label: str) -> str:
        async with _slot_semaphore():
            return await _run_agent(spec, task, label)

    def _resolve(subagent_type: str):
        agents = discover_agents(repo_path, _ALLOWED_TOOLS)
        wanted = (subagent_type or "explore").strip()
        # Case-insensitive + Claude Code's own type names, so skills written
        # for Claude Code ("general-purpose", "Explore", "Plan") just work.
        wanted = _TYPE_ALIASES.get(wanted.lower(), wanted)
        spec = agents.get(wanted) or next((a for n, a in agents.items() if n.lower() == wanted.lower()), None)
        if spec is None:
            return None, f"Error: unknown subagent_type {subagent_type!r}. Available: {', '.join(agents)}."
        if spec.can_mutate(available) and work_mode.current_work_mode.get() == work_mode.PLAN:
            return None, (
                f"Error: '{spec.name}' can modify files, and this turn is in PLAN mode (read-only). "
                "Use explore or plan instead, and put the change into your plan."
            )
        return spec, None

    @tool
    async def agent(
        description: str, prompt: str, subagent_type: str = "explore",
        run_in_background: bool = False, model: str = "",
    ) -> str:
        """Launch a sub-agent on the same model with its OWN context window and
        step budget. `subagent_type` picks what it can do — the available
        types and when to use each are listed in the system prompt (explore:
        read-only investigation, plan: read-only implementation plan,
        general: does a whole task including edits). The sub-agent does NOT
        see this conversation and can't ask the user anything: `prompt` must
        be a complete, self-contained task with all the context it needs.
        `description` — 3-5 words naming the task (shown to the user).

        Returns the sub-agent's final report. With run_in_background=true it
        returns immediately with an agent_id and you keep working; get the
        report later with agent_result(agent_id) — always collect it before
        your final answer. Sub-agents share one model: how many run at the
        same time is limited by the user's settings, extra ones wait.
        `model` is accepted for compatibility and ignored — every sub-agent
        runs on the one local model."""
        spec, error = _resolve(subagent_type)
        if error:
            return error
        label = " ".join((description or prompt).split())[:80]
        if not run_in_background:
            result = await _run_limited(spec, prompt, label)
            return f"[agent {spec.name}: {label}]\n{result}"

        # A fresh Context, not a copy of the caller's: LangChain passes the
        # parent run's callbacks through contextvars, and inheriting them
        # would stream this sub-agent's tokens into the MAIN agent's output
        # while the main agent keeps generating. Only the vars the run
        # actually needs are carried over.
        agent_id = uuid.uuid4().hex[:6]
        ctx = contextvars.Context()
        ctx.run(current_on_event.set, current_on_event.get())
        ctx.run(work_mode.current_work_mode.set, work_mode.current_work_mode.get())
        task = asyncio.get_running_loop().create_task(_run_limited(spec, prompt, label), context=ctx)
        background[agent_id] = {"task": task, "type": spec.name, "label": label}
        return (
            f"Started background agent {agent_id} ({spec.name}: {label}). Keep working; "
            f"call agent_result(agent_id=\"{agent_id}\") to get its report before your final answer."
        )

    @tool
    async def agent_result(agent_id: str, wait: bool = True) -> str:
        """Get the report of a background agent started with
        agent(..., run_in_background=true). wait=true blocks until it
        finishes; wait=false returns right away saying whether it's done."""
        entry = background.get(agent_id.strip())
        if entry is None:
            known = ", ".join(background) or "none"
            return f"Error: no background agent {agent_id!r}. Known: {known}."
        task = entry["task"]
        if not task.done() and not wait:
            return f"Agent {agent_id} ({entry['type']}: {entry['label']}) is still running."
        try:
            result = await asyncio.shield(task)
        except Exception as e:
            result = f"Agent failed: {e}"
        background.pop(agent_id, None)
        return f"[agent {entry['type']}: {entry['label']}]\n{result}"

    return [agent, agent_result]


_TYPE_ALIASES = {"general-purpose": "general", "general_purpose": "general"}


# One semaphore per process, sized by settings.parallel_slots — how many
# sub-agents may generate at once on the single resident model. Rebuilt only
# when the setting changes and nothing holds it.
_semaphore: asyncio.Semaphore | None = None
_semaphore_size = 0


def _slot_semaphore() -> asyncio.Semaphore:
    global _semaphore, _semaphore_size
    from expert_streaming import MAX_PARALLEL_SLOTS
    size = max(1, min(int(settings.get("parallel_slots") or 1), MAX_PARALLEL_SLOTS))
    if _semaphore is None or (size != _semaphore_size and _semaphore._value == _semaphore_size):
        _semaphore = asyncio.Semaphore(size)
        _semaphore_size = size
    return _semaphore
