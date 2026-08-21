"""
delegate — последовательный сабагент для многофайлового расследования.

Реальная цена того, что ВСЁ расследование идёт в ОДНОМ общем контексте с
ОДНИМ общим бюджетом шагов (RECURSION_LIMIT): при достаточно длинной цепочке
зависимостей (например Job -> Service -> Repository -> Persister ->
UnitStarter) агент может закончить шаги ещё до готового ответа.

Параллельных сабагентов здесь НЕТ и не будет: qwen3-coder:30b — 18 GB
(`ollama list`), WSL этой машины реально даёт процессам ~23 GB RAM
(`free -h`) — второй одновременно загруженный инстанс той же модели (не
говоря о другой) в этот бюджет не помещается. delegate поэтому —
ПОСЛЕДОВАТЕЛЬНЫЙ вызов: он переиспользует уже резидентную модель (тот же
объект `model`, тот же keep_alive/тег, что и у основного агента — просто
ещё один ainvoke, ровно как self_heal.py уже делает лишний вызов на judge
после каждого круга), выполняется полностью, и только потом возвращает
управление основному циклу.

Раньше это было ТОЛЬКО предположением в этом комментарии, ничем не
закреплённым в коде — langgraph/prebuilt/tool_node.py исполняет ВСЕ
tool_calls одного AIMessage через asyncio.gather разом, так что если
модель за один раунд решит вызвать delegate несколько раз (наблюдалось
живьём: 3 одновременных вызова), они реально стартовали бы конкурентно.
own_read_history/compact_research (см. build_delegate_tool ниже) —
ОБЩЕЕ на все вызовы состояние, которое каждый delegate() очищает при
входе, полагаясь именно на строгую последовательность — при реальной
конкурентности один вызов мог стереть кэш другого посреди его работы.
_delegate_lock ниже — настоящее принуждение к тому, что до сих пор было
только описано словами: если модель всё же запустит несколько delegate
разом, они физически дождутся друг друга по очереди (FIFO у asyncio.Lock),
а не тронут общее состояние одновременно.

Тулы у сабагента — READ-ONLY подмножество (_ALLOWED_TOOLS): без bash,
без записи файлов, без git-мутаций. Это осознанное упрощение, а не
временная заглушка — сабагент здесь только для разведки ("как оно
устроено", "где реально определено", "что вызывает что"), не для внесения
изменений. Из этого следует и то, что ему не нужны
HumanInTheLoopMiddleware/ask_user: ни один тул в его наборе не требует
approval, а сам он не может ничего спросить у пользователя — задача (task)
должна быть самодостаточной, сабагент не видит остального разговора.

compact_research (_CompactResearchMiddleware, mcp_agent/compaction.py) —
без неё длинное расследование (много раундов read_file/grep_search/
search_code_semantic) накапливает историю, которую НИКТО не сжимает: у
внешнего агента compact_research есть, у сабагента не было. Многораундовое
расследование без неё реально может упереться в 400 "request exceeds the
available context size" ещё до готового ответа. Тот же judge_model, что и
у внешнего агента (передаётся в build_delegate_tool) — не отдельная
загрузка весов. is_context_overflow_error (та же детекция, что
agent.py:_stream_round) — страховка НА СЛУЧАЙ, если даже компакция не
спасла (см. delegate() ниже): raw sticky-контент (read_file/grep_search/
glob_search — см. compaction.py:STICKY_TOOL_NAMES) компакция принципиально
не трогает, так что чисто от нескольких больших честно прочитанных файлов
упереться в потолок всё ещё можно.
"""
import asyncio
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
from mcp_agent.message_utils import _DedupeToolResultsMiddleware, _tool_text
from mcp_agent.model_config import DEBUG, DELEGATE_RECURSION_LIMIT, TOOL_OUTPUT_CHAR_CAP
from mcp_agent.roles import MAIN_INVESTIGATION_TOOL_NAMES
from mcp_agent.self_heal import (
    _execute_leaked_tool_call,
    _leaked_tool_call_syntax,
    _parse_leaked_tool_calls,
)
from mcp_agent.tool_wrappers import _dedupe_read_tool
from mcp_agent.web_read_tool import build_web_read_tool
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
_ALLOWED_TOOLS = MAIN_INVESTIGATION_TOOL_NAMES

_DELEGATE_SYSTEM_PROMPT = (
    "You are a focused research sub-agent, delegated ONE specific "
    "investigation by another agent. You have READ-ONLY tools — no writes, "
    "no shell, no git mutations — and you cannot ask the user anything, so "
    "treat the task you were given as the ONLY information you have.\n\n"
    "You have your OWN step budget, separate from whoever delegated this to "
    "you — spend it efficiently: don't re-read a file you already read this "
    "session, don't retry a search with a slightly different guessed regex "
    "when the exact string is right there in what you already read, and "
    "prefer lsp (goToDefinition/findReferences) over guessing a symbol's "
    "location by pattern-matching its name.\n\n"
    "When you have enough to answer, STOP and write a complete, concrete "
    "final answer with exact file paths and line numbers, so the caller can "
    "verify without re-reading everything you already read. If you run out "
    "of steps before finishing, say plainly what you found and what's still "
    "unknown — never guess to fill the gap."
)

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

        used_delegate = any(isinstance(m, ToolMessage) and m.name == "delegate" for m in messages)
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
            "delegate. If there's still more ground to cover, STOP reading "
            "files yourself: call delegate with a complete, self-contained "
            "description of what's left to investigate, and continue from "
            "its summary instead of reading more files manually.)"
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
_SUBAGENT_TOOLS = frozenset({"delegate"})


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


async def _run_subagent_streaming(sub_agent, conversation: list, config: dict) -> tuple[dict, int, int, bool]:
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

    Возвращает (final_state, tokens_in, tokens_out, hit_recursion_limit).
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
    hit_recursion_limit = False
    try:
        async for state in sub_agent.astream({"messages": conversation}, config, stream_mode="values"):
            final_state = state
            msgs = state.get("messages") or []
            for m in msgs[prev_len:]:
                if isinstance(m, AIMessage) and m.usage_metadata:
                    tokens_in += m.usage_metadata.get("input_tokens", 0) or 0
                    tokens_out += m.usage_metadata.get("output_tokens", 0) or 0
                if on_event is None:
                    continue
                if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                    for tc in m.tool_calls:
                        await on_event({
                            "type": "tool_start",
                            "name": f"delegate → {tc['name']}",
                            "args": tc.get("args", {}),
                            "id": tc.get("id"),
                        })
                elif isinstance(m, ToolMessage):
                    # Match the model's own output cap instead of a smaller flat cutoff — see agent.py's tool_end.
                    await on_event({
                        "type": "tool_end",
                        "name": f"delegate → {m.name}",
                        "result": _tool_text(m.content)[:TOOL_OUTPUT_CHAR_CAP],
                        "id": m.tool_call_id,
                    })
            prev_len = len(msgs)
    except GraphRecursionError:
        hit_recursion_limit = True
    return final_state, tokens_in, tokens_out, hit_recursion_limit


def build_delegate_tool(model, tools: list, raw_read_file_tool=None, judge_model=None):
    """Собирает delegate как closure над уже поднятыми model/tools этой
    сессии — никакой второй подгрузки весов, никакого нового MCP-сервера.
    Вызывается из agent_builder._build_agent, где и model, judge_model и
    tools уже есть в области видимости. judge_model — тот же самый,
    которым внешний агент судит свои self-heal раунды (_build_chat_model
    с format="json", reasoning=False), не отдельная модель под сабагент —
    используется здесь ТОЛЬКО для _CompactResearchMiddleware, той же
    комбинации, что уже проверена в agent_builder.py:_build_role_agent.

    raw_read_file_tool — read_file ДО того, как agent_builder.py обернул
    его в _dedupe_read_tool с ОБЩИМ read_history внешней роли. Заново
    оборачиваем его тут с СОБСТВЕННЫМ, изолированным read_history —
    delegate — свежий, отдельный sub-agent разговор; если бы он делил
    read_history с внешней ролью, более раннее чтение файла, сделанное
    ВНЕШНИМ агентом (до вызова delegate), заставило бы первое же чтение
    ТОГО ЖЕ пути внутри delegate попасть в "(You already read `path`...
    reuse that earlier result)" — а реального результата, который можно
    бы переиспользовать, у delegate нет: то чтение было в ДРУГОМ
    разговоре. Модели тогда нечем ответить кроме как выдумать анализ,
    выглядящий как разбор файла, который она на самом деле не видела."""
    # Cleared at the start of every delegate() call below (same convention
    # as the outer role's read_history.clear() at the start of every
    # stream_chat) — otherwise this would just move the SAME cross-
    # conversation leak one level down: a SECOND delegate() call later in
    # the same session would hit "(You already read...)" stubs left by the
    # FIRST call's own, now-finished conversation. delegate() calls are
    # strictly sequential (module docstring), never concurrent, so
    # clearing on entry is safe.
    own_read_history: dict = {}
    delegate_tools = [t for t in tools if t.name in _ALLOWED_TOOLS]
    if raw_read_file_tool is not None:
        wrapped_read_file = _dedupe_read_tool(raw_read_file_tool, own_read_history)
        delegate_tools = [wrapped_read_file if t.name == "read_file" else t for t in delegate_tools]
    # web_read (web_read_tool.py) isn't an MCP tool, so the `t.name in
    # _ALLOWED_TOOLS` filter above can never have picked it up from `tools`
    # — added directly instead, using the SAME `model` this sub-agent
    # itself runs on (no second load). _ALLOWED_TOOLS (roles.py:
    # MAIN_INVESTIGATION_TOOL_NAMES) includes "web_read" unconditionally,
    # so this is not gated on anything per-call.
    delegate_tools.append(build_web_read_tool(model))
    delegate_tools_by_name = {t.name: t for t in delegate_tools}
    # See module docstring — without this, a long enough investigation
    # (many read_file/grep_search/search_code_semantic rounds) accumulates
    # an ever-growing, never-compacted history and can hit a real 400
    # "exceeds the available context size" before ever producing an
    # answer. clear_cache() runs at the start of every delegate() call
    # below, same reason own_read_history.clear() does.
    compact_research = _CompactResearchMiddleware(judge_model)
    sub_agent = create_agent(
        model,
        delegate_tools,
        system_prompt=_DELEGATE_SYSTEM_PROMPT,
        middleware=[_ToolErrorGuardMiddleware(), _DedupeToolResultsMiddleware(), compact_research],
        checkpointer=InMemorySaver(),
    )

    # Forces the sequential-only guarantee the module docstring already
    # claimed but never enforced — see there for why a real conflict is
    # possible, not just a wasted VRAM slot. One lock per built delegate
    # tool (i.e. per session), held for the ENTIRE body of a call: if the
    # model fires several delegate calls in one round, they queue here in
    # FIFO order and run one at a time, exactly like a single call would.
    _delegate_lock = asyncio.Lock()

    async def _run_delegate(task: str) -> str:
        own_read_history.clear()
        compact_research.clear_cache()
        conversation = [HumanMessage(content=task)]
        final_text = ""
        tokens_in_total = tokens_out_total = 0

        async def _emit_token_usage() -> None:
            # Reported ONCE, right before delegate() actually returns —
            # not per-round — so the outer loop's running counter jumps by
            # this call's real total exactly when the visible tool_end for
            # "delegate" fires, not piecemeal mid-investigation.
            on_event = current_on_event.get()
            if on_event and (tokens_in_total or tokens_out_total):
                await on_event({
                    "type": "tokens_add",
                    "tokens_in": tokens_in_total, "tokens_out": tokens_out_total,
                })

        # До _MAX_LEAK_RECOVERIES+1 попыток: каждая — свежий invoke графа со
        # своим ПОЛНЫМ recursion_limit'ом (не растягиваем один и тот же
        # исчерпанный лимит на ретраи). thread_id новый каждый раз — мы сами
        # несём всю историю в conversation, а не полагаемся на checkpointer
        # между попытками.
        for _ in range(_MAX_LEAK_RECOVERIES + 1):
            config = {
                "configurable": {"thread_id": f"delegate-{uuid.uuid4().hex[:8]}"},
                "recursion_limit": DELEGATE_RECURSION_LIMIT,
            }
            try:
                result, round_tokens_in, round_tokens_out, hit_recursion_limit = await _run_subagent_streaming(
                    sub_agent, conversation, config
                )
                tokens_in_total += round_tokens_in
                tokens_out_total += round_tokens_out
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
                    "the task into narrower delegate() calls covering one "
                    "part of the investigation each, or investigate the "
                    "remainder yourself."
                )

            messages = result.get("messages") or []

            if hit_recursion_limit:
                # Steps ran out mid-investigation — the sub-agent never got
                # to write its own considered final answer (see
                # _DELEGATE_SYSTEM_PROMPT). _summarize_research (same
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
                        "are its real conclusions). Delegate again with a "
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
                    "yourself — don't delegate the exact same task again."
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
                call_result = await _execute_leaked_tool_call(delegate_tools_by_name, call["name"], call["args"])
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

    @tool
    async def delegate(task: str) -> str:
        """Delegate an open-ended, multi-file investigation to a fresh
        sub-agent with its OWN context window and its OWN step budget,
        separate from this conversation's. Use this instead of digging
        through the codebase yourself when a question needs tracing
        something across MANY files/layers (e.g. "how does X's retry logic
        actually work end to end" spanning a Job -> Service -> Repository ->
        Persister chain) — that kind of investigation can burn most of THIS
        conversation's own step budget on file reads alone, and if THAT
        budget runs out mid-investigation the whole turn is lost with
        nothing to show for it.

        The sub-agent reports back ONE text summary with concrete
        file:line citations — you still decide what to do with it, it does
        not take any action itself. It is READ-ONLY (no writes/bash/git
        mutations) and can't ask the user anything — write `task` as a
        complete, self-contained question with whatever context it needs;
        it does not see the rest of this conversation. Calling this several
        times in one turn is fine — each call runs to completion before the
        next one starts, and each result is labeled with its own task so
        you can tell them apart."""
        async with _delegate_lock:
            result = await _run_delegate(task)
        label = " ".join(task.split())[:80]
        return f"[delegate: {label}]\n{result}"

    return delegate
