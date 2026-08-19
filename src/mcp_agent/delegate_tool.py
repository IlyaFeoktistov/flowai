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

Тулы у сабагента — READ-ONLY подмножество (_ALLOWED_TOOLS): без bash,
без записи файлов, без git-мутаций. Это осознанное упрощение, а не
временная заглушка — сабагент здесь только для разведки ("как оно
устроено", "где реально определено", "что вызывает что"), не для внесения
изменений. Из этого следует и то, что ему не нужны
HumanInTheLoopMiddleware/ask_user: ни один тул в его наборе не требует
approval, а сам он не может ничего спросить у пользователя — задача (task)
должна быть самодостаточной, сабагент не видит остального разговора.
"""
import uuid
from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from mcp_agent.ask_user_tool import _ToolErrorGuardMiddleware
from mcp_agent.message_utils import _DedupeToolResultsMiddleware, _tool_text
from mcp_agent.model_config import DEBUG, DELEGATE_RECURSION_LIMIT
from mcp_agent.roles import LEGACY_INVESTIGATION_TOOL_NAMES
from mcp_agent.self_heal import (
    _execute_leaked_tool_call,
    _leaked_tool_call_syntax,
    _parse_leaked_tool_calls,
)
import settings
from ui.console import console

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
# LEGACY_INVESTIGATION_TOOL_NAMES — фиксированный (без per-turn флагов
# router.py, которых у легаси-агента нет) read-only+web набор, БЕЗ shell —
# держит верным собственный системный промпт этого файла ниже ("no shell").
_ALLOWED_TOOLS = LEGACY_INVESTIGATION_TOOL_NAMES

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
            console.print(f"[dim][MCP-AGENT] delegate nudge injected after {explore_count} exploration calls[/]")

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
# том же объекте `model`, что и внешняя роль/легаси-агент — delegate
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
# просто без рваных чужих текстовых фрагментов поверх. Общее для легаси
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


async def _run_subagent_streaming(sub_agent, conversation: list, config: dict) -> dict:
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
    раньше делал ainvoke."""
    on_event = current_on_event.get()
    prev_len = 0
    final_state: dict = {}
    async for state in sub_agent.astream({"messages": conversation}, config, stream_mode="values"):
        final_state = state
        msgs = state.get("messages") or []
        for m in msgs[prev_len:]:
            if on_event is None:
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    await on_event({
                        "type": "tool_start",
                        "name": f"delegate → {tc['name']}",
                        "args": tc.get("args", {}),
                    })
            elif isinstance(m, ToolMessage):
                await on_event({
                    "type": "tool_end",
                    "name": f"delegate → {m.name}",
                    "result": _tool_text(m.content)[:2000],
                })
        prev_len = len(msgs)
    return final_state


def build_delegate_tool(model, tools: list):
    """Собирает delegate как closure над уже поднятыми model/tools этой
    сессии — никакой второй подгрузки весов, никакого нового MCP-сервера.
    Вызывается из agent_builder._build_agent, где и model, и tools уже
    есть в области видимости."""
    delegate_tools = [t for t in tools if t.name in _ALLOWED_TOOLS]
    delegate_tools_by_name = {t.name: t for t in delegate_tools}
    sub_agent = create_agent(
        model,
        delegate_tools,
        system_prompt=_DELEGATE_SYSTEM_PROMPT,
        middleware=[_ToolErrorGuardMiddleware(), _DedupeToolResultsMiddleware()],
        checkpointer=InMemorySaver(),
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
        it does not see the rest of this conversation."""
        conversation = [HumanMessage(content=task)]
        final_text = ""

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
                result = await _run_subagent_streaming(sub_agent, conversation, config)
            except GraphRecursionError:
                return (
                    f"Sub-agent used its full {DELEGATE_RECURSION_LIMIT}-step "
                    "budget without reaching a final answer. Either delegate "
                    "again with a narrower task, or investigate the remainder "
                    "yourself — don't delegate the exact same task again."
                )

            messages = result.get("messages") or []
            final = messages[-1] if messages else None
            final_text = _tool_text(final.content) if isinstance(final, AIMessage) else ""

            if not final_text:
                return "Sub-agent finished without producing a final answer."
            if not _leaked_tool_call_syntax(final_text):
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

        return (
            "Sub-agent kept generating malformed tool-call markup instead of "
            "real tool calls after multiple recovery attempts — giving up. "
            f"Its last output was:\n\n{final_text}"
        )

    return delegate
