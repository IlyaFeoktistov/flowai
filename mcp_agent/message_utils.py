"""
Мелкие хелперы над LangChain-сообщениями, не завязанные ни на self-heal
цикл, ни на сборку тулов: сериализация content в текст (_content_text),
дедуп буквально одинаковых ToolMessage внутри одного хода
(_dedupe_identical_tool_results + _DedupeToolResultsMiddleware) и перевод
входных сообщений cli.py в формат LangChain (_to_lc_messages).
"""
import json
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True, default=str)


def _calls_by_id(messages: list) -> dict[str, dict]:
    """{tool_call_id: tc} for every tool call across `messages` — ToolMessage
    itself never carries the call's own name/args, only the matching
    AIMessage.tool_calls does, so anything that needs to join a ToolMessage
    result back to what invoked it has to build this same join first.
    Single home for a lookup that was independently reimplemented
    byte-for-byte in agent.py (_round_call_info), self_heal.py
    (_written_paths, _bash_commands) and compaction.py (_render_transcript)
    — each one's own docstring already pointed at one of the others as
    "same lookup pattern", so the duplication was known, just never
    collapsed."""
    calls_by_id: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                if tc.get("id"):
                    calls_by_id[tc["id"]] = tc
    return calls_by_id


def _find_call_by_id(messages: list, tool_call_id: str) -> AIMessage | None:
    """The AIMessage that made a given tool_call_id — same reversed-scan
    shape independently written twice: ask_user_tool.py's
    _sibling_tool_names (wants every SIBLING call from that same
    AIMessage, to block a tool called alongside ask_user before it's
    answered) and dnd_agent.py's post-tool-call middleware (wants the
    message's own .content text, to check whether the turn that triggered
    this tool call ended on a question). Not the same job as _calls_by_id
    above — that flattens every call across the whole conversation into an
    id->call dict, which loses which message a call belonged to; both
    callers here need the MESSAGE itself, not just its tool_calls entry."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and any(tc.get("id") == tool_call_id for tc in (m.tool_calls or [])):
            return m
    return None


# Живой прогон (mail-server, 20260707-155842-8b900098): MCP tool results
# normally come back as a LIST of content blocks ([{'type': 'text', 'text':
# '...'}, ...]), not a plain string — langchain_mcp_adapters' standard shape
# for every file_ops/bash tool result. str() on that
# list stringifies the whole Python structure and, critically, ESCAPES any
# real newlines inside the text into literal two-character '\n' sequences.
# That silently broke several things that assumed m.content was already
# clean text:
#   - self_heal.py's _execution_evidence_shows_failure did
#     str(m.content).startswith("error") — never matches, since the string
#     actually starts with "[{'type': 'text'..." — a failed bash (even
#     a fatal crash) was never caught, letting the model report success
#     off nothing but an unrelated earlier `php -l`.
#   - ui/stream.py's tool_end renderer splits the result on real newlines to
#     detect and color a unified diff (write_file/edit_file's own diff
#     output) — with newlines escaped to text, a multi-line diff collapses
#     into ONE "line", so the diff never rendered; the user saw a truncated
#     garbled repr instead of the actual (tool-generated, not
#     LLM-generated) diff.
# _content_text (above) doesn't fix this — json.dumps has the exact same
# escaping problem. This extracts and joins the real text blocks instead;
# falls back to str() for a plain string or an unrecognized shape.
def _tool_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if parts:
            return "\n".join(parts)
    return str(content)


def _dedupe_identical_tool_results(messages: list) -> list:
    """Между несколькими раундами тул-коллинга ВНУТРИ ОДНОГО хода (create_agent
    зовёт модель заново на каждый раунд, таща за собой всю историю) иногда
    накапливаются два ToolMessage с одним и тем же тулом+аргументами И
    ДОСЛОВНО одинаковым результатом — например, модель дважды вызвала
    один и тот же bash, не заметив, что уже это сделала.

    В отличие от _dedupe_read_tool (который блокирует ПОВТОРНОЕ выполнение
    read_file — он идемпотентен, блокировать безопасно),
    здесь мы НИКОГДА не трогаем сам вызов тула: bash и любой другой тул
    с побочными эффектами обязан выполняться каждый раз, когда модель его
    зовёт — состояние могло измениться между вызовами (git status, pytest
    после правки), и повторный вызов часто НАМЕРЕННО перепроверяет текущее
    состояние. Мы срезаем только то, что уходит МОДЕЛИ НА ВХОД в следующих
    раундах этого же хода: если результат ДОСЛОВНО совпал с уже показанным
    раньше — старая копия заменяется плейсхолдером вместо повторной отправки
    того же текста. Если контент отличается (состояние реально изменилось
    между вызовами) — обе копии остаются нетронутыми: это не дубликат, а
    полезная разница, которую нельзя терять."""
    call_info: dict[str, tuple[str, str]] = {}
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in (m.tool_calls or []):
                call_info[tc["id"]] = (
                    tc["name"],
                    json.dumps(tc.get("args", {}), sort_keys=True, default=str),
                )

    total_count: dict[tuple, int] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id in call_info:
            name, args_json = call_info[m.tool_call_id]
            key = (name, args_json, _content_text(m.content))
            total_count[key] = total_count.get(key, 0) + 1

    seen_count: dict[tuple, int] = {}
    result = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id in call_info:
            name, args_json = call_info[m.tool_call_id]
            key = (name, args_json, _content_text(m.content))
            if total_count.get(key, 0) > 1:
                seen_count[key] = seen_count.get(key, 0) + 1
                if seen_count[key] < total_count[key]:
                    m = m.model_copy(update={
                        "content": (
                            f"(Identical result to a later `{name}` call in this turn "
                            "with the same arguments — collapsed here to save context; "
                            "see the later call below for the actual output.)"
                        )
                    })
                elif total_count[key] > 1:
                    # This IS the newest copy of a repeated (name, args, result)
                    # — unlike the older copies above (already collapsed, the
                    # model has seen them), THIS one is what the model is about
                    # to reason from next, so the nudge has to land here, not on
                    # a copy that gets replaced. Live incident (2026-08-18):
                    # Verifier called `go get pkg@<bogus-pseudo-version>` 5 times
                    # in a row, ~10s apart, getting the exact same "invalid
                    # version" error every time — no state changed between
                    # calls, so blind repetition could never have produced a
                    # different result, and the run never reached a verdict
                    # (looked like a hang). _dedupe_read_tool already nudges
                    # this way for read_file's offset/limit hunting; side-effect
                    # tools like bash have no such backstop otherwise, since
                    # they're never blocked from re-running (state legitimately
                    # can change between calls, e.g. re-running tests after a
                    # fix — see this function's own docstring).
                    hint = (
                        f"\n\n[You've now called `{name}` with these exact "
                        f"arguments {total_count[key]} times this turn and "
                        "gotten the EXACT SAME result every time — nothing "
                        "changed, so repeating it unchanged again will not "
                        "produce a different outcome. Change the command/"
                        "arguments/approach, or stop and report this as a "
                        "blocker instead of retrying it as-is.]"
                    )
                    # _tool_text, not _content_text — the latter json.dumps's
                    # list-shaped MCP content (escaping real newlines into
                    # literal '\n', see this module's own docstring on why
                    # that's wrong for anything the model has to actually
                    # read); _content_text stays fine for the dedup KEY above
                    # since equality doesn't care about escaping.
                    m = m.model_copy(update={"content": _tool_text(m.content) + hint})
        result.append(m)
    return result


class _DedupeToolResultsMiddleware(AgentMiddleware):
    """Применяет _dedupe_identical_tool_results к истории ПЕРЕД каждым
    вызовом модели — request.override(messages=...) не трогает реальное
    состояние графа/чекпоинтер, только то, что физически уйдёт в этот
    конкретный запрос к Ollama."""

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=_dedupe_identical_tool_results(request.messages)))


def _to_lc_messages(messages: list[dict]) -> list[tuple[str, str]]:
    return [(m["role"], m.get("content", "")) for m in messages if m.get("content")]
