"""_DropStaleReadsMiddleware's log deduplication (mcp_agent/compaction.py) —
the content override is non-persistent (real graph state untouched), so
without dedup the SAME stale read gets rediscovered and re-logged on every
later model call for the rest of the conversation, reading on screen like
the agent looping/repeating even though nothing new happened."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import mcp_agent.compaction as compaction_module
from mcp_agent.compaction import _DropStaleReadsMiddleware


def _make_request(messages):
    def _override(**kwargs):
        return SimpleNamespace(messages=kwargs.get("messages", messages))
    return SimpleNamespace(messages=messages, override=_override)


async def _handler(request):
    return request


def _conversation():
    return [
        AIMessage(content="", tool_calls=[{"id": "r1", "name": "read_file", "args": {"path": "run.sh"}}]),
        ToolMessage(name="read_file", content="old content", tool_call_id="r1"),
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "write_file", "args": {"path": "run.sh"}}]),
        ToolMessage(name="write_file", content="Created 'run.sh' (1 lines).", tool_call_id="w1", status="success"),
    ]


@pytest.mark.asyncio
async def test_first_call_logs_and_overrides_content(monkeypatch):
    logged = []
    monkeypatch.setattr(compaction_module, "log_event", lambda *a, **k: logged.append(k.get("count")))
    middleware = _DropStaleReadsMiddleware()
    request = _make_request(_conversation())

    result = await middleware.awrap_model_call(request, _handler)

    assert logged == [1]
    stale_msg = result.messages[1]
    assert stale_msg.content.startswith("(stale —")


@pytest.mark.asyncio
async def test_second_call_same_staleness_overrides_but_does_not_relog(monkeypatch):
    logged = []
    monkeypatch.setattr(compaction_module, "log_event", lambda *a, **k: logged.append(k.get("count")))
    middleware = _DropStaleReadsMiddleware()

    await middleware.awrap_model_call(_make_request(_conversation()), _handler)
    assert logged == [1]

    # Same middleware instance, a FRESH (unmutated) copy of the same
    # conversation — mirrors how the real graph state is never touched,
    # so every later round re-derives the same staleness from scratch.
    result = await middleware.awrap_model_call(_make_request(_conversation()), _handler)

    assert logged == [1]  # not re-logged
    stale_msg = result.messages[1]
    assert stale_msg.content.startswith("(stale —")  # override still applied


@pytest.mark.asyncio
async def test_a_genuinely_new_stale_read_still_logs_again(monkeypatch):
    logged = []
    monkeypatch.setattr(compaction_module, "log_event", lambda *a, **k: logged.append(k.get("count")))
    middleware = _DropStaleReadsMiddleware()

    await middleware.awrap_model_call(_make_request(_conversation()), _handler)
    assert logged == [1]

    extended = _conversation() + [
        AIMessage(content="", tool_calls=[{"id": "r2", "name": "read_file", "args": {"path": "Makefile"}}]),
        ToolMessage(name="read_file", content="old makefile", tool_call_id="r2"),
        AIMessage(content="", tool_calls=[{"id": "w2", "name": "edit_file", "args": {"path": "Makefile"}}]),
        ToolMessage(name="edit_file", content="Edited 'Makefile' (1 replacement).", tool_call_id="w2", status="success"),
    ]
    await middleware.awrap_model_call(_make_request(extended), _handler)

    assert logged == [1, 1]  # the NEW Makefile staleness logs; run.sh's does not repeat
