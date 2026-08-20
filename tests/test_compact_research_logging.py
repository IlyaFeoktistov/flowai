"""_CompactResearchMiddleware's debug print (mcp_agent/compaction.py) — like
_DropStaleReadsMiddleware's log dedup (test_drop_stale_reads_logging.py),
awrap_model_call runs before EVERY model call and re-applies a cached
digest whenever the prefix it's keyed on hasn't changed yet. Without the
is_fresh guard, that reapplication printed "compacted history: ..." on
every single round even though the actual (judge-model) summarization only
ran once — reading on screen like the agent was recompacting on every tool
call."""
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import mcp_agent.compaction as compaction_module
from mcp_agent.compaction import _CompactResearchMiddleware


def _make_request(messages):
    def _override(**kwargs):
        return SimpleNamespace(messages=kwargs.get("messages", messages), system_message=None, tools=None)
    return SimpleNamespace(messages=messages, system_message=None, tools=None, override=_override)


async def _handler(request):
    return request


def _conversation_with_one_write():
    return [
        HumanMessage(content="fix the bug"),
        AIMessage(content="", tool_calls=[{"id": "r1", "name": "read_file", "args": {"path": "a.go"}}]),
        ToolMessage(name="read_file", content="package main", tool_call_id="r1"),
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "write_file", "args": {"path": "a.go"}}]),
        ToolMessage(name="write_file", content="Created 'a.go' (1 lines).", tool_call_id="w1", status="success"),
    ]


@pytest.fixture(autouse=True)
def _force_compaction(monkeypatch):
    monkeypatch.setattr(compaction_module, "DEBUG", True)
    monkeypatch.setattr(compaction_module, "_needs_compaction", lambda *a, **k: True)
    monkeypatch.setattr(compaction_module.settings, "get", lambda key, *a, **k: True)

    async def _fake_summarize(judge_model, prefix):
        return "DIGEST TEXT"
    monkeypatch.setattr(compaction_module, "_summarize_research", _fake_summarize)


@pytest.mark.asyncio
async def test_first_compaction_of_a_prefix_prints(monkeypatch):
    printed = []
    monkeypatch.setattr(compaction_module, "debug_print", lambda msg: printed.append(msg))
    middleware = _CompactResearchMiddleware(judge_model=None)

    result = await middleware.awrap_model_call(_make_request(_conversation_with_one_write()), _handler)

    assert len(printed) == 1
    assert "compacted history" in printed[0]
    assert any("DIGEST TEXT" in str(m.content) for m in result.messages)


@pytest.mark.asyncio
async def test_reapplying_the_same_cached_digest_does_not_reprint(monkeypatch):
    printed = []
    monkeypatch.setattr(compaction_module, "debug_print", lambda msg: printed.append(msg))
    middleware = _CompactResearchMiddleware(judge_model=None)

    await middleware.awrap_model_call(_make_request(_conversation_with_one_write()), _handler)
    assert len(printed) == 1

    # Same prefix (no NEW write yet), one more tool-call round appended —
    # mirrors the real graph state never shrinking, so this middleware
    # re-derives the same cut/digest from scratch every round.
    extended = _conversation_with_one_write() + [
        AIMessage(content="", tool_calls=[{"id": "b1", "name": "bash", "args": {"command": "go build ./..."}}]),
        ToolMessage(name="bash", content="build ok", tool_call_id="b1"),
    ]
    result = await middleware.awrap_model_call(_make_request(extended), _handler)

    assert len(printed) == 1  # not re-logged
    assert any("DIGEST TEXT" in str(m.content) for m in result.messages)  # digest still applied


@pytest.mark.asyncio
async def test_submitted_plan_survives_compaction_verbatim(monkeypatch):
    """submit_plan's own AIMessage (and everything after it, up to the
    write cut) must never be folded into the digest — see
    _last_plan_message_index's docstring. Research BEFORE the plan was
    submitted is still fair game for summarization."""
    conversation = [
        HumanMessage(content="fix the bug"),
        AIMessage(content="", tool_calls=[{"id": "r1", "name": "read_file", "args": {"path": "a.go"}}]),
        ToolMessage(name="read_file", content="package main", tool_call_id="r1"),
        AIMessage(content="1. Fix the loop in a.go", tool_calls=[
            {"id": "p1", "name": "submit_plan", "args": {"steps": ["Fix the loop in a.go"]}},
        ]),
        ToolMessage(name="submit_plan", content="Registered plan with 1 step(s).", tool_call_id="p1"),
        AIMessage(content="", tool_calls=[{"id": "w1", "name": "write_file", "args": {"path": "a.go"}}]),
        ToolMessage(name="write_file", content="Created 'a.go' (1 lines).", tool_call_id="w1", status="success"),
    ]
    middleware = _CompactResearchMiddleware(judge_model=None)

    result = await middleware.awrap_model_call(_make_request(conversation), _handler)

    plan_call_names = [
        tc["name"]
        for m in result.messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    ]
    assert "submit_plan" in plan_call_names
    assert any("1. Fix the loop in a.go" == m.content for m in result.messages)
    assert any("DIGEST TEXT" in str(m.content) for m in result.messages)


@pytest.mark.asyncio
async def test_a_genuinely_new_write_moves_the_cut_and_prints_again(monkeypatch):
    printed = []
    monkeypatch.setattr(compaction_module, "debug_print", lambda msg: printed.append(msg))
    middleware = _CompactResearchMiddleware(judge_model=None)

    await middleware.awrap_model_call(_make_request(_conversation_with_one_write()), _handler)
    assert len(printed) == 1

    extended = _conversation_with_one_write() + [
        AIMessage(content="", tool_calls=[{"id": "r2", "name": "read_file", "args": {"path": "b.go"}}]),
        ToolMessage(name="read_file", content="package main", tool_call_id="r2"),
        AIMessage(content="", tool_calls=[{"id": "w2", "name": "write_file", "args": {"path": "b.go"}}]),
        ToolMessage(name="write_file", content="Created 'b.go' (1 lines).", tool_call_id="w2", status="success"),
    ]
    await middleware.awrap_model_call(_make_request(extended), _handler)

    assert len(printed) == 2  # the NEW prefix (up to w2) is a fresh compaction
