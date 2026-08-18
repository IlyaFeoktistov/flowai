"""mcp_agent/stages/legacy.py — the verdict/guidance tree extracted from
the old mcp_agent/agent.py monolith. Covers both entry paths (no tool
calls vs. has tool calls) and the LLM-judge fallback that only this role
(unlike every pipeline stage) needs."""
import pytest

from conftest import ai_message, tool_message

import mcp_agent.stages.legacy as legacy_mod
from mcp_agent.stages.legacy import legacy_guidance, make_legacy_verdict


@pytest.fixture
def judge_stub(monkeypatch):
    """Records what it was called with and returns a controllable verdict."""
    calls = []

    def install(verdict):
        async def fake_semantic_check(model, task, tool_msgs, answer, ask_user_called):
            calls.append((task, tool_msgs, answer, ask_user_called))
            return verdict
        monkeypatch.setattr(legacy_mod, "_semantic_check", fake_semantic_check)
        return calls

    return install


@pytest.mark.asyncio
async def test_no_tools_no_question_is_relevant_without_calling_the_judge(judge_stub):
    calls = judge_stub({"relevant": False, "reason": "should never be called"})
    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event=None)
    v = await verdict_fn([], [], "All done, nothing more to do.")
    assert v == {"relevant": True, "reason": "plain text answer, nothing to verify"}
    assert calls == []


@pytest.mark.asyncio
async def test_no_tools_with_question_defers_to_the_judge(judge_stub):
    judge_stub({"relevant": True, "reason": "fine"})
    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event=None)
    v = await verdict_fn([], [], "Which option do you want?")
    assert v == {"relevant": True, "reason": "fine"}


@pytest.mark.asyncio
async def test_failed_write_is_rejected_before_reaching_the_judge(judge_stub):
    calls = judge_stub({"relevant": False, "reason": "should never be called"})
    round_msgs = [
        ai_message([{"id": "1", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", content="Error: bad", status="error", tool_call_id="1"),
    ]
    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event=None)
    v = await verdict_fn(round_msgs, round_msgs[1:], "wrote it")
    assert v["relevant"] is False
    assert "nothing was actually written/edited" in v["reason"]
    assert calls == []


@pytest.mark.asyncio
async def test_execution_failure_kind_set_when_bash_check_fails(judge_stub):
    judge_stub({"relevant": False, "reason": "unused"})
    round_msgs = [
        ai_message([{"id": "1", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", tool_call_id="1"),
        ai_message([{"id": "2", "name": "bash", "args": {"command": "python x.py"}}]),
        tool_message("bash", content="Error (exit 1): boom", tool_call_id="2"),
    ]
    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event=None)
    v = await verdict_fn(round_msgs, [m for m in round_msgs if hasattr(m, "tool_call_id")], "done")
    assert v == {"relevant": False, "kind": "execution_failure", "reason": v["reason"]}


@pytest.mark.asyncio
async def test_falls_through_deterministic_tree_to_judge_and_tags_kind_semantic(judge_stub):
    judge_stub({"relevant": True, "reason": "looks right"})
    round_msgs = [
        ai_message([{"id": "1", "name": "read_file", "args": {"path": "x.py"}}]),
        tool_message("read_file", content="some content", tool_call_id="1"),
    ]
    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event=None)
    v = await verdict_fn(round_msgs, round_msgs[1:], "here is the answer")
    assert v["relevant"] is True
    assert v["kind"] == "semantic"


@pytest.mark.asyncio
async def test_verifying_events_wrap_every_judge_call(judge_stub):
    judge_stub({"relevant": True, "reason": "ok"})
    events = []

    async def on_event(e):
        events.append(e["type"])

    verdict_fn = make_legacy_verdict("JUDGE", "do X", on_event)
    await verdict_fn([], [], "what now?")
    assert events == ["verifying_start", "verifying_end"]


def test_guidance_no_tools_branch_does_not_repeat_the_reason():
    # run_stage's own _seed_retry already prepends "(reason: ...)" — this
    # guidance must not duplicate that verbatim phrase.
    g = legacy_guidance({"relevant": False, "reason": "x"}, [], [], "what now?")
    assert "(reason:" not in g
    assert "call the ask_user tool now" in g


def test_guidance_execution_failure_includes_the_real_bash_error():
    round_msgs = [
        ai_message([{"id": "1", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", tool_call_id="1"),
        ai_message([{"id": "2", "name": "bash", "args": {"command": "python x.py"}}]),
        tool_message("bash", content="Error (exit 1): NameError: x", tool_call_id="2"),
    ]
    new_tool_msgs = [m for m in round_msgs if hasattr(m, "tool_call_id")]
    g = legacy_guidance({"relevant": False, "kind": "execution_failure", "reason": "r"}, round_msgs, new_tool_msgs, "")
    assert "NameError: x" in g
    assert "still there" in g


def test_guidance_falls_back_to_generic_advice_when_no_specific_condition_matches():
    # kind=None (not "semantic") and ask_user WAS called — this suppresses
    # every specific guidance_parts branch, forcing the generic fallback.
    round_msgs = [
        ai_message([{"id": "1", "name": "read_file", "args": {"path": "x.py"}}]),
        tool_message("read_file", tool_call_id="1"),
        ai_message([{"id": "2", "name": "ask_user", "args": {}}]),
        tool_message("ask_user", tool_call_id="2"),
    ]
    new_tool_msgs = [m for m in round_msgs if hasattr(m, "tool_call_id")]
    g = legacy_guidance({"relevant": False, "reason": "off topic"}, round_msgs, new_tool_msgs, "some answer")
    assert "judge's reason above is the real signal" in g
