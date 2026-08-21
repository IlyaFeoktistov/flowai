"""mcp_agent/stage_runner.py:run_stage — the two gaps found and closed
while collapsing mcp_agent/agent.py onto this engine: a missing
MAX_SELF_HEAL_ASKS cap on the punt-to-user rescue (a model that keeps
punting would otherwise loop forever, since max_attempts grows in lockstep
with attempt), and gen_duration_ms/attempts_used not being threaded into
StageResult at all (silently dropped, needed by the main caller's stats
event and error messages)."""
import pytest

import mcp_agent.agent as agent_mod
import mcp_agent.stage_runner as sr
from mcp_agent.model_config import MAX_SELF_HEAL_ASKS


def _always_asks_a_question(monkeypatch, mock_ask_user):
    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        from langchain_core.messages import AIMessage
        return {"messages": [AIMessage(content="What should I do next?")]}, 1, 1, 1, 1, False, False, 7

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    async def fake_extract_shape(judge_model, text):
        return {"question": text, "options": [], "recommended": None}

    monkeypatch.setattr(sr, "_extract_ask_user_shape", fake_extract_shape)
    monkeypatch.setattr(sr, "ask_user_question", mock_ask_user)


@pytest.mark.asyncio
async def test_punt_to_user_cap_prevents_infinite_loop(monkeypatch):
    async def mock_ask_user(question, options, recommended):
        return "some answer"

    _always_asks_a_question(monkeypatch, mock_ask_user)

    def verdict_fn(round_msgs, new_tool_msgs, round_final_text):
        return {"relevant": False, "reason": "always asks a question, never resolves"}

    def guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text):
        return "stop asking, just answer"

    result = await __import__("asyncio").wait_for(
        sr.run_stage(
            agent=None, payload={"messages": []}, on_event=None,
            judge_model=None, tools_by_name={}, read_history={},
            verdict_fn=verdict_fn, guidance_fn=guidance_fn,
            max_attempts=3, recursion_limit=10, stage_name="test",
        ),
        timeout=5,
    )
    # Terminates instead of looping forever — the concrete count depends on
    # MAX_SELF_HEAL_ASKS + the leftover normal-retry budget, but it MUST
    # terminate and it must have actually used the ask-rescue at least once.
    assert result.attempts_used >= MAX_SELF_HEAL_ASKS
    assert not result.verdict["relevant"]


@pytest.mark.asyncio
async def test_punt_to_user_respects_the_cap_exactly(monkeypatch):
    ask_calls = []

    async def mock_ask_user(question, options, recommended):
        ask_calls.append(1)
        return "some answer"

    _always_asks_a_question(monkeypatch, mock_ask_user)

    def verdict_fn(round_msgs, new_tool_msgs, round_final_text):
        return {"relevant": False, "reason": "always asks a question"}

    def guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text):
        return "guidance"

    # Give it a huge max_attempts so the cap (not attempt exhaustion) is
    # what actually stops the interactive asking.
    await sr.run_stage(
        agent=None, payload={"messages": []}, on_event=None,
        judge_model=None, tools_by_name={}, read_history={},
        verdict_fn=verdict_fn, guidance_fn=guidance_fn,
        max_attempts=3, recursion_limit=10, stage_name="test",
    )
    assert len(ask_calls) == MAX_SELF_HEAL_ASKS


@pytest.mark.asyncio
async def test_gen_duration_ms_accumulates_across_rounds(monkeypatch):
    round_gen_ms = [100, 250]

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        from langchain_core.messages import AIMessage
        ms = round_gen_ms.pop(0)
        return {"messages": [AIMessage(content="round text")]}, 1, 10, 10, 1, False, False, ms

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    calls = {"n": 0}

    def verdict_fn(round_msgs, new_tool_msgs, round_final_text):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"relevant": False, "reason": "retry once"}
        return {"relevant": True, "reason": "ok now"}

    def guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text):
        return "try again"

    result = await sr.run_stage(
        agent=None, payload={"messages": []}, on_event=None,
        judge_model=None, tools_by_name={}, read_history={},
        verdict_fn=verdict_fn, guidance_fn=guidance_fn,
        max_attempts=3, recursion_limit=10, stage_name="test",
    )
    assert result.gen_duration_ms == 350
    assert result.attempts_used == 2


@pytest.mark.asyncio
async def test_mid_turn_queue_is_forwarded_to_stream_round(monkeypatch):
    received = {}

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        from langchain_core.messages import AIMessage
        received["mid_turn_queue"] = mid_turn_queue
        return {"messages": [AIMessage(content="done")]}, 1, 1, 1, 1, False, False, 1

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    def verdict_fn(round_msgs, new_tool_msgs, round_final_text):
        return {"relevant": True, "reason": "ok"}

    def guidance_fn(verdict, round_msgs, new_tool_msgs, round_final_text):
        return ""

    sentinel = object()
    await sr.run_stage(
        agent=None, payload={"messages": []}, on_event=None,
        judge_model=None, tools_by_name={}, read_history={},
        verdict_fn=verdict_fn, guidance_fn=guidance_fn,
        max_attempts=1, recursion_limit=10, stage_name="test",
        mid_turn_queue=sentinel,
    )
    assert received["mid_turn_queue"] is sentinel
