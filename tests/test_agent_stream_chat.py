"""End-to-end mcp_agent/agent.py:stream_chat — the legacy monolithic path
(voice_mode / pipeline_mode=off), now a thin wrapper around
mcp_agent/stage_runner.py:run_stage with mcp_agent/stages/legacy.py's
verdict/guidance. Every real model/tool call is mocked at _stream_round —
these confirm the collapse (agent.py: 1476 -> 747 lines) kept behavior
intact for the scenarios documented in the old code's own comments."""
import pytest
from langchain_core.messages import AIMessage

import mcp_agent.agent as agent_mod
import mcp_agent.stage_runner as sr
import mcp_agent.stages.legacy as legacy_mod
import settings

from conftest import ai_message, tool_message


class _FakeCompactResearch:
    def clear_cache(self):
        pass


@pytest.fixture(autouse=True)
def fake_get_agent(monkeypatch):
    async def _fake():
        return ("AGENT", "MODEL", "JUDGE", {}, {}, _FakeCompactResearch())

    monkeypatch.setattr(agent_mod, "_get_agent", _fake)

    async def _no_knowledge(repo_path):
        return None

    monkeypatch.setattr(agent_mod, "load_knowledge", _no_knowledge)
    settings.set_value("self_heal_enabled", True)
    yield


async def _collect(gen):
    out = []
    async for chunk in gen:
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_successful_round_yields_final_text_and_stats(monkeypatch):
    round_msgs = [
        ai_message([{"id": "1", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", tool_call_id="1"),
        ai_message([{"id": "2", "name": "bash", "args": {"command": "python x.py"}}]),
        tool_message("bash", content="ok output", tool_call_id="2"),
        AIMessage(content="Done, wrote x.py and it runs fine."),
    ]

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        return {"messages": round_msgs}, len(round_msgs), 100, 50, 1, False, False, 500

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    events = []

    async def on_event(e):
        events.append(e)

    outputs = await _collect(agent_mod.stream_chat([{"role": "user", "content": "write and run x.py"}], on_event=on_event))

    assert outputs == ["Done, wrote x.py and it runs fine."]
    stats = [e for e in events if e["type"] == "stats"]
    assert stats and stats[0]["gen_duration_ms"] == 500
    assert any(e["type"] == "done" for e in events)


@pytest.mark.asyncio
async def test_recursion_limit_on_every_attempt_reports_after_all_attempts(monkeypatch):
    calls = {"n": 0}

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        calls["n"] += 1
        round_msgs = [
            ai_message([{"id": "1", "name": "read_file", "args": {"path": "a.py"}}]),
            tool_message("read_file", tool_call_id="1"),
        ]
        return {"messages": round_msgs}, 2, 10, 5, 1, True, False, 100

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    outputs = await _collect(agent_mod.stream_chat([{"role": "user", "content": "investigate everything"}]))
    assert len(outputs) == 1
    assert "Не удалось получить ответ" in outputs[0]
    assert calls["n"] == 3  # MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_execution_failure_on_last_attempt_triggers_auto_revert(monkeypatch):
    reverted = []

    def fake_revert(touched_paths, turn_start_wall):
        reverted.append(set(touched_paths))
        return [f"reverted {p}" for p in touched_paths]

    monkeypatch.setattr(agent_mod, "_revert_turn_paths", fake_revert)

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        round_msgs = [
            ai_message([{"id": "1", "name": "write_file", "args": {"path": "buggy.py"}}]),
            tool_message("write_file", tool_call_id="1"),
            ai_message([{"id": "2", "name": "bash", "args": {"command": "python buggy.py"}}]),
            tool_message("bash", content="Error (exit 1): SyntaxError", tool_call_id="2"),
            AIMessage(content="I wrote buggy.py but it fails."),
        ]
        return {"messages": round_msgs}, len(round_msgs), 10, 5, 1, False, False, 50

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    outputs = await _collect(agent_mod.stream_chat([{"role": "user", "content": "fix the bug"}]))
    assert len(outputs) == 1
    assert "Проверка (bash) показала" in outputs[0]
    assert "reverted buggy.py" in outputs[0]
    assert reverted == [{"buggy.py"}]


@pytest.mark.asyncio
async def test_self_heal_disabled_gives_exactly_one_attempt_and_no_banner(monkeypatch):
    settings.set_value("self_heal_enabled", False)
    calls = {"n": 0}

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        calls["n"] += 1
        round_msgs = [
            ai_message([{"id": "1", "name": "write_file", "args": {"path": "x.py"}}]),
            tool_message("write_file", tool_call_id="1"),
            AIMessage(content="wrote it, did not test"),
        ]
        return {"messages": round_msgs}, len(round_msgs), 10, 5, 1, False, False, 50

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    outputs = await _collect(agent_mod.stream_chat([{"role": "user", "content": "do x"}]))
    assert calls["n"] == 1
    assert outputs == ["wrote it, did not test"]  # no "не удалось до конца проверить" banner


@pytest.mark.asyncio
async def test_mid_turn_queue_reaches_stream_round_unchanged(monkeypatch):
    received = {}

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        received["q"] = mid_turn_queue
        return {"messages": [AIMessage(content="done, no questions")]}, 1, 1, 1, 1, False, False, 5

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    sentinel = object()
    await _collect(agent_mod.stream_chat([{"role": "user", "content": "go"}], mid_turn_queue=sentinel))
    assert received["q"] is sentinel


@pytest.mark.asyncio
async def test_punt_to_user_cap_then_falls_back_to_normal_retries(monkeypatch):
    """Stateful fake mimicking LangGraph's checkpointer: the SAME thread_id
    accumulates messages across punt-to-user rounds (which don't get a new
    thread), while a genuinely new thread (from a normal digest-retry)
    starts fresh — same distinction real LangGraph's checkpointer makes."""
    threads = {}
    calls = {"n": 0}

    async def fake_stream_round(agent, current_input, config, on_event, emitted, mid_turn_queue=None):
        calls["n"] += 1
        tid = config["configurable"]["thread_id"]
        history = threads.setdefault(tid, [])
        history.extend(current_input["messages"])
        history.append(AIMessage(content="Which option do you want?"))
        return {"messages": list(history)}, len(history), 5, 5, 1, False, False, 10

    monkeypatch.setattr(agent_mod, "_stream_round", fake_stream_round)

    async def fake_semantic_check(model, task, tool_msgs, answer, ask_user_called):
        return {"relevant": False, "reason": "keeps asking instead of deciding"}

    monkeypatch.setattr(legacy_mod, "_semantic_check", fake_semantic_check)

    async def fake_ask_user_question(question, options, recommended):
        return "some answer"

    monkeypatch.setattr(sr, "ask_user_question", fake_ask_user_question)

    async def fake_extract_shape(judge_model, text):
        return {"question": text, "options": [], "recommended": None}

    monkeypatch.setattr(sr, "_extract_ask_user_shape", fake_extract_shape)

    outputs = await _collect(agent_mod.stream_chat([{"role": "user", "content": "pick something"}]))
    assert len(outputs) == 1
    assert "Не удалось до конца проверить" in outputs[0]
    assert calls["n"] > 3  # more than just the 3 capped interactive punts
