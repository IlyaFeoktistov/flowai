"""mcp_agent/subagents.py + delegate_tool.py's agent/agent_result — agent
types, custom agent files, plan-mode downgrade and the slot limit, without a
real model."""
import asyncio

import pytest

import settings
from mcp_agent import subagents, work_mode
from mcp_agent import delegate_tool

RO = frozenset({"read_file", "grep_search"})


@pytest.fixture(autouse=True)
def _no_user_agents(tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "user_agents_dir", lambda: tmp_path / "_user_agents")


def test_builtin_types_and_mutation():
    agents = subagents.discover_agents(None, RO)
    assert set(agents) >= {"explore", "plan", "general"}
    assert not agents["explore"].can_mutate({"read_file", "write_file"})
    assert agents["general"].can_mutate({"read_file", "write_file"})


def test_custom_agent_file(tmp_path):
    d = tmp_path / ".flowai" / "agents"
    d.mkdir(parents=True)
    (d / "tester.md").write_text("---\ndescription: writes tests\ntools: read_file, write_file\n---\nYou write tests.\n")
    spec = subagents.discover_agents(str(tmp_path), RO)["tester"]
    assert spec.tools == frozenset({"read_file", "write_file"}) and spec.source == "project"
    assert spec.system_prompt.startswith("You write tests.")
    assert "tester: writes tests" in subagents.agents_prompt_block(subagents.discover_agents(str(tmp_path), RO))


def _agent_tools():
    class _T:
        def __init__(self, name):
            self.name = name
    return delegate_tool.build_delegate_tool(None, [_T("read_file"), _T("write_file")], repo_path=None)


def test_unknown_type():
    agent, _result = _agent_tools()

    async def call(kind, mode):
        token = work_mode.current_work_mode.set(mode)
        try:
            return await agent.ainvoke({"description": "x", "prompt": "do x", "subagent_type": kind})
        finally:
            work_mode.current_work_mode.reset(token)

    assert "unknown subagent_type" in asyncio.run(call("nope", work_mode.BUILD))


def test_writing_agent_is_downgraded_in_plan_mode():
    general = subagents.discover_agents(None, RO)["general"]
    available = {"read_file", "grep_search", "write_file", "edit_file", "bash"}
    spec = delegate_tool.plan_mode_spec(general, available)
    assert spec.tools == frozenset({"read_file", "grep_search", "bash"})
    assert "PLAN mode" in spec.system_prompt


def test_agent_result_unknown_id():
    _agent, result = _agent_tools()
    assert "no background agent" in asyncio.run(result.ainvoke({"agent_id": "zzz"}))


def test_slot_semaphore_follows_setting(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_semaphore", None)
    monkeypatch.setattr(settings, "get", lambda k: 3 if k == "parallel_slots" else None)
    assert delegate_tool._slot_semaphore()._value == 3
    monkeypatch.setattr(settings, "get", lambda k: 99 if k == "parallel_slots" else None)
    monkeypatch.setattr(delegate_tool, "_semaphore", None)
    assert delegate_tool._slot_semaphore()._value == 8  # capped
