"""mcp_agent/work_mode.py + agent_builder._PlanModeMiddleware — plan mode
blocks writes/mutating bash mechanically; plans are saved and found."""
import asyncio

from langchain_core.messages import ToolMessage

from mcp_agent import work_mode
from mcp_agent.agent_builder import _PlanModeMiddleware


class _Req:
    def __init__(self, name, args=None):
        self.tool_call = {"name": name, "args": args or {}, "id": "1"}


def _run(name, args=None, mode=work_mode.PLAN):
    async def handler(_req):
        return "ran"

    async def go():
        token = work_mode.current_work_mode.set(mode)
        try:
            return await _PlanModeMiddleware().awrap_tool_call(_Req(name, args), handler)
        finally:
            work_mode.current_work_mode.reset(token)
    return asyncio.run(go())


def test_plan_mode_blocks_writes_and_mutating_bash():
    assert isinstance(_run("write_file", {"path": "a"}), ToolMessage)
    assert isinstance(_run("edit_file"), ToolMessage)
    assert isinstance(_run("bash", {"command": "rm -rf build"}), ToolMessage)
    assert isinstance(_run("some_plugin_tool"), ToolMessage)


def test_plan_mode_allows_reads_and_read_only_bash():
    assert _run("read_file", {"path": "a"}) == "ran"
    assert _run("bash", {"command": "git log --oneline"}) == "ran"


def test_build_mode_is_a_no_op():
    assert _run("write_file", {"path": "a"}, mode=work_mode.BUILD) == "ran"


def test_save_and_latest_plan(tmp_path):
    first = work_mode.save_plan(str(tmp_path), "добавь кэш в API", "## Goal\nx")
    assert first.parent == tmp_path / ".flowai" / "plans"
    assert "добавь-кэш-в-api" in first.name
    assert work_mode.latest_plan(str(tmp_path)) == first
    task = work_mode.build_task_from_plan(first, "без тестов")
    assert "## Goal" in task and "без тестов" in task
