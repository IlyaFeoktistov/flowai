"""mcp_agent/roles.py — flowai_guide (guide_server.py) must reach every
pipeline role that can plausibly get asked "what are you"/"what can you
do" mid-turn, and survive the optimized_tools filter (mcp_agent/
optimized_tools.py) the same way ask_user/mark_plan_step_current do."""
import mcp_agent.optimized_tools as optimized_tools
import mcp_agent.roles as roles


def test_present_in_every_role_tool_set():
    assert "flowai_guide" in roles.investigator_tools()
    assert "flowai_guide" in roles.planner_tools()
    assert "flowai_guide" in roles.coder_tools()
    assert "flowai_guide" in roles.verifier_tools()
    assert "flowai_guide" in roles.executor_tools(needs_project=True)
    assert "flowai_guide" in roles.executor_tools(needs_project=False)
    assert "flowai_guide" in roles.MAIN_INVESTIGATION_TOOL_NAMES


def test_survives_optimized_tools_filter():
    assert "flowai_guide" in optimized_tools.OPTIMIZED_TOOL_NAMES
