"""mcp_agent/optimized_tools.py:build_optimized_tools — filters the loaded
tool list down to OPTIMIZED_TOOL_NAMES when settings.optimized_tools is
on. extra_names lets a plugin's tools (mcp_agent/plugins.py) survive this
filter too — their names can't be in the static OPTIMIZED_TOOL_NAMES set
since they aren't known ahead of time."""
from types import SimpleNamespace

from mcp_agent.optimized_tools import OPTIMIZED_TOOL_NAMES, build_optimized_tools


def _tool(name):
    return SimpleNamespace(name=name)


def test_keeps_only_optimized_names_by_default():
    tools = [_tool("bash"), _tool("some_removed_tool")]
    filtered, by_name = build_optimized_tools(tools)
    assert [t.name for t in filtered] == ["bash"]
    assert set(by_name) == {"bash"}
    assert "bash" in OPTIMIZED_TOOL_NAMES


def test_extra_names_survive_the_filter():
    tools = [_tool("bash"), _tool("plugin_tool"), _tool("some_removed_tool")]
    filtered, by_name = build_optimized_tools(tools, extra_names=frozenset({"plugin_tool"}))
    assert {t.name for t in filtered} == {"bash", "plugin_tool"}
    assert set(by_name) == {"bash", "plugin_tool"}


def test_extra_names_do_not_widen_the_result_if_absent_from_tools():
    tools = [_tool("bash")]
    filtered, _by_name = build_optimized_tools(tools, extra_names=frozenset({"never_loaded"}))
    assert [t.name for t in filtered] == ["bash"]
