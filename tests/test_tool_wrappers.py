"""mcp_agent/tool_wrappers.py:_rewrap_tool — single home for the 7-field
StructuredTool reconstruction that used to be hand-copied in 7 places
(6 wrappers in this module + snapshots.py:_snapshot_before_write)."""
import pytest
from langchain_core.tools import StructuredTool


async def _base_coroutine(**kwargs):
    return ("base result", None)


def _make_base_tool():
    return StructuredTool(
        name="my_tool",
        description="does a thing",
        args_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        coroutine=_base_coroutine,
    )


def test_rewrap_preserves_name_and_unchanged_fields():
    from mcp_agent.tool_wrappers import _rewrap_tool

    tool = _make_base_tool()

    async def new_coroutine(**kwargs):
        return ("new result", None)

    wrapped = _rewrap_tool(tool, coroutine=new_coroutine)
    assert wrapped.name == "my_tool"
    assert wrapped.description == "does a thing"
    assert wrapped.args_schema == tool.args_schema
    assert wrapped.response_format == tool.response_format
    assert wrapped.handle_tool_error == tool.handle_tool_error


def test_rewrap_overrides_only_the_fields_given():
    from mcp_agent.tool_wrappers import _rewrap_tool

    tool = _make_base_tool()
    new_schema = {"type": "object", "properties": {}}

    async def new_coroutine(**kwargs):
        return ("v", None)

    wrapped = _rewrap_tool(tool, coroutine=new_coroutine, description="new desc", args_schema=new_schema)
    assert wrapped.description == "new desc"
    assert wrapped.args_schema == new_schema
    assert wrapped.name == tool.name  # untouched


@pytest.mark.asyncio
async def test_rewrap_actually_uses_the_new_coroutine():
    from mcp_agent.tool_wrappers import _rewrap_tool

    tool = _make_base_tool()

    async def new_coroutine(**kwargs):
        return ("replaced", None)

    wrapped = _rewrap_tool(tool, coroutine=new_coroutine)
    content, artifact = await wrapped.coroutine()
    assert content == "replaced"
