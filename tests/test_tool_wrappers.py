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


def _make_path_tool(name, coroutine):
    return StructuredTool(
        name=name,
        description="does a thing",
        args_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        coroutine=coroutine,
    )


@pytest.mark.asyncio
async def test_require_fresh_read_refuses_unread_existing_file(tmp_path):
    from mcp_agent.tool_wrappers import _require_fresh_read_tool

    target = tmp_path / "f.txt"
    target.write_text("hello")

    async def raw_write(**kwargs):
        return ("written", None)

    write_tool = _require_fresh_read_tool(_make_path_tool("write_file", raw_write), read_mtimes={})
    content, _ = await write_tool.coroutine(path=str(target))
    assert content.startswith("Error")
    assert "hasn't been read yet this session" in content


@pytest.mark.asyncio
async def test_require_fresh_read_allows_write_after_recorded_read(tmp_path):
    from mcp_agent.tool_wrappers import _require_fresh_read_tool
    import os

    target = tmp_path / "f.txt"
    target.write_text("hello")
    read_mtimes = {str(target): os.path.getmtime(target)}

    async def raw_write(**kwargs):
        return ("written", None)

    write_tool = _require_fresh_read_tool(_make_path_tool("write_file", raw_write), read_mtimes)
    content, _ = await write_tool.coroutine(path=str(target))
    assert content == "written"


@pytest.mark.asyncio
async def test_require_fresh_read_refuses_write_after_external_modification(tmp_path):
    from mcp_agent.tool_wrappers import _require_fresh_read_tool
    import os
    import time

    target = tmp_path / "f.txt"
    target.write_text("hello")
    read_mtimes = {str(target): os.path.getmtime(target)}

    time.sleep(1.1)  # coarse mtime resolution on some filesystems
    target.write_text("changed underneath the model")

    async def raw_write(**kwargs):
        return ("written", None)

    write_tool = _require_fresh_read_tool(_make_path_tool("write_file", raw_write), read_mtimes)
    content, _ = await write_tool.coroutine(path=str(target))
    assert content.startswith("Error")
    assert "changed on disk" in content


@pytest.mark.asyncio
async def test_require_fresh_read_allows_write_to_new_file(tmp_path):
    from mcp_agent.tool_wrappers import _require_fresh_read_tool

    target = tmp_path / "new.txt"  # does not exist yet — nothing to have read

    async def raw_write(**kwargs):
        return ("written", None)

    write_tool = _require_fresh_read_tool(_make_path_tool("write_file", raw_write), read_mtimes={})
    content, _ = await write_tool.coroutine(path=str(target))
    assert content == "written"


@pytest.mark.asyncio
async def test_track_read_mtime_records_path_on_successful_read(tmp_path):
    from mcp_agent.tool_wrappers import _track_read_mtime_tool
    import os

    target = tmp_path / "f.txt"
    target.write_text("hello")
    read_mtimes = {}

    async def raw_read(**kwargs):
        return ("hello", None)

    read_tool = _track_read_mtime_tool(_make_path_tool("read_file", raw_read), read_mtimes)
    await read_tool.coroutine(path=str(target))
    assert read_mtimes[str(target)] == os.path.getmtime(target)


@pytest.mark.asyncio
async def test_track_read_mtime_and_require_fresh_read_compose_across_separate_calls(tmp_path):
    """Regression test for the bug this pair of wrappers fixes: read_file and
    write_file used to guard freshness with a dict living INSIDE the
    file_ops_server.py MCP subprocess — but MultiServerMCPClient starts a
    fresh subprocess per tool call, so that dict was empty again by the time
    write_file ran, and the guard refused every write unconditionally, even
    right after a real read_file call. read_mtimes must live outside both
    calls (as it does here, shared across two independently-invoked
    wrapped tools) for a read followed by a write to actually work."""
    from mcp_agent.tool_wrappers import _require_fresh_read_tool, _track_read_mtime_tool

    target = tmp_path / "f.txt"
    target.write_text("hello")
    read_mtimes = {}

    async def raw_read(**kwargs):
        return ("hello", None)

    async def raw_write(**kwargs):
        return ("written", None)

    read_tool = _track_read_mtime_tool(_make_path_tool("read_file", raw_read), read_mtimes)
    write_tool = _require_fresh_read_tool(_make_path_tool("write_file", raw_write), read_mtimes)

    await read_tool.coroutine(path=str(target))
    content, _ = await write_tool.coroutine(path=str(target))
    assert content == "written"
