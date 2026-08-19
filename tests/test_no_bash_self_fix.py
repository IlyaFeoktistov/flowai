"""_NoBashSelfFixMiddleware (mcp_agent/agent_builder.py) — now attached to
verifier/coder/quick_fix (previously verifier only), so Coder/quick_fix
can use bash to run real checks without being able to use it to bypass
write_file/edit_file's snapshot safety net. Also: roles.py's coder_tools/
executor_tools now include bash at all."""
from types import SimpleNamespace

import pytest

from mcp_agent.agent_builder import _NoBashSelfFixMiddleware
from mcp_agent.roles import coder_tools, executor_tools


def _request(command):
    return SimpleNamespace(tool_call={"name": "bash", "args": {"command": command}, "id": "1"})


async def _handler(request):
    return "handler ran"


@pytest.mark.asyncio
async def test_mutating_command_denied_regardless_of_role():
    for has_write_tools in (True, False):
        middleware = _NoBashSelfFixMiddleware("/repo", has_write_tools=has_write_tools)
        result = await middleware.awrap_tool_call(_request("sed -i 's/x/y/' /repo/foo.py"), _handler)
        assert result.status == "error"
        assert "Denied" in result.content


@pytest.mark.asyncio
async def test_denial_points_at_write_tools_when_role_has_them():
    middleware = _NoBashSelfFixMiddleware("/repo", has_write_tools=True)
    result = await middleware.awrap_tool_call(_request("cat > /repo/foo.py << 'EOF'\nx\nEOF"), _handler)
    assert "write_file/edit_file" in result.content
    assert "Report this as a failure" not in result.content


@pytest.mark.asyncio
async def test_denial_points_at_reporting_when_role_has_no_write_tools():
    middleware = _NoBashSelfFixMiddleware("/repo", has_write_tools=False)
    result = await middleware.awrap_tool_call(_request("cat > /repo/foo.py << 'EOF'\nx\nEOF"), _handler)
    assert "Report this as a failure" in result.content


@pytest.mark.asyncio
async def test_check_command_passes_through():
    middleware = _NoBashSelfFixMiddleware("/repo", has_write_tools=True)
    result = await middleware.awrap_tool_call(_request("go build ./..."), _handler)
    assert result == "handler ran"


def test_coder_and_executor_tools_now_include_bash():
    assert "bash" in coder_tools()
    assert "bash" in executor_tools(needs_project=True)
