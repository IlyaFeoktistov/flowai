"""mcp_agent/agent_builder.py:_load_tools_resilient — loads each MCP
server's tools independently (one failing server shouldn't cost every
other server its tools) but now does so CONCURRENTLY instead of one
server at a time (perf audit finding: 9-12 independent subprocess spawns
have no data dependency on each other). Also reports which loaded tool
names came from a plugin-provided server (mcp_agent/plugins.py) — roles.py's
per-role tool_names are static allowlists that can't possibly know a
plugin's tool names ahead of time, so agent_builder.py needs this to let
plugin tools through regardless of role."""
import asyncio
from types import SimpleNamespace

import pytest

from mcp_agent.agent_builder import _load_tools_resilient


class _FakeClient:
    def __init__(self, behavior: dict):
        self._behavior = behavior

    async def get_tools(self, server_name: str):
        behavior = self._behavior[server_name]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def _tool(name):
    return SimpleNamespace(name=name)


@pytest.mark.asyncio
async def test_merges_tools_from_every_server():
    client = _FakeClient({"a": ["tool_a1", "tool_a2"], "b": ["tool_b1"]})
    tools, plugin_names = await _load_tools_resilient(client, ["a", "b"])
    assert tools == ["tool_a1", "tool_a2", "tool_b1"]
    assert plugin_names == frozenset()


@pytest.mark.asyncio
async def test_one_failing_server_does_not_cost_the_others_their_tools():
    client = _FakeClient({"good": ["tool_1"], "bad": RuntimeError("npx not found")})
    tools, _plugin_names = await _load_tools_resilient(client, ["good", "bad"])
    assert tools == ["tool_1"]


@pytest.mark.asyncio
async def test_reports_tool_names_from_plugin_servers_only():
    client = _FakeClient({
        "bash": [_tool("bash")],
        "my_plugin_server": [_tool("plugin_tool_a"), _tool("plugin_tool_b")],
    })
    tools, plugin_names = await _load_tools_resilient(
        client, ["bash", "my_plugin_server"], plugin_server_names=frozenset({"my_plugin_server"}),
    )
    assert {t.name for t in tools} == {"bash", "plugin_tool_a", "plugin_tool_b"}
    assert plugin_names == frozenset({"plugin_tool_a", "plugin_tool_b"})


@pytest.mark.asyncio
async def test_servers_load_concurrently_not_one_at_a_time():
    """Regression guard: a for-loop of sequential awaits would take
    len(server_names) * delay; gather()'ing them takes ~one delay total."""
    order = []

    class _SlowClient:
        async def get_tools(self, server_name: str):
            order.append(f"start:{server_name}")
            await asyncio.sleep(0.05)
            order.append(f"end:{server_name}")
            return [server_name]

    loop = asyncio.get_event_loop()
    start = loop.time()
    await _load_tools_resilient(_SlowClient(), ["s1", "s2", "s3"])
    elapsed = loop.time() - start

    assert elapsed < 0.05 * 3
    # every server must have STARTED before any of them finished —
    # a sequential loop would fully finish s1 before s2 ever starts.
    starts_before_first_end = order[: order.index("end:s1")]
    assert {"start:s1", "start:s2", "start:s3"} <= set(starts_before_first_end)
