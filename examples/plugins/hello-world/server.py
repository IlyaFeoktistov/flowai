"""Example MCP server — see plugin.json's "mcp_servers" entry. A plugin's
server is a completely standalone script (own process, own stdio) with no
access to flowai's own internals — write it exactly like any other MCP
server, using the `mcp` package already in flowai's requirements.txt.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hello-world")


@mcp.tool()
async def hello_world_echo(text: str) -> str:
    """Echo text back, prefixed — proves this plugin's MCP server is
    reachable by the model as an ordinary tool."""
    return f"hello-world plugin says: {text}"


if __name__ == "__main__":
    mcp.run()
