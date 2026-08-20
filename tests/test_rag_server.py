"""rag_server.py's search_code_semantic — the model can no longer trigger
a reindex itself (reindex_code_search removed as a tool entirely; building
the index is expensive and the user decides via /reindex in cli.py, not
the model). When the index hasn't been built yet, search_code_semantic
must say so and point at /reindex instead of naming a tool that no longer
exists."""
import pytest

import mcp_agent.servers.rag_server as rag_server
from mcp_agent.servers.rag_server import search_code_semantic
from rag import VectorStore


def test_reindex_code_search_tool_removed():
    assert not hasattr(rag_server, "reindex_code_search")


@pytest.mark.asyncio
async def test_empty_index_message_points_at_reindex_command_not_a_tool(monkeypatch):
    empty_store = VectorStore(path="/dev/null")
    monkeypatch.setattr(rag_server, "_get_code_store", lambda: empty_store)

    result = await search_code_semantic(query="anything")

    assert "/reindex" in result
    assert "reindex_code_search" not in result
