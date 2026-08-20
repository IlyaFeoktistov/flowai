"""rag_server.py's search_code_semantic — the model can no longer trigger
a reindex itself (reindex_code_search removed as a tool entirely; building
the index is expensive and the user decides via /reindex in cli.py, not
the model). When the index hasn't been built yet, search_code_semantic
must say so and point at /reindex instead of naming a tool that no longer
exists."""
import os

import pytest

import mcp_agent.servers.rag_server as rag_server
import storage
from mcp_agent.servers.rag_server import search_code_semantic
from rag import VectorStore
from rag.index_code import code_store_path


def test_reindex_code_search_tool_removed():
    assert not hasattr(rag_server, "reindex_code_search")


@pytest.mark.asyncio
async def test_empty_index_message_points_at_reindex_command_not_a_tool(monkeypatch):
    empty_store = VectorStore(path="/dev/null")
    monkeypatch.setattr(rag_server, "_get_code_store", lambda: empty_store)

    result = await search_code_semantic(query="anything")

    assert "/reindex" in result
    assert "reindex_code_search" not in result


@pytest.fixture
def _isolated_repo(tmp_path, monkeypatch):
    """Points rag_server at a throwaway repo_path/data_dir, and resets its
    module-global code-store cache — real state left over from another
    test (or the real machine's ~/.local/share/flowai) must not leak in."""
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path / "flowai_data")
    monkeypatch.setattr(rag_server, "_REPO_PATH", str(tmp_path / "project"))
    monkeypatch.setattr(rag_server, "_code_store", None)
    monkeypatch.setattr(rag_server, "_code_store_mtime", None)
    return rag_server._REPO_PATH


def test_get_code_store_reloads_when_the_file_changes_on_disk(_isolated_repo):
    """Live motivation: /reindex (cli.py) and the background auto-reindex
    (plugin_hooks.py) both run in the MAIN process and always write a
    fresh code.json — but this MCP subprocess used to load its
    VectorStore ONCE and cache it forever, so search_code_semantic within
    the same session never saw anything written after its first call,
    even though the file on disk was already current. mtime-based
    invalidation closes that gap without re-parsing the file on every
    single call when nothing has actually changed."""
    path = code_store_path(_isolated_repo)
    store = VectorStore(path)
    store.add("a.py:0", "hello", [0.1], {"source": "a.py"})
    store.save()

    first = rag_server._get_code_store()
    assert len(first) == 1

    # Force a distinctly different mtime rather than relying on real
    # wall-clock sleep — some filesystems have coarse mtime resolution,
    # which would make this test flaky under real timing.
    newer_store = VectorStore(path)
    newer_store.add("a.py:0", "hello", [0.1], {"source": "a.py"})
    newer_store.add("b.py:0", "world", [0.2], {"source": "b.py"})
    newer_store.save()
    current_mtime = os.path.getmtime(path)
    os.utime(path, (current_mtime + 10, current_mtime + 10))

    second = rag_server._get_code_store()
    assert len(second) == 2


def test_get_code_store_does_not_reload_when_the_file_is_unchanged(_isolated_repo):
    path = code_store_path(_isolated_repo)
    store = VectorStore(path)
    store.add("a.py:0", "hello", [0.1], {"source": "a.py"})
    store.save()

    first = rag_server._get_code_store()
    second = rag_server._get_code_store()

    assert first is second  # same object — not reloaded when nothing changed
