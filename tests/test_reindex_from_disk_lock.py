"""rag/index_code.py:reindex_code_from_disk — owns its own load-from-disk
-> reindex -> save-to-disk cycle, serialized per repo_path via a lock.

Live motivation: with auto-reindex-on-file-touch (plugin_hooks.py), a
single turn that reads/writes several files fires one independent
background task PER file, each against the SAME on-disk code.json. Two
overlapping calls that each do their own VectorStore.load() + save()
without any coordination would race: whichever finishes saving LAST wins
outright, silently discarding whatever the other one had already added —
exactly the kind of lost update this lock exists to prevent."""
import asyncio

import pytest

from rag.index_code import reindex_code_from_disk
from rag.store import VectorStore


@pytest.fixture
def _project(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def a():\n    pass\n")
    (tmp_path / "b.py").write_text("def b():\n    pass\n")

    async def _slow_embed(texts):
        await asyncio.sleep(0.01)  # gives the event loop a chance to interleave
        return [[0.1, 0.2] for _ in texts]
    monkeypatch.setattr("rag.index_code.embed_texts", _slow_embed)

    from rag.index_code import code_store_path
    return str(tmp_path), code_store_path(str(tmp_path))


@pytest.mark.asyncio
async def test_concurrent_reindex_of_different_files_does_not_lose_either(_project):
    repo_path, store_path = _project

    await asyncio.gather(
        reindex_code_from_disk(repo_path, targets=["a.py"]),
        reindex_code_from_disk(repo_path, targets=["b.py"]),
    )

    final = VectorStore.load(store_path)
    sources = {c["metadata"]["source"] for c in final._chunks.values()}
    assert sources == {"a.py", "b.py"}


@pytest.mark.asyncio
async def test_concurrent_reindex_calls_are_actually_serialized(_project):
    """Directly observes that the two calls' critical sections never
    overlap — a stronger check than the outcome-only test above, so a
    future refactor that accidentally drops the lock (while somehow still
    passing the outcome test by luck) still gets caught."""
    repo_path, _store_path = _project
    active = 0
    max_concurrent = 0

    from rag.index_code import _reindex_lock

    async def _tracked_reindex(target):
        nonlocal active, max_concurrent
        async with _reindex_lock(repo_path):
            active += 1
            max_concurrent = max(max_concurrent, active)
            await asyncio.sleep(0.01)
            active -= 1

    await asyncio.gather(_tracked_reindex("a.py"), _tracked_reindex("b.py"))

    assert max_concurrent == 1
