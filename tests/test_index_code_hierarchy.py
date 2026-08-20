"""Hierarchical /reindex — a monorepo subdirectory that was already opened
and reindexed on its own (its own code.json under storage.project_dir)
must NOT be re-walked/re-embedded by a reindex run one level up. Instead
the parent's store carries a live reference (VectorStore.child_indexes)
that federated search follows lazily, prefixed with the child's relative
directory so results stay meaningful from the parent's point of view."""
import pytest

import storage
from rag.index_code import code_store_path, reindex_code
from rag.store import VectorStore


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path_factory, monkeypatch):
    data_dir = tmp_path_factory.mktemp("flowai_data")
    monkeypatch.setattr(storage, "data_dir", lambda: data_dir)


@pytest.fixture
def _fake_embed(monkeypatch):
    async def _embed(texts, **kwargs):
        return [[0.1, 0.2] for _ in texts]
    monkeypatch.setattr("rag.index_code.embed_texts", _embed)


def _seed_child_index(child_dir):
    """Simulates 'core/ was already opened and reindexed on its own' —
    writes a real code.json at exactly the path code_store_path(child_dir)
    would resolve to, with one chunk in it."""
    store = VectorStore(code_store_path(str(child_dir)), model="nomic-embed-text")
    store.add("service.go:0", "func Handle() {}", [1.0, 0.0], {"source": "service.go", "chunk_idx": 0})
    store.save()
    return store


@pytest.mark.asyncio
async def test_full_reindex_skips_subdir_with_own_index_and_references_it(tmp_path, _fake_embed):
    (tmp_path / "a.py").write_text("def a():\n    pass\n")
    core = tmp_path / "core"
    core.mkdir()
    (core / "service.go").write_text("func Handle() {}\n")
    _seed_child_index(core)

    parent_store = VectorStore(str(tmp_path / "index.json"))
    result = await reindex_code(str(tmp_path), parent_store)

    assert result["referenced"] == 1
    sources = {c["metadata"]["source"] for c in parent_store._chunks.values()}
    assert sources == {"a.py"}  # core/service.go was NOT re-embedded here
    assert parent_store.child_indexes == [{"dir": "core", "index_path": code_store_path(str(core))}]


@pytest.mark.asyncio
async def test_explicit_target_on_an_already_indexed_dir_is_reindexed_for_real(tmp_path, _fake_embed):
    core = tmp_path / "core"
    core.mkdir()
    (core / "service.go").write_text("func Handle() {}\n")
    _seed_child_index(core)

    parent_store = VectorStore(str(tmp_path / "index.json"))
    result = await reindex_code(str(tmp_path), parent_store, targets=["core"])

    # Explicitly named -> processed directly, not skipped/referenced.
    assert result["referenced"] == 0
    sources = {c["metadata"]["source"] for c in parent_store._chunks.values()}
    assert sources == {"core/service.go"}


@pytest.mark.asyncio
async def test_scoped_reindex_preserves_previously_discovered_children(tmp_path, _fake_embed):
    (tmp_path / "a.py").write_text("def a():\n    pass\n")
    (tmp_path / "b.py").write_text("def b():\n    pass\n")
    core = tmp_path / "core"
    core.mkdir()
    (core / "service.go").write_text("func Handle() {}\n")
    _seed_child_index(core)

    parent_store = VectorStore(str(tmp_path / "index.json"))
    await reindex_code(str(tmp_path), parent_store)  # full reindex first, discovers core/
    assert len(parent_store.child_indexes) == 1

    result = await reindex_code(str(tmp_path), parent_store, targets=["b.py"])

    assert result["referenced"] == 0  # nothing NEW discovered by this scoped call
    assert parent_store.child_indexes == [{"dir": "core", "index_path": code_store_path(str(core))}]


@pytest.mark.asyncio
async def test_federated_search_finds_child_results_with_prefixed_source(tmp_path, _fake_embed):
    (tmp_path / "a.py").write_text("def a():\n    pass\n")
    core = tmp_path / "core"
    core.mkdir()
    (core / "service.go").write_text("func Handle() {}\n")
    _seed_child_index(core)

    parent_store = VectorStore(str(tmp_path / "index.json"))
    await reindex_code(str(tmp_path), parent_store)

    results = parent_store.search([1.0, 0.0], top_k=5)

    sources = {r["metadata"]["source"] for r in results}
    assert "core/service.go" in sources


def test_search_cycle_protection(tmp_path):
    """A hand-edited/corrupt JSON could reference itself — must not
    infinite-loop."""
    p = tmp_path / "index.json"
    store = VectorStore(str(p), child_indexes=[{"dir": "self", "index_path": str(p)}])
    store.add("a:0", "text", [1.0], {"source": "a"})
    store.save()

    reloaded = VectorStore.load(str(p))
    results = reloaded.search([1.0], top_k=5)

    assert len(results) == 1  # own chunk once, not duplicated via the self-reference
