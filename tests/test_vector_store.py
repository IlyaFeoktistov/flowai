from rag.store import VectorStore


def test_remove_by_source_drops_only_matching_chunks(tmp_path):
    store = VectorStore(str(tmp_path / "index.json"))
    store.add("a.py:0", "text a0", [0.1], {"source": "a.py"})
    store.add("a.py:1", "text a1", [0.1], {"source": "a.py"})
    store.add("b.py:0", "text b0", [0.1], {"source": "b.py"})

    removed = store.remove_by_source({"a.py"})

    assert removed == 2
    assert len(store) == 1
    assert "b.py:0" in store._chunks


def test_remove_by_source_no_match_removes_nothing():
    store = VectorStore("unused.json")
    store.add("a.py:0", "text", [0.1], {"source": "a.py"})

    removed = store.remove_by_source({"nonexistent.py"})

    assert removed == 0
    assert len(store) == 1
