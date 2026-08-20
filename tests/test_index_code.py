"""rag/index_code.py — reindex_code's implementation, now driven only by
cli.py's /reindex command (the model can no longer trigger this itself).

Live-caught motivation (user report): reindex looked like it could hang
forever on a real project, with no cap and a real SKIP_DIRS bug —
_is_skipped_dir used to do a literal `d not in SKIP_DIRS` string-equality
check, so a glob entry like "venv*" (meant to also catch "venv-tts") never
matched any real directory name at all (fnmatch's "*" only works through
fnmatch.fnmatch, not `==`) — a plain "venv" directory (as opposed to the
already-exact-matched ".venv") walked straight through the filter and got
fully indexed, chunk by chunk, with no feedback and no ceiling."""
import pytest

from rag.index_code import _is_skipped_dir, _iter_chunks, reindex_code
from rag.store import VectorStore


def test_is_skipped_dir_matches_glob_star_pattern():
    assert _is_skipped_dir("venv")       # bare "venv" must match the "venv*" pattern
    assert _is_skipped_dir("venv-tts")
    assert _is_skipped_dir("node_modules")
    assert _is_skipped_dir(".venv")
    assert not _is_skipped_dir("vendor_docs")  # doesn't match "vendor" exactly nor any pattern


def test_iter_chunks_stops_at_cap(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.index_code.chunk_text", lambda text: [text] * 5)
    for i in range(3):
        (tmp_path / f"f{i}.txt").write_text("hello world")

    chunks = list(_iter_chunks(str(tmp_path), max_chunks=7))

    assert len(chunks) == 7  # stops mid-file, doesn't overshoot to 15


def test_iter_chunks_skips_matched_dirs(tmp_path):
    venv_dir = tmp_path / "venv"
    venv_dir.mkdir()
    (venv_dir / "lib.py").write_text("import this\n" * 5)
    (tmp_path / "real.py").write_text("def handler():\n    pass\n")

    chunks = list(_iter_chunks(str(tmp_path), max_chunks=1000))

    sources = {rel for rel, _, _ in chunks}
    assert "real.py" in sources
    assert not any(s.startswith("venv") for s in sources)


@pytest.mark.asyncio
async def test_reindex_code_reports_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.index_code.MAX_INDEXED_CHUNKS", 2)
    monkeypatch.setattr("rag.index_code.chunk_text", lambda text: [text] * 5)
    (tmp_path / "big.py").write_text("x = 1\n")

    async def _fake_embed(texts):
        return [[0.1, 0.2] for _ in texts]
    monkeypatch.setattr("rag.index_code.embed_texts", _fake_embed)

    store = VectorStore(str(tmp_path / "index.json"))
    result = await reindex_code(str(tmp_path), store)

    assert result == {"chunks": 2, "truncated": True}
    assert len(store) == 2


@pytest.mark.asyncio
async def test_reindex_code_not_truncated_under_cap(tmp_path, monkeypatch):
    (tmp_path / "small.py").write_text("x = 1\n")

    async def _fake_embed(texts):
        return [[0.1, 0.2] for _ in texts]
    monkeypatch.setattr("rag.index_code.embed_texts", _fake_embed)

    store = VectorStore(str(tmp_path / "index.json"))
    result = await reindex_code(str(tmp_path), store)

    assert result["truncated"] is False
    assert result["chunks"] >= 1
