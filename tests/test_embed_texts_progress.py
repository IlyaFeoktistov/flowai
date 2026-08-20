"""rag/embeddings.py:embed_texts's on_progress callback — /reindex (cli.py)
could take minutes on a real project with no feedback at all before this;
on_progress(done, total) fires after each Ollama batch so callers can show
at least a percentage."""
import pytest

import rag.embeddings as embeddings


class _FakeOllamaClient:
    def __init__(self, batch_sizes_seen):
        self._batch_sizes_seen = batch_sizes_seen

    async def embed(self, model, input):
        self._batch_sizes_seen.append(len(input))
        return {"embeddings": [[0.1] for _ in input]}


@pytest.mark.asyncio
async def test_on_progress_fires_after_each_batch_with_running_totals(monkeypatch):
    batch_sizes_seen = []
    monkeypatch.setattr(embeddings, "ollama", type("M", (), {
        "AsyncClient": staticmethod(lambda: _FakeOllamaClient(batch_sizes_seen)),
    }))
    monkeypatch.setattr(embeddings, "_BATCH_SIZE", 2)

    calls = []
    texts = ["a", "b", "c", "d", "e"]  # 3 batches of size 2, 2, 1
    await embeddings.embed_texts(texts, on_progress=lambda done, total: calls.append((done, total)))

    assert calls == [(2, 5), (4, 5), (5, 5)]
    assert batch_sizes_seen == [2, 2, 1]


@pytest.mark.asyncio
async def test_on_progress_not_called_for_empty_input(monkeypatch):
    calls = []
    await embeddings.embed_texts([], on_progress=lambda done, total: calls.append((done, total)))
    assert calls == []


@pytest.mark.asyncio
async def test_no_progress_callback_is_fine(monkeypatch):
    monkeypatch.setattr(embeddings, "ollama", type("M", (), {
        "AsyncClient": staticmethod(lambda: _FakeOllamaClient([])),
    }))
    result = await embeddings.embed_texts(["a", "b"])  # on_progress omitted entirely
    assert result == [[0.1], [0.1]]
