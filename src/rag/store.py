import base64
import json
import math
from array import array
from pathlib import Path


def _encode(embedding: list[float]) -> str:
    return base64.b64encode(array("f", embedding).tobytes()).decode("ascii")


def _decode(embedding_b64: str) -> list[float]:
    a = array("f")
    a.frombytes(base64.b64decode(embedding_b64))
    return a.tolist()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorStore:
    """Плоский векторный индекс на stdlib — без numpy/faiss/chromadb.
    Корпус этого проекта (код одного репозитория + история диалогов +
    горстка сохранённых страниц) реалистично — сотни-низкие тысячи чанков;
    полный перебор косинусного сходства на чистом Python занимает десятки
    миллисекунд, round-trip к Ollama за эмбеддингом и так доминирует по
    времени. Заводить первую numeric-зависимость в проекте с 8 зависимостями
    ради этого масштаба неоправданно."""

    def __init__(self, path: str, model: str = "", dim: int = 0):
        self.path = Path(path)
        self.model = model
        self.dim = dim
        self._chunks: dict[str, dict] = {}

    def add(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        self._chunks[id] = {"text": text, "embedding": embedding, "metadata": metadata}

    def clear(self) -> None:
        self._chunks = {}

    def remove_by_source(self, sources: set[str]) -> int:
        """Drops every chunk whose metadata["source"] is in `sources` — used
        by a scoped /reindex (index_code.py:reindex_code, targets= given)
        to replace a re-indexed file's chunks outright instead of just
        overwriting by id: if the file shrank (fewer chunks than its
        previous version), the old version's now-orphaned tail chunk ids
        (e.g. "file.py:3"/"file.py:4" when the new version only produces
        0..2) would otherwise never get overwritten and linger in the
        index with stale content forever."""
        stale_ids = [cid for cid, c in self._chunks.items() if c["metadata"].get("source") in sources]
        for cid in stale_ids:
            del self._chunks[cid]
        return len(stale_ids)

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scored = [
            {"id": cid, "text": c["text"], "metadata": c["metadata"], "score": _cosine(query_embedding, c["embedding"])}
            for cid, c in self._chunks.items()
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "dim": self.dim,
            "chunks": [
                {"id": cid, "text": c["text"], "embedding_b64": _encode(c["embedding"]), "metadata": c["metadata"]}
                for cid, c in self._chunks.items()
            ],
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str, model: str = "", dim: int = 0) -> "VectorStore":
        p = Path(path)
        if not p.exists():
            return cls(path, model, dim)
        data = json.loads(p.read_text(encoding="utf-8"))
        store = cls(path, data.get("model", model), data.get("dim", dim))
        for chunk in data.get("chunks", []):
            store._chunks[chunk["id"]] = {
                "text": chunk["text"],
                "embedding": _decode(chunk["embedding_b64"]),
                "metadata": chunk["metadata"],
            }
        return store
