from .embeddings import embed_texts
from .store import VectorStore


async def index_episodic_entry(entry: dict, store: VectorStore) -> None:
    """Индексирует одну запись из episodic/writer.py — одно сообщение = один
    чанк, дробить не нужно, реплики короткие. Зависимость только в одну
    сторону: rag читает формат episodic, не наоборот."""
    if not entry.get("content", "").strip():
        return
    embedding = (await embed_texts([entry["content"]]))[0]
    store.add(
        f"{entry['session_id']}:{entry['seq']}",
        entry["content"],
        embedding,
        {
            "source_type": "dialog",
            "source": entry["session_id"],
            "role": entry["role"],
            "ts": entry["ts"],
        },
    )
