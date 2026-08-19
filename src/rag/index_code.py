import os
from pathlib import Path

from mcp_agent.servers.file_ops_server import SKIP_DIRS

from .chunking import chunk_text
from .embeddings import EMBED_MODEL, embed_texts
from .store import VectorStore

# Файлы крупнее этого пропускаются — сгенерированные/минифицированные
# артефакты не несут смысловой ценности для семантического поиска и только
# раздувают индекс.
MAX_FILE_BYTES = 200_000


async def reindex_code(repo_path: str, store: VectorStore) -> int:
    """Полная пересборка индекса кода/доков проекта. Всегда с нуля (не
    инкрементально) — на масштабе одного проекта это проще и надёжнее, чем
    диффать; вызывается только вручную по тулу, не на каждом старте."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            try:
                if fpath.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # бинарники и нечитаемые файлы просто пропускаем

            rel = str(fpath.relative_to(repo_path))
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{rel}:{i}")
                texts.append(chunk)
                metadatas.append({"source_type": "code", "source": rel, "chunk_idx": i})

    embeddings = await embed_texts(texts)

    store.clear()
    for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
        store.add(id_, text, embedding, metadata)
    store.model = EMBED_MODEL
    store.dim = len(embeddings[0]) if embeddings else store.dim
    store.save()

    return len(ids)
