import fnmatch
import os
from pathlib import Path

from mcp_agent.debug_log import log_event
from mcp_agent.servers.file_ops_server import SKIP_DIRS

from .chunking import chunk_text
from .embeddings import EMBED_MODEL, embed_texts
from .store import VectorStore

# Файлы крупнее этого пропускаются — сгенерированные/минифицированные
# артефакты не несут смысловой ценности для семантического поиска и только
# раздувают индекс.
MAX_FILE_BYTES = 200_000

# Верхний потолок на общее число чанков за один reindex — не лимит "на
# нормальный проект" (это ~1.2М строк при 60 строках/чанк, см. chunking.py),
# а страховка от переиндексации, которая выглядит зависшей: например, если
# SKIP_DIRS однажды не поймает какую-то генерируемую/вендорную директорию
# (см. _is_skipped_dir ниже — раньше сравнение было буквенным, "venv*"
# никогда не совпадал ни с одной реальной директорией), reindex тихо ходил
# бы по ней целиком, эмбеддя тысячи нерелевантных чанков одним sequential
# батчем за батчем, без какой-либо обратной связи пользователю о том, что
# происходит.
MAX_INDEXED_CHUNKS = 20_000


def _is_skipped_dir(name: str) -> bool:
    """fnmatch, не буквенное сравнение — SKIP_DIRS содержит glob-паттерны
    (например "venv*", чтобы поймать и venv/, и venv-tts/), а file_ops_
    server.py's grep_search/glob_search уже применяют их через fnmatch/
    --glob (см. file_ops_server.py:_is_skipped/315/345). Прежняя версия
    здесь делала `d not in SKIP_DIRS` — буквальное равенство строк, при
    котором паттерн "venv*" требовал бы directory с ИМЕНЕМ "venv*"
    (звёздочка как обычный символ), т.е. никогда реально не совпадал —
    обычная директория "venv" (без точки, в отличие от уже покрытого
    ".venv") проходила мимо фильтра и индексировалась целиком."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in SKIP_DIRS)


def _iter_chunks(repo_path: str, max_chunks: int):
    """Генератор — обрывает обход os.walk на месте, как только достигнут
    max_chunks, вместо того чтобы сначала пройти всё дерево и порезать
    список постфактум (тот же объём файловых чтений всё равно был бы
    потрачен впустую)."""
    n = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not _is_skipped_dir(d)]
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
                if n >= max_chunks:
                    return
                yield rel, i, chunk
                n += 1


async def reindex_code(repo_path: str, store: VectorStore) -> dict:
    """Полная пересборка индекса кода/доков проекта. Всегда с нуля (не
    инкрементально) — на масштабе одного проекта это проще и надёжнее, чем
    диффать; вызывается только вручную по тулу, не на каждом старте.
    Индекс сохраняется на диск (store.save(), см. rag_server.py:_INDEX_DIR)
    и переживает перезапуск процесса — search_code_semantic лениво
    подгружает его заново из файла, а не держит в памяти живого процесса,
    так что once-per-project реально означает once, а не once-per-session."""
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for rel, i, chunk in _iter_chunks(repo_path, MAX_INDEXED_CHUNKS):
        ids.append(f"{rel}:{i}")
        texts.append(chunk)
        metadatas.append({"source_type": "code", "source": rel, "chunk_idx": i})
    # Едет вплотную к потолку ТОЛЬКО если реально срезано (см. _iter_chunks'
    # ранний return) — ложноположительный случай (проект даёт РОВНО
    # MAX_INDEXED_CHUNKS чанков без единого лишнего) не стоит отдельной
    # проверки: он лишь напечатает лишнее предупреждение, не потеряет данные.
    truncated = len(texts) >= MAX_INDEXED_CHUNKS

    embeddings = await embed_texts(texts)

    store.clear()
    for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
        store.add(id_, text, embedding, metadata)
    store.model = EMBED_MODEL
    store.dim = len(embeddings[0]) if embeddings else store.dim
    store.save()

    log_event(
        "code_reindexed", repo_path=repo_path, chunks=len(ids),
        truncated=truncated, chunk_cap=MAX_INDEXED_CHUNKS,
    )
    return {"chunks": len(ids), "truncated": truncated}
