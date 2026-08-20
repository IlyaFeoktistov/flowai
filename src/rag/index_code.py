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


def code_store_path(repo_path: str) -> str:
    """Единое место, где живёт путь к коду-индексу — и cli.py (команда
    /reindex, пользователь запускает вручную) и rag_server.py
    (search_code_semantic — читает готовый индекс, сам никогда не пишет)
    должны указывать на ОДИН и тот же файл; отдельное дублирование этой
    строки в двух местах рисковало бы разойтись при следующей правке
    storage.project_dir's аргументов."""
    from storage import project_dir
    return str(project_dir(repo_path, "rag_index") / "code.json")

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


def _iter_files(root_path: str):
    """Yields absolute file paths under root_path — root_path itself if
    it's already a file (a /reindex file.py target), or a SKIP_DIRS-
    filtered recursive walk if it's a directory (whole repo, or a /reindex
    src target)."""
    if os.path.isfile(root_path):
        yield root_path
        return
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if not _is_skipped_dir(d)]
        for fname in files:
            yield str(Path(root) / fname)


def _chunk_file(fpath: str, repo_path: str):
    """Yields (rel, chunk_idx, chunk_text) for one file — rel is relative
    to repo_path (not to fpath's own root), so a partial /reindex of a
    subdirectory still produces the SAME id/source form
    (f"{rel}:{i}"/metadata["source"]) a full reindex would have for that
    same file, and _iter_chunks' cross-target dedup (see reindex_code)
    can match them up."""
    p = Path(fpath)
    try:
        if p.stat().st_size > MAX_FILE_BYTES:
            return
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return  # бинарники и нечитаемые файлы просто пропускаем
    rel = str(p.relative_to(repo_path))
    for i, chunk in enumerate(chunk_text(text)):
        yield rel, i, chunk


def _iter_chunks(repo_path: str, max_chunks: int, roots: list[str] | None = None):
    """Генератор — обрывает обход на месте, как только достигнут
    max_chunks, вместо того чтобы сначала пройти всё дерево и порезать
    список постфактум (тот же объём файловых чтений всё равно был бы
    потрачен впустую). roots — абсолютные пути (файлы или директории) для
    точечного /reindex; None означает «весь repo_path», как раньше."""
    n = 0
    for root_path in (roots if roots is not None else [repo_path]):
        for fpath in _iter_files(root_path):
            for rel, i, chunk in _chunk_file(fpath, repo_path):
                if n >= max_chunks:
                    return
                yield rel, i, chunk
                n += 1


def _resolve_targets(repo_path: str, targets: list[str]) -> tuple[list[str], list[str]]:
    """targets — как их передал пользователь (относительно repo_path, или
    абсолютные) — возвращает (существующие абсолютные пути, не найденные
    как были переданы). Не бросает исключение на отсутствующий путь —
    /reindex src typo1 typo2 должен всё равно проиндексировать src и
    отдельно сказать пользователю про typo1/typo2, а не молча не сделать
    ничего."""
    resolved, missing = [], []
    for t in targets:
        abs_t = t if os.path.isabs(t) else os.path.join(repo_path, t)
        abs_t = os.path.normpath(abs_t)
        if os.path.exists(abs_t):
            resolved.append(abs_t)
        else:
            missing.append(t)
    return resolved, missing


async def reindex_code(repo_path: str, store: VectorStore, targets: list[str] | None = None) -> dict:
    """Пересобирает индекс кода/доков — либо весь проект с нуля
    (targets=None, старое поведение: store.clear() + полный обход), либо
    ТОЛЬКО перечисленные файлы/директории (targets — из /reindex src
    file.py ..., см. cli.py), не трогая остальной уже собранный индекс.
    Точечный режим — это ЗАМЕНА чанков затронутых файлов, не добавление:
    у каждого файла в targets сначала удаляются ВСЕ его старые чанки (см.
    store.remove_by_source), а затем добавляются свежие — иначе
    отредактированный файл, у которого стало МЕНЬШЕ чанков, чем раньше,
    оставил бы в индексе устаревшие "хвостовые" чанки от прежней, более
    длинной версии (id "file.py:3"/"file.py:4" никогда не перезапишутся,
    если после правки в файле осталось только 3 чанка: 0,1,2).

    Индекс сохраняется на диск (store.save(), см. rag_server.py:_INDEX_DIR)
    и переживает перезапуск процесса — search_code_semantic лениво
    подгружает его заново из файла, а не держит в памяти живого процесса,
    так что once-per-project реально означает once, а не once-per-session."""
    missing: list[str] = []
    roots: list[str] | None = None
    if targets:
        roots, missing = _resolve_targets(repo_path, targets)
        if not roots:
            return {"chunks": 0, "truncated": False, "missing": missing, "scoped": True}

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []

    for rel, i, chunk in _iter_chunks(repo_path, MAX_INDEXED_CHUNKS, roots):
        ids.append(f"{rel}:{i}")
        texts.append(chunk)
        metadatas.append({"source_type": "code", "source": rel, "chunk_idx": i})
    # Едет вплотную к потолку ТОЛЬКО если реально срезано (см. _iter_chunks'
    # ранний return) — ложноположительный случай (проект даёт РОВНО
    # MAX_INDEXED_CHUNKS чанков без единого лишнего) не стоит отдельной
    # проверки: он лишь напечатает лишнее предупреждение, не потеряет данные.
    truncated = len(texts) >= MAX_INDEXED_CHUNKS

    embeddings = await embed_texts(texts)

    if roots is None:
        store.clear()
    else:
        touched_sources = {m["source"] for m in metadatas}
        store.remove_by_source(touched_sources)
    for id_, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
        store.add(id_, text, embedding, metadata)
    store.model = EMBED_MODEL
    store.dim = len(embeddings[0]) if embeddings else store.dim
    store.save()

    log_event(
        "code_reindexed", repo_path=repo_path, chunks=len(ids),
        truncated=truncated, chunk_cap=MAX_INDEXED_CHUNKS,
        scoped=roots is not None, missing=len(missing),
    )
    return {"chunks": len(ids), "truncated": truncated, "missing": missing, "scoped": roots is not None}
