"""
Единая точка хранения персистентных данных самого flowAI (память о
пользователе, знания о проектах, usage-статистика) — SQLite-файл в
персистентной, не зависящей ни от текущей директории пользователя, ни от
каталога установки flowAI location.

Раньше эти данные писались как плоские JSON-файлы:
  - memory.json — cwd-relative (MCP-серверы memory/knowledge запускаются с
    cwd=repo_path, см. mcp_agent/config.py) → создавался в ЛЮБОЙ директории,
    откуда пользователь запускал flowai (например, в папке с версткой,
    которую его попросили сделать), а не в фиксированном месте.
  - usage.json — Path(__file__).parent, то есть каталог УСТАНОВКИ flowAI,
    который сам является git-репозиторием — отсюда "M usage.json" в
    git status самого flowAI на каждый чат.

Один sqlite-файл в стандартной XDG data-директории решает обе проблемы разом.

rag_index/ (семантические индексы кода/диалогов/сохранённых страниц,
mcp_agent/servers/rag_server.py) — тот же cwd-relative баг, просто не был
мигрирован вместе с остальным: писался прямо в <repo_path>/rag_index/,
засоряя git status ЛЮБОГО проекта, в котором открыт flowai, неотслеживаемыми
файлами. project_dir() ниже даёт то же решение (фиксированное место вне
проекта пользователя), но с сохранением per-project скоупинга — индекс кода
одного репозитория не должен мешаться с индексом другого.
"""
import hashlib
import os
import sqlite3
from pathlib import Path


def data_dir() -> Path:
    override = os.getenv("FLOWAI_DATA_DIR")
    if override:
        path = Path(override)
    else:
        xdg = os.getenv("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        path = base / "flowai"
    path.mkdir(parents=True, exist_ok=True)
    return path


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(data_dir() / "flowai.db")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """The project's schema-evolution primitive — see docs/persistence.md's
    "Миграции" section for the full convention. `CREATE TABLE IF NOT
    EXISTS` only covers a table that doesn't exist AT ALL yet — it does
    nothing once the table already exists with an older column set. A
    column added to that CREATE statement after real installs already have
    the table (with fewer columns) would otherwise silently never appear
    for them, causing "no such column: X" the first time old data meets
    new code. Adds whatever's missing from `columns` ({name: "TYPE DEFAULT
    ..."}) — safe to call every time a module connects, cheap (one PRAGMA
    read); every column added here should also exist in that module's own
    CREATE TABLE statement (this only helps an ALREADY-existing table
    catch up — a table created fresh right now already has everything
    from CREATE TABLE). Was local to mcp_agent/dnd_store.py (the only
    caller so far) — moved here so the NEXT module that needs to add a
    column to an existing table reuses this instead of copy-pasting its
    own private version."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _project_hash(repo_path: str) -> str:
    return hashlib.sha256(os.path.abspath(repo_path).encode()).hexdigest()[:16]


def project_dir(repo_path: str, *parts: str) -> Path:
    """A subdirectory under data_dir(), namespaced by an absolute project
    path — for project-scoped local files (e.g. rag_index) that must NOT
    live inside the user's own project tree (see module docstring). Hashed
    rather than the raw path so it works as a directory name on any OS
    regardless of path length/separators. *parts join onto the result as
    subdirectories (all created) — pass a filename yourself on top of the
    returned path instead of as a trailing part."""
    path = data_dir().joinpath("projects", _project_hash(repo_path), *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_dir_path(repo_path: str, *parts: str) -> Path:
    """Same path project_dir() would return, WITHOUT creating it — for
    callers that only want to know/check the path (e.g. rag/index_code.py
    probing whether some OTHER directory already has its own code index,
    while walking a project's subdirectories: probing dozens of candidate
    directories with project_dir() would have created an empty, never-used
    projects/<hash>/ cache folder for every single one just from looking)."""
    return data_dir().joinpath("projects", _project_hash(repo_path), *parts)
