"""
Снимки файлов (file_snapshots в storage.py:connect()) и их автоматический
откат — единственное, что стоит между моделью, пишущей в файлы, и
пользователем, разгребающим последствия.

- _save_file_snapshot/_snapshot_before_write: снимок содержимого файла ДО
  каждой мутирующей записи (write_file/edit_file/replace_lines/copy_lines/
  git_restore_file) — обёртка над тулами применяется в agent_builder.py.
- list_file_snapshots/restore_file_snapshot: ручной откат к конкретному
  снимку по запросу модели/пользователя.
- _revert_turn_paths: автоматический откат всех правок ОДНОГО хода, когда
  self-heal в stream_chat (mcp_agent/agent.py) исчерпал попытки, а
  bash_exec-проверка так и не прошла (см. _execution_evidence_shows_failure
  в mcp_agent/self_heal.py) — вместо того, чтобы оставить сломанный код в
  проекте и понадеяться, что пользователь заметит.

Снимки — per-process (_SESSION_ID), не per-turn: они не пишутся куда-либо
за пределы этого запуска flowai и чистятся при выходе
(clear_session_file_snapshots, вызывается из cli.py).
"""
import os
import shutil
import uuid
from datetime import datetime

from langchain_core.tools import BaseTool, StructuredTool, tool

import storage

# Уникален на каждый запуск процесса flowai — НЕ os.getpid() (PID пересдаётся
# ОС между запусками, риск редкого, но реального коллизии со сброшенной
# сессией). Снимки — рабочее состояние текущего сеанса редактирования, а не
# долгоживущая история (в отличие от memory/usage/settings/episodic) —
# видеть чекпоинт из прошлого запуска в list_file_snapshots было бы просто
# путаницей, а не полезной функцией.
_SESSION_ID = uuid.uuid4().hex

_SNAPSHOT_CONN = None


def _snapshot_conn():
    global _SNAPSHOT_CONN
    if _SNAPSHOT_CONN is None:
        _SNAPSHOT_CONN = storage.connect()
        cols = {row[1] for row in _SNAPSHOT_CONN.execute("PRAGMA table_info(file_snapshots)")}
        if cols and "session_id" not in cols:
            # Снимки эфемерны по дизайну (чистятся при каждом выходе) —
            # старая схема без session_id всё равно не переживёт следующий
            # запуск, поэтому вместо ALTER-миграции просто пересоздаём
            # таблицу под новую схему.
            _SNAPSHOT_CONN.execute("DROP TABLE file_snapshots")
        _SNAPSHOT_CONN.execute(
            "CREATE TABLE IF NOT EXISTS file_snapshots ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, repo_path TEXT NOT NULL, "
            "path TEXT NOT NULL, ts TEXT NOT NULL, tool_name TEXT NOT NULL, "
            "content TEXT NOT NULL, session_id TEXT NOT NULL)"
        )
        _SNAPSHOT_CONN.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_snapshots_path_ts "
            "ON file_snapshots(path, ts)"
        )
        _SNAPSHOT_CONN.commit()
    return _SNAPSHOT_CONN


def clear_session_file_snapshots() -> None:
    """Snapshots are scoped to THIS process's run (see _SESSION_ID) — call
    once when flowai is exiting (cli.py) to drop them. Best-effort: a hard
    kill/crash skips this, leaving harmless orphaned rows tagged with a
    session_id no future process will ever reuse — nothing reads them
    without that exact id, so they're just inert until a future cleanup
    pass, never mistaken for a live checkpoint."""
    try:
        conn = _snapshot_conn()
        conn.execute("DELETE FROM file_snapshots WHERE session_id = ?", (_SESSION_ID,))
        conn.commit()
    except Exception:
        pass


def _save_file_snapshot(repo_path: str, path: str, tool_name: str) -> None:
    """Snapshot a file's content right BEFORE a mutating tool touches it.
    git_restore_file only reaches back to a committed ref — it can't return
    to an intermediate uncommitted state (edit A, then edit B, both
    uncommitted, revert only B). Snapshotting before every write_file/
    edit_file/git_restore_file call gives restore_file_snapshot real
    checkpoints to go back to, independent of git history. Best-effort: a
    missing/new/unreadable file just means no snapshot for this call — never
    blocks the actual edit that's about to happen."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return
    try:
        conn = _snapshot_conn()
        conn.execute(
            "INSERT INTO file_snapshots (repo_path, path, ts, tool_name, content, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (repo_path, path, datetime.now().isoformat(timespec="seconds"), tool_name, content, _SESSION_ID),
        )
        conn.commit()
    except Exception:
        pass


def _snapshot_before_write(tool: BaseTool, repo_path: str, path_key: str = "path") -> BaseTool:
    """Wraps write_file/edit_file/git_restore_file/copy_lines to snapshot
    the file's pre-call content — see _save_file_snapshot. Runs regardless
    of whether the call itself later succeeds; a snapshot identical to a
    failed call's no-op result is harmless. path_key differs for copy_lines
    (mutates dest_path, not path)."""
    original_coroutine = tool.coroutine
    if original_coroutine is None:
        return tool

    async def _call(**kwargs):
        path = kwargs.get(path_key)
        if path:
            _save_file_snapshot(repo_path, path, tool.name)
        return await original_coroutine(**kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_call,
        response_format=tool.response_format,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
    )


@tool
async def list_file_snapshots(path: str, limit: int = 10) -> str:
    """List saved checkpoints of a file's content, most recent first — each
    one was captured automatically right before a write_file/edit_file/
    git_restore_file call touched this file. Use this before
    restore_file_snapshot to find the right snapshot_id: e.g. the user says
    'undo my last change but keep the one before it' — list first to see
    which id corresponds to which point, don't guess an id."""
    limit = max(1, min(limit, 50))
    conn = _snapshot_conn()
    rows = conn.execute(
        "SELECT id, ts, tool_name, length(content) FROM file_snapshots "
        "WHERE path = ? AND session_id = ? ORDER BY ts DESC LIMIT ?",
        (path, _SESSION_ID, limit),
    ).fetchall()
    if not rows:
        return f"No snapshots found for {path!r} — it was never touched by write_file/edit_file/git_restore_file in this session history."
    lines = [f"id={r[0]}  ts={r[1]}  before={r[2]}  size={r[3]} chars" for r in rows]
    return "\n".join(lines)


@tool
async def restore_file_snapshot(path: str, snapshot_id: int) -> str:
    """Restore a file to an EARLIER UNCOMMITTED checkpoint saved by
    list_file_snapshots — for reverting to an intermediate edit (not just the
    last git commit). Use this instead of git_restore_file when the target
    state was never committed: e.g. the file was edited twice without a
    commit in between and only the second edit should be undone.
    git_restore_file only goes back to a git ref, discarding EVERY
    uncommitted change to the file at once — it cannot stop at a specific
    prior edit. Call list_file_snapshots(path) first to find the right
    snapshot_id; this call overwrites the file's CURRENT content, so get the
    id right before calling."""
    conn = _snapshot_conn()
    row = conn.execute(
        "SELECT content, ts, tool_name FROM file_snapshots WHERE id = ? AND path = ? AND session_id = ?",
        (snapshot_id, path, _SESSION_ID),
    ).fetchone()
    if row is None:
        return f"Error: no snapshot with id={snapshot_id} for {path!r} — call list_file_snapshots(path) to see valid ids."
    content, ts, tool_name = row
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: {e}"
    return f"Restored {path!r} to its snapshot from {ts} (taken before a {tool_name} call)."


def _revert_turn_paths(paths: set[str], turn_start_wall: str) -> list[str]:
    """Best-effort auto-revert for stream_chat's execution-failure fallback
    (see _execution_evidence_shows_failure): undoes every write/edit made
    to `paths` SINCE turn_start_wall, restoring each to its state right
    before this turn touched it. A file with no snapshot at all this turn
    was newly CREATED this turn (_save_file_snapshot only ever saves a
    PRE-EXISTING file's content) — for those, "revert" means moving the
    new file to trash instead, same recoverable mechanism as fs_extra_server
    delete_path, rather than a permanent os.remove. Never raises — this
    already runs on a failure path, and a partial revert (reported to the
    user) beats crashing the fallback itself."""
    conn = _snapshot_conn()
    reverted = []
    for path in sorted(paths):
        try:
            row = conn.execute(
                "SELECT content FROM file_snapshots WHERE path = ? AND session_id = ? "
                "AND ts >= ? ORDER BY ts ASC LIMIT 1",
                (path, _SESSION_ID, turn_start_wall),
            ).fetchone()
            if row is not None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(row[0])
                reverted.append(f"{path} — restored to its state before this turn")
            elif os.path.exists(path):
                trash_dir = storage.data_dir() / "trash"
                trash_dir.mkdir(parents=True, exist_ok=True)
                trash_path = str(trash_dir / f"{uuid.uuid4().hex[:12]}__{os.path.basename(path.rstrip('/')) or 'root'}")
                shutil.move(path, trash_path)
                reverted.append(f"{path} — created this turn, moved to trash ({trash_path})")
        except OSError:
            continue
    return reverted
