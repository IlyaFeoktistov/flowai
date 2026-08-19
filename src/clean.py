"""
/clean — reclaims disk space that accumulates over the app's whole
lifetime and nothing else ever purges on its own: diagnostic logs
(mcp_agent/debug_log.py's run-logs, mcp_agent/config.py's per-subprocess
stderr logs), deleted-file trash (file_ops_server.py's delete_path),
orphaned file snapshots from crashed sessions (mcp_agent/snapshots.py —
its own docstring already flagged these as waiting on "a future cleanup
pass"), and per-project semantic search indices (storage.project_dir()).

Not the same command as /clear (cli.py) — /clear resets the CURRENT
session's chat history/screen, entirely in-memory, nothing on disk. /clean
never touches chat history, only the accumulated-junk categories above.

No args: dry-run report only (sizes/counts per category), never deletes —
a command whose whole job is deleting things needs an explicit target, not
a bare invocation that silently wipes everything. `scope` picks one
category or "all".
"""
import os
import shutil

import storage
from mcp_agent.config import _LOG_DIR as _MCP_STDERR_LOG_DIR
from mcp_agent.debug_log import _LOG_DIR as _RUN_LOG_DIR
from mcp_agent.snapshots import _SESSION_ID

SCOPES = ("logs", "trash", "snapshots", "projects")


def _dir_size_and_count(path) -> tuple[int, int]:
    total = 0
    count = 0
    try:
        for name in os.listdir(path):
            p = os.path.join(path, name)
            try:
                total += os.path.getsize(p)
                count += 1
            except OSError:
                continue
    except OSError:
        pass
    return total, count


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _clean_logs() -> str:
    total = 0
    removed = 0
    for d in (_RUN_LOG_DIR, _MCP_STDERR_LOG_DIR):
        size, count = _dir_size_and_count(d)
        total += size
        removed += count
        try:
            shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    return f"Логи: удалено {removed} файлов, освобождено {_fmt_size(total)}."


def _clean_trash() -> str:
    trash_dir = storage.data_dir() / "trash"
    size, count = _dir_size_and_count(trash_dir)
    conn = storage.connect()
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trash ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, original_path TEXT NOT NULL, "
            "trash_path TEXT NOT NULL, ts TEXT NOT NULL)"
        )
        n_rows = conn.execute("SELECT COUNT(*) FROM trash").fetchone()[0]
        conn.execute("DELETE FROM trash")
        conn.commit()
    finally:
        conn.close()
    shutil.rmtree(trash_dir, ignore_errors=True)
    return f"Корзина: удалено {count} файлов ({n_rows} записей), освобождено {_fmt_size(size)}."


def _clean_snapshots() -> str:
    """Only rows from OTHER (crashed/killed) sessions — the live session's
    own rows are cleared on clean exit by clear_session_file_snapshots(),
    and deleting them mid-session would break /gen_model-style mid-turn
    restores still in flight right now."""
    conn = storage.connect()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(file_snapshots)")}
        if not cols:
            return "Снимки файлов: нечего чистить (таблица ещё не создана)."
        n = conn.execute(
            "SELECT COUNT(*) FROM file_snapshots WHERE session_id != ?", (_SESSION_ID,)
        ).fetchone()[0]
        conn.execute("DELETE FROM file_snapshots WHERE session_id != ?", (_SESSION_ID,))
        conn.commit()
    finally:
        conn.close()
    return f"Снимки файлов от прошлых (незавершённых) сессий: удалено {n} записей."


def _clean_projects() -> str:
    """storage.project_dir() hashes the repo path one-way (no reverse
    mapping is kept anywhere), so there's no way to tell WHICH project
    hashes still correspond to a real, currently-existing directory on
    disk — wiping the whole tree is the only option. Safe to do: every
    project subdirectory is a lazily-rebuilt cache (rag_index/ — code/
    dialog/external-page search index), rebuilt automatically the next
    time each project is opened / reindex_code_search runs, never the only
    copy of anything."""
    projects_dir = storage.data_dir() / "projects"
    total = 0
    count = 0
    if projects_dir.is_dir():
        for root, _dirs, files in os.walk(projects_dir):
            for name in files:
                p = os.path.join(root, name)
                try:
                    total += os.path.getsize(p)
                    count += 1
                except OSError:
                    continue
    shutil.rmtree(projects_dir, ignore_errors=True)
    return f"Индексы проектов (rag_index): удалено {count} файлов, освобождено {_fmt_size(total)}. Пересоздаются автоматически при следующем обращении."


_CLEANERS = {
    "logs": _clean_logs,
    "trash": _clean_trash,
    "snapshots": _clean_snapshots,
    "projects": _clean_projects,
}


def _report() -> str:
    lines = ["[dim]Ничего не удалено — это отчёт. Запусти /clean <категория> или /clean all.[/]\n"]
    run_size, run_count = _dir_size_and_count(_RUN_LOG_DIR)
    mcp_size, mcp_count = _dir_size_and_count(_MCP_STDERR_LOG_DIR)
    lines.append(f"[bold]logs[/]      — {run_count + mcp_count} файлов, {_fmt_size(run_size + mcp_size)} (диагностические логи ходов и MCP-подпроцессов)")

    trash_size, trash_count = _dir_size_and_count(storage.data_dir() / "trash")
    lines.append(f"[bold]trash[/]     — {trash_count} файлов, {_fmt_size(trash_size)} (удалённые тулом delete_path файлы/папки)")

    conn = storage.connect()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(file_snapshots)")}
        n_snap = (
            conn.execute("SELECT COUNT(*) FROM file_snapshots WHERE session_id != ?", (_SESSION_ID,)).fetchone()[0]
            if cols else 0
        )
    finally:
        conn.close()
    lines.append(f"[bold]snapshots[/] — {n_snap} записей от завершённых сессий")

    proj_dir = storage.data_dir() / "projects"
    proj_size = 0
    if proj_dir.is_dir():
        for root, _dirs, files in os.walk(proj_dir):
            for name in files:
                try:
                    proj_size += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    lines.append(f"[bold]projects[/]  — {_fmt_size(proj_size)} (индексы семантического поиска по проектам, gen3d rag_index)")

    return "\n".join(lines)


def run_clean(scope: str | None) -> str:
    if not scope:
        return _report()
    scope = scope.strip().lower()
    if scope == "all":
        return "\n".join(cleaner() for cleaner in _CLEANERS.values())
    cleaner = _CLEANERS.get(scope)
    if cleaner is None:
        return f"Неизвестная категория {scope!r}. Доступные: {', '.join(SCOPES)}, all."
    return cleaner()
