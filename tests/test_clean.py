"""clean.py:/clean — reclaims disk space that accumulates over the app's
whole lifetime and nothing else ever purges (run-logs, MCP subprocess
stderr logs, delete_path's trash, orphaned file_snapshots rows from
crashed sessions, per-project rag_index caches). Isolated from the real
~/.local/share/flowai/ by monkeypatching storage.data_dir() and clean's
own captured log-dir constants, not FLOWAI_DATA_DIR — those constants are
resolved once at import time, before any env var override could apply."""
import os

import pytest

import clean
import storage


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "data_dir", lambda: tmp_path)
    run_logs = tmp_path / "run-logs"
    mcp_logs = tmp_path / "mcp-logs"
    run_logs.mkdir()
    mcp_logs.mkdir()
    monkeypatch.setattr(clean, "_RUN_LOG_DIR", str(run_logs))
    monkeypatch.setattr(clean, "_MCP_STDERR_LOG_DIR", str(mcp_logs))
    return tmp_path


def test_report_does_not_delete_anything(isolated_data_dir):
    (isolated_data_dir / "run-logs" / "a.jsonl").write_text("{}")
    report = clean.run_clean(None)
    assert "logs" in report
    assert os.path.exists(isolated_data_dir / "run-logs" / "a.jsonl")


def test_clean_logs_removes_both_log_dirs(isolated_data_dir):
    (isolated_data_dir / "run-logs" / "a.jsonl").write_text("{}")
    (isolated_data_dir / "mcp-logs" / "bash.log").write_text("log line")
    result = clean.run_clean("logs")
    assert "удалено 2 файлов" in result
    assert not os.path.exists(isolated_data_dir / "run-logs")
    assert not os.path.exists(isolated_data_dir / "mcp-logs")


def test_clean_trash_empties_dir_and_table(isolated_data_dir):
    trash_dir = isolated_data_dir / "trash"
    trash_dir.mkdir()
    (trash_dir / "deadbeef__foo.py").write_text("old content")
    conn = storage.connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trash (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "original_path TEXT NOT NULL, trash_path TEXT NOT NULL, ts TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO trash (original_path, trash_path, ts) VALUES (?, ?, ?)",
        ("foo.py", str(trash_dir / "deadbeef__foo.py"), "2026-08-19T00:00:00"),
    )
    conn.commit()
    conn.close()

    clean.run_clean("trash")

    assert not trash_dir.exists()
    conn = storage.connect()
    assert conn.execute("SELECT COUNT(*) FROM trash").fetchone()[0] == 0
    conn.close()


def test_clean_snapshots_keeps_current_session_drops_others(isolated_data_dir):
    conn = storage.connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS file_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "repo_path TEXT NOT NULL, path TEXT NOT NULL, ts TEXT NOT NULL, "
        "tool_name TEXT NOT NULL, content TEXT NOT NULL, session_id TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO file_snapshots (repo_path, path, ts, tool_name, content, session_id) "
        "VALUES ('/p', 'f.py', '2026-08-19T00:00:00', 'write_file', 'old', ?)",
        (clean._SESSION_ID,),
    )
    conn.execute(
        "INSERT INTO file_snapshots (repo_path, path, ts, tool_name, content, session_id) "
        "VALUES ('/p', 'g.py', '2026-08-19T00:00:00', 'write_file', 'stale', 'a-dead-session')",
    )
    conn.commit()
    conn.close()

    clean.run_clean("snapshots")

    conn = storage.connect()
    rows = conn.execute("SELECT path, session_id FROM file_snapshots").fetchall()
    conn.close()
    assert rows == [("f.py", clean._SESSION_ID)]


def test_clean_projects_wipes_the_whole_tree(isolated_data_dir):
    proj = isolated_data_dir / "projects" / "somehash" / "rag_index"
    proj.mkdir(parents=True)
    (proj / "dialog.json").write_text("[]")

    clean.run_clean("projects")

    assert not (isolated_data_dir / "projects").exists()


def test_unknown_scope_reports_error_without_touching_disk(isolated_data_dir):
    (isolated_data_dir / "run-logs" / "a.jsonl").write_text("{}")
    result = clean.run_clean("bogus")
    assert "Неизвестная категория" in result
    assert os.path.exists(isolated_data_dir / "run-logs" / "a.jsonl")


def test_all_runs_every_cleaner(isolated_data_dir, monkeypatch):
    calls = []
    for name in clean.SCOPES:
        monkeypatch.setitem(clean._CLEANERS, name, lambda n=name: calls.append(n) or f"{n} done")
    result = clean.run_clean("all")
    assert set(calls) == set(clean.SCOPES)
    for name in clean.SCOPES:
        assert f"{name} done" in result
