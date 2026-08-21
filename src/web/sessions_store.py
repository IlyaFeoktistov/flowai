"""Reads episodic_messages (episodic/writer.py) back out for the web UI's
session sidebar — only "user"/"assistant" rows are real chat history; under
DEBUG=1 the same table also collects raw pipeline events (cli.py's
_DEBUG_SKIP_EVENTS comment), which must stay out of a reconstructed
conversation."""
import storage

_CHAT_ROLES = ("user", "assistant")


def _ensure_titles_table(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session_titles ("
        "session_id TEXT PRIMARY KEY, title TEXT NOT NULL)"
    )


def save_title(session_id: str, title: str) -> None:
    """Generated once, after a brand-new session's first turn (see
    main.py's process_turns + mcp_agent/router.py:generate_session_title)
    — not for every turn, and not backfilled for sessions that predate this
    feature (list_sessions falls back to the raw first-message excerpt for
    those, see below)."""
    conn = storage.connect()
    try:
        _ensure_titles_table(conn)
        conn.execute(
            "INSERT INTO session_titles (session_id, title) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET title = excluded.title",
            (session_id, title),
        )
        conn.commit()
    finally:
        conn.close()


def list_sessions(limit: int = 200) -> list[dict]:
    conn = storage.connect()
    try:
        _ensure_titles_table(conn)
        rows = conn.execute(
            "SELECT session_id, MIN(ts), MAX(ts), COUNT(*) FROM episodic_messages "
            "WHERE role IN (?, ?) GROUP BY session_id ORDER BY MAX(ts) DESC LIMIT ?",
            (*_CHAT_ROLES, limit),
        ).fetchall()
        sessions = []
        for session_id, started_at, last_at, count in rows:
            title_row = conn.execute(
                "SELECT title FROM session_titles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if title_row:
                preview = title_row[0]
            else:
                preview_row = conn.execute(
                    "SELECT content FROM episodic_messages "
                    "WHERE session_id = ? AND role = 'user' ORDER BY seq ASC LIMIT 1",
                    (session_id,),
                ).fetchone()
                preview = preview_row[0][:200] if preview_row else ""
            sessions.append({
                "session_id": session_id,
                "started_at": started_at,
                "last_at": last_at,
                "message_count": count,
                "preview": preview,
            })
        return sessions
    finally:
        conn.close()


def get_session(session_id: str) -> list[dict]:
    conn = storage.connect()
    try:
        rows = conn.execute(
            "SELECT role, content, ts FROM episodic_messages "
            "WHERE session_id = ? AND role IN (?, ?) ORDER BY seq ASC",
            (session_id, *_CHAT_ROLES),
        ).fetchall()
        return [{"role": role, "content": content, "ts": ts} for role, content, ts in rows]
    finally:
        conn.close()


def next_seq(session_id: str) -> int:
    """One past the highest seq already stored for session_id (across ALL
    roles, not just chat ones — DEBUG-mode event rows share the same
    sequence counter) — 0 if the session doesn't exist yet."""
    conn = storage.connect()
    try:
        row = conn.execute(
            "SELECT MAX(seq) FROM episodic_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return (row[0] + 1) if row and row[0] is not None else 0
    finally:
        conn.close()
