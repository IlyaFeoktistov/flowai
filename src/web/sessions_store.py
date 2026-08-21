"""Reads episodic_messages (episodic/writer.py) back out for the web UI's
session sidebar — only "user"/"assistant" rows are real chat history; under
DEBUG=1 the same table also collects raw pipeline events (cli.py's
_DEBUG_SKIP_EVENTS comment), which must stay out of a reconstructed
conversation."""
import json

import storage

_CHAT_ROLES = ("user", "assistant")


def _ensure_turn_traces_table(conn) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS turn_traces ("
        "session_id TEXT NOT NULL, seq INTEGER NOT NULL, events_json TEXT NOT NULL, "
        "PRIMARY KEY (session_id, seq))"
    )


def save_turn_trace(session_id: str, seq: int, events: list[dict]) -> None:
    """seq — тот же seq, что достался ASSISTANT-строке этого хода в
    episodic_messages (episodic.append()'s возвращаемый entry) — join'ится
    с ней 1:1 в get_session() ниже. events — сырой список on_event-
    payload'ов за весь ход (main.py's process_turns, за вычетом
    permission_request/ask_user_request — они нерезолвимы задним числом,
    см. её же комментарий), в точности то же, что улетело в вебсокет живьём.
    Веб-only UI-удобство поверх episodic_messages, как и session_titles —
    episodic сам хранит только финальный текст хода, полная трасса
    (тулы/delegate/thinking) раньше пропадала при переоткрытии сессии."""
    conn = storage.connect()
    try:
        _ensure_turn_traces_table(conn)
        conn.execute(
            "INSERT INTO turn_traces (session_id, seq, events_json) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id, seq) DO UPDATE SET events_json = excluded.events_json",
            (session_id, seq, json.dumps(events)),
        )
        conn.commit()
    finally:
        conn.close()


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
        _ensure_turn_traces_table(conn)
        rows = conn.execute(
            "SELECT role, content, ts, seq FROM episodic_messages "
            "WHERE session_id = ? AND role IN (?, ?) ORDER BY seq ASC",
            (session_id, *_CHAT_ROLES),
        ).fetchall()
        traces = dict(conn.execute(
            "SELECT seq, events_json FROM turn_traces WHERE session_id = ?", (session_id,),
        ).fetchall())
        out = []
        for role, content, ts, seq in rows:
            msg = {"role": role, "content": content, "ts": ts}
            # Только у ассистентских строк — see save_turn_trace's docstring
            # за тем, почему seq совпадает 1:1. Сессии старше этой фичи
            # просто не найдут своих trace-строк — get_session тогда
            # отдаёт message БЕЗ "detail", фронтенд откатывается на плоский
            # рендер (см. entities/chat's buildEntriesFromHistory).
            if role == "assistant" and seq in traces:
                msg["detail"] = json.loads(traces[seq])
            out.append(msg)
        return out
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
