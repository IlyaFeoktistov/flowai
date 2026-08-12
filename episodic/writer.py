import uuid
from datetime import datetime

from memory import DEFAULT_USER
from storage import connect


class EpisodicWriter:
    """Персистентная история диалога — переживает выход из cli.py, в отличие
    от messages: list[dict] в памяти процесса. Одна строка в SQLite на
    сообщение, commit сразу же (переживает креш процесса). Формат не связан
    с compress_history — episodic хранит полную неизменную историю,
    compress_history сжимает только рабочую копию в памяти текущей сессии."""

    def __init__(self):
        self._conn = connect()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS episodic_messages ("
            "session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL, user_id TEXT NOT NULL, "
            "PRIMARY KEY (session_id, seq))"
        )
        self._conn.commit()
        self._session_id: str | None = None
        self._seq = 0

    def new_session(self) -> str:
        self._session_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self._seq = 0
        return self._session_id

    def append(self, role: str, content: str) -> dict:
        if self._session_id is None:
            self.new_session()
        entry = {
            "session_id": self._session_id,
            "seq": self._seq,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
            "user_id": DEFAULT_USER,
        }
        self._conn.execute(
            "INSERT INTO episodic_messages "
            "(session_id, seq, ts, role, content, user_id) VALUES (?, ?, ?, ?, ?, ?)",
            (entry["session_id"], entry["seq"], entry["ts"],
             entry["role"], entry["content"], entry["user_id"]),
        )
        self._conn.commit()
        self._seq += 1
        return entry

    def close(self) -> None:
        self._conn.close()
