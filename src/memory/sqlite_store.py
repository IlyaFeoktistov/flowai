import json

from .base import MemoryStore
from storage import connect


class SQLiteMemoryStore(MemoryStore):
    """Тот же key -> JSON-blob контракт, что и FileMemoryStore (см.
    memory/__init__.py), только поверх SQLite вместо плоского файла."""

    def __init__(self):
        self._conn = connect()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS memory (key TEXT PRIMARY KEY, data TEXT NOT NULL)"
        )
        self._conn.commit()

    async def load(self, user_id: str) -> dict:
        row = self._conn.execute(
            "SELECT data FROM memory WHERE key = ?", (user_id,)
        ).fetchone()
        if row is None:
            return {}
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return {}

    async def save(self, user_id: str, data: dict) -> None:
        self._conn.execute(
            "INSERT INTO memory (key, data) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET data = excluded.data",
            (user_id, json.dumps(data, ensure_ascii=False)),
        )
        self._conn.commit()
