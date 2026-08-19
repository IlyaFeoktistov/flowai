"""
Синхронный доступ к тому же SQLite 'memory'-хранилищу, что и memory/
sqlite_store.py (тот же key -> JSON-blob контракт) — специально БЕЗ async.
SQLiteMemoryStore.load/save синхронны ВНУТРИ (обычные self._conn.execute),
просто обёрнуты в `async def` ради интерфейса MCP-тула (update_memory/
get_knowledge и т.п.) — у /memory (curses-меню, см. ui/tui/memory_view.py)
тот же принцип, что у /settings: синхронный код прямо на потоке главного
event loop (см. model_lifecycle.py про то, почему asyncio.run() там
небезопасен для async-объектов агента). Для этих сорутин, которые внутри
и так ничего не ждут, никакого моста не нужно — читаем/пишем ровно ту же
таблицу напрямую, минуя async-обёртку целиком.

Две отдельные области, как и в самих MCP-серверах:
- факты о пользователе (DEFAULT_USER-строка, ключ "facts") — memory_server.py
- знания о ТЕКУЩЕМ проекте (project:<abspath>-строка, ключ "knowledge") —
  knowledge_server.py, тот же принцип скоупинга по cwd.
"""
import json
import os
from datetime import datetime

from memory import DEFAULT_USER
from storage import connect


def _project_key() -> str:
    return f"project:{os.path.abspath(os.getcwd())}"


def _load(key: str) -> dict:
    conn = connect()
    row = conn.execute("SELECT data FROM memory WHERE key = ?", (key,)).fetchone()
    if row is None:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def _save(key: str, data: dict) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO memory (key, data) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET data = excluded.data",
        (key, json.dumps(data, ensure_ascii=False)),
    )
    conn.commit()


def get_facts() -> list[str]:
    return _load(DEFAULT_USER).get("facts", [])


def delete_fact(index: int) -> bool:
    data = _load(DEFAULT_USER)
    facts = data.get("facts", [])
    if not (0 <= index < len(facts)):
        return False
    facts.pop(index)
    data["facts"] = facts
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _save(DEFAULT_USER, data)
    return True


def clear_facts() -> int:
    data = _load(DEFAULT_USER)
    n = len(data.get("facts", []))
    if n:
        data["facts"] = []
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        _save(DEFAULT_USER, data)
    return n


def get_knowledge() -> list[tuple[str, str, str]]:
    """[(category, key, value), ...] для ТЕКУЩЕГО проекта (os.getcwd()),
    расплющенное под один список для curses-меню."""
    knowledge = _load(_project_key()).get("knowledge", {})
    return [(cat, k, v) for cat, entries in knowledge.items() for k, v in entries.items()]


def delete_knowledge_entry(category: str, key: str) -> bool:
    pkey = _project_key()
    data = _load(pkey)
    knowledge = data.get("knowledge", {})
    if category not in knowledge or key not in knowledge[category]:
        return False
    del knowledge[category][key]
    if not knowledge[category]:
        del knowledge[category]
    data["knowledge"] = knowledge
    _save(pkey, data)
    return True


def clear_knowledge() -> int:
    pkey = _project_key()
    data = _load(pkey)
    knowledge = data.get("knowledge", {})
    n = sum(len(entries) for entries in knowledge.values())
    if n:
        data["knowledge"] = {}
        _save(pkey, data)
    return n


def clear_all() -> dict:
    """Полностью стирает факты о пользователе И знания о текущем проекте —
    вызывается кнопкой "удалить всё" ПОСЛЕ подтверждения в UI, не отсюда."""
    return {"facts": clear_facts(), "knowledge": clear_knowledge()}
