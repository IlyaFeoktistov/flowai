"""
Кастомный MCP-сервер: memory (персистентные факты о пользователе).

Готовый community "Memory" MCP-сервер строит knowledge graph — другая
модель данных, не совместимая с нашим простым списком фактов +
recap-строкой. Переиспользуем существующий memory/ (SQLiteMemoryStore) как
есть — меняется только транспорт (MCP вместо прямого импорта), не формат
хранения.

Запуск: python3 -m mcp_agent.servers.memory_server
"""
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from memory import get_store, DEFAULT_USER  # noqa: E402

mcp = FastMCP("memory")


@mcp.tool()
async def update_memory(facts: list[str], action: str = "add") -> str:
    """Remember important facts about the user. Call automatically when you
    learn: name, role, project, preferences. action: add (default) | replace.
    Not for code/project investigation — unrelated to diffs, bugs, or files."""
    store = get_store()
    data = await store.load(DEFAULT_USER)

    if not facts:
        return "Error: facts list is empty"

    existing: list[str] = data.get("facts", [])
    if action == "replace":
        updated = facts
    else:
        existing_lower = {f.lower() for f in existing}
        updated = existing + [f for f in facts if f.lower() not in existing_lower]

    data["facts"] = updated
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    await store.save(DEFAULT_USER, data)

    return f"Remembered ({len(updated)} facts): " + "; ".join(updated)


@mcp.tool()
async def list_memory() -> str:
    """Get all remembered facts about the user. Only relevant when the task
    is about the user themselves — never call this while investigating code,
    diffs, bugs, or project state, it has no bearing on those."""
    # Раньше сюда же подмешивался data["recap"] — но recap это НЕ факт о
    # пользователе, это UI-only сводка недавней переписки для футера
    # (compress.py:compress_history -> ui/app.py "※ recap: ..."), которая
    # хранится в той же строке SQLite просто по удобству storage-слоя. Живой
    # прогон: агент на задаче "разбери дифф и найди баги" вызвал list_memory
    # без явной причины, получил recap от СОВЕРШЕННО другого, старого
    # разговора ("перечисли факты про числа 0-16") и полностью съехал на
    # него, ответив про числа вместо диффа — тул с описанием "факты о
    # пользователе" тихо протащил контент чужой, не относящейся к задаче
    # темы. Отдаём только facts, как и заявлено в докстринге.
    store = get_store()
    data = await store.load(DEFAULT_USER)
    facts = data.get("facts", [])
    if not facts:
        return "No facts remembered yet."
    return "facts:\n" + "\n".join(f"- {f}" for f in facts)


if __name__ == "__main__":
    mcp.run()
