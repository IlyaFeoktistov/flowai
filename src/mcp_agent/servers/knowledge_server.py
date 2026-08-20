"""
Кастомный MCP-сервер: knowledge (структурированные знания о проекте).

Отдельно от memory_server.py: тот — плоский список фактов о ПОЛЬЗОВАТЕЛЕ
+ recap-строка, хранится глобально под DEFAULT_USER. Этот — категоризированная
база знаний о ПРОЕКТЕ (архитектура, решения, конвенции) — category -> key ->
value. Тот же общий SQLite-store (memory/), но под отдельным ключом на
каждый проект — иначе при централизованном хранилище знания разных проектов
легли бы в одну и ту же строку и перезатирали друг друга.

Сама логика (формат данных, project-ключ) живёт в mcp_agent/knowledge.py —
общая с agent.py, который читает/пишет knowledge НАПРЯМУЮ (без MCP-подпроцесса)
для auto-inject/auto-capture (см. его докстринг). Этот файл — только тонкая
MCP-обёртка поверх неё для случая, когда МОДЕЛЬ сама решает позвать
get_knowledge/update_knowledge.

Запуск: python3 -m mcp_agent.servers.knowledge_server
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_agent.knowledge import format_knowledge, load_knowledge, save_knowledge_entry, search_knowledge  # noqa: E402

mcp = FastMCP("knowledge")

# Сервер запускается с cwd=repo_path (см. build_mcp_connections) — os.getcwd()
# здесь надёжно указывает на проект пользователя, а не на каталог установки
# flowAI.
_REPO_PATH = os.getcwd()


@mcp.tool()
async def update_knowledge(category: str, key: str, value: str) -> str:
    """Remember a structured piece of knowledge about the PROJECT (not the
    user — use 'memory' tools for that). category groups related entries,
    e.g. 'architecture', 'decisions', 'conventions'. Overwrites any existing
    entry with the same category+key."""
    if not category or not key:
        return "Error: category and key are required"
    await save_knowledge_entry(_REPO_PATH, category, key, value)
    return f"Remembered under [{category}] {key}: {value}"


@mcp.tool()
async def get_knowledge(category: str = "", query: str = "") -> str:
    """Get structured knowledge about the project — call this FIRST,
    before investigating anything from scratch. Two independent ways to
    use it:
    - `category` — EXACT match against however update_knowledge originally
      filed it (e.g. 'architecture', 'decisions', 'conventions'). Use this
      only if you already know the real category name from a previous
      call or an update_knowledge you made yourself.
    - `query` — free-text search instead: case-insensitive substring match
      across EVERY stored entry's category, key, AND value text, matches
      returned regardless of which category they're actually filed under.
      Use this whenever you don't already know exact category names —
      there's no need to call get_knowledge once to list categories and
      again to filter; one query call does both.
    Leave both empty to get everything across all categories."""
    knowledge = await load_knowledge(_REPO_PATH)
    if query.strip():
        return search_knowledge(knowledge, query.strip())
    return format_knowledge(knowledge, category)


if __name__ == "__main__":
    mcp.run()
