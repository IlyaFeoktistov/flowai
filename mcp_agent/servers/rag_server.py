"""
Кастомный MCP-сервер: rag (семантический поиск — код/доки проекта, история
диалогов, сохранённые внешние страницы).

Отдельно от code_search (буквальный grep) и web_search/fetch (разовый,
непостоянный результат) — здесь поиск идёт по смыслу через эмбеддинги
(Ollama, nomic-embed-text) поверх собственного stdlib-хранилища (rag/).

Запуск: python3 -m mcp_agent.servers.rag_server
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from rag import EMBED_MODEL, VectorStore, embed_texts  # noqa: E402
from rag.index_code import reindex_code  # noqa: E402
from rag.index_external import remember_url as _remember_url  # noqa: E402
from storage import project_dir  # noqa: E402

mcp = FastMCP("rag")

# Сервер запускается с cwd=repo_path (см. build_mcp_connections) — os.getcwd()
# здесь надёжно указывает на проект пользователя, а не на каталог установки
# flowAI. Сам индекс — НЕ внутри repo_path (раньше был <repo_path>/rag_index/
# — засорял git status ЛЮБОГО проекта, в котором открыт flowai,
# неотслеживаемыми файлами); project_dir даёт то же per-project разделение,
# но физически вне дерева проекта пользователя (см. storage.py).
_REPO_PATH = os.getcwd()
_INDEX_DIR = str(project_dir(_REPO_PATH, "rag_index"))

_code_store: VectorStore | None = None
_dialog_store: VectorStore | None = None
_external_store: VectorStore | None = None


def _get_code_store() -> VectorStore:
    global _code_store
    if _code_store is None:
        _code_store = VectorStore.load(os.path.join(_INDEX_DIR, "code.json"), model=EMBED_MODEL)
    return _code_store


def _get_dialog_store() -> VectorStore:
    global _dialog_store
    if _dialog_store is None:
        _dialog_store = VectorStore.load(os.path.join(_INDEX_DIR, "dialog.json"), model=EMBED_MODEL)
    return _dialog_store


def _get_external_store() -> VectorStore:
    global _external_store
    if _external_store is None:
        _external_store = VectorStore.load(os.path.join(_INDEX_DIR, "external.json"), model=EMBED_MODEL)
    return _external_store


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No results found"
    return "\n---\n".join(
        f"{r['score']:.3f} | {r['metadata'].get('source', '?')}:{r['metadata'].get('chunk_idx', 0)}\n{r['text']}"
        for r in results
    )


@mcp.tool()
async def search_code_semantic(query: str, top_k: int = 5) -> str:
    """Semantic (meaning-based) search over the project's code/docs — finds
    relevant files even when your query's wording differs from the actual
    code. Use search_code (grep) for exact strings/patterns; use this for
    conceptual questions like 'where do we limit tool output size'. Requires
    reindex_code_search to have been run at least once."""
    store = _get_code_store()
    if len(store) == 0:
        return "Code index is empty — run reindex_code_search first"
    query_embedding = (await embed_texts([query]))[0]
    return _format_results(store.search(query_embedding, top_k=top_k))


@mcp.tool()
async def reindex_code_search() -> str:
    """Rebuild the semantic code/docs index from scratch. Run this once
    before using search_code_semantic, and again after significant code
    changes — it does not update automatically."""
    global _code_store
    store = _get_code_store()
    n = await reindex_code(_REPO_PATH, store)
    _code_store = store
    return f"Indexed {n} chunks from {_REPO_PATH}"


@mcp.tool()
async def search_dialog_history(query: str, top_k: int = 5) -> str:
    """Semantic search over PAST conversation turns (across sessions, not
    just the current one) — use this for 'what did we discuss about X
    earlier', beyond what the current (possibly compressed) context holds."""
    store = _get_dialog_store()
    if len(store) == 0:
        return "Dialog history index is empty — nothing indexed yet"
    query_embedding = (await embed_texts([query]))[0]
    return _format_results(store.search(query_embedding, top_k=top_k))


@mcp.tool()
async def remember_url(url: str) -> str:
    """Fetch a URL, extract its content, and save it permanently for later
    semantic search via search_external_sources. Unlike fetch/web_search
    (one-off, not retained), this is for pages worth remembering long-term —
    only call it when the user asks to remember/save a page, not for every
    fetch."""
    store = _get_external_store()
    return await _remember_url(url, store)


@mcp.tool()
async def search_external_sources(query: str, top_k: int = 5) -> str:
    """Semantic search over pages previously saved via remember_url. Does
    NOT search the live internet — use web_search/fetch for that."""
    store = _get_external_store()
    if len(store) == 0:
        return "No external sources remembered yet — use remember_url first"
    query_embedding = (await embed_texts([query]))[0]
    return _format_results(store.search(query_embedding, top_k=top_k))


if __name__ == "__main__":
    mcp.run()
