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
from storage import connect, project_dir  # noqa: E402

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


def _preview(text: str, max_chars: int = 100) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _episodic_connect():
    """rag_server.py runs as its own subprocess with its own sqlite3
    connection to the same flowai.db — episodic/writer.py's own
    CREATE TABLE IF NOT EXISTS runs in the main cli.py process and isn't
    guaranteed to have executed yet by the time this server handles its
    first tool call, so the table is (re-)declared here too rather than
    assuming that ordering."""
    conn = connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS episodic_messages ("
        "session_id TEXT NOT NULL, seq INTEGER NOT NULL, ts TEXT NOT NULL, "
        "role TEXT NOT NULL, content TEXT NOT NULL, user_id TEXT NOT NULL, "
        "PRIMARY KEY (session_id, seq))"
    )
    return conn


@mcp.tool()
async def list_episodic_sessions(limit: int = 20) -> str:
    """List past chat sessions (this project's full conversation history,
    every session ever run here) chronologically, most recent first — each
    entry has a session_id, start/end time, message count, and a preview of
    the first user message. Use this to browse/reflect on your own history
    structurally ('how many sessions have we had', 'when did we last talk',
    'what have sessions here generally been about') — unlike
    search_dialog_history (meaning-based lookup for a specific topic), this
    doesn't need you to already know what you're looking for. Pass a
    session_id from the result to read_episodic_session for the full
    transcript of one session."""
    conn = _episodic_connect()
    try:
        total = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM episodic_messages"
        ).fetchone()[0]
        if total == 0:
            return "No past sessions recorded yet."
        rows = conn.execute(
            "SELECT session_id, MIN(ts) AS started, MAX(ts) AS ended, COUNT(*) AS n "
            "FROM episodic_messages GROUP BY session_id ORDER BY started DESC LIMIT ?",
            (limit,),
        ).fetchall()
        lines = [f"{total} session(s) total, showing {len(rows)} most recent:"]
        for session_id, started, ended, n in rows:
            first_user = conn.execute(
                "SELECT content FROM episodic_messages WHERE session_id = ? "
                "AND role = 'user' ORDER BY seq ASC LIMIT 1",
                (session_id,),
            ).fetchone()
            preview = _preview(first_user[0]) if first_user else "(no user message)"
            lines.append(f"- {session_id} | {started} .. {ended} | {n} messages | {preview}")
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.tool()
async def read_episodic_session(session_id: str) -> str:
    """Read the full transcript of ONE past session (get session_id from
    list_episodic_sessions) — every user/assistant turn in original order,
    not fragmented chunks like search_dialog_history returns. Use this once
    you've identified WHICH session is relevant and want the whole
    conversation, e.g. to reflect on how a specific past task actually
    unfolded."""
    conn = _episodic_connect()
    try:
        rows = conn.execute(
            "SELECT ts, role, content FROM episodic_messages WHERE session_id = ? "
            "ORDER BY seq ASC",
            (session_id,),
        ).fetchall()
        if not rows:
            return f"No session found with id {session_id!r} — check list_episodic_sessions for valid ids."
        return "\n\n".join(f"[{ts}] {role}: {content}" for ts, role, content in rows)
    finally:
        conn.close()


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
