"""
Кастомный MCP-сервер: web_search (локальный SearXNG, с фоллбэком на DDG).

В официальном/community MCP-реестре нет сервера под конкретно наш локальный
SearXNG — общих "brave-search"/"duckduckgo" серверов под чужие платные API
достаточно, но не под self-hosted SearXNG. Поэтому свой, переиспользуя
логику из tools/web_search.py как есть.

read_page НЕ здесь — для чтения страниц используем готовый mcp-server-fetch
(официальный, PyPI), он справляется лучше нашего httpx+HTMLParser фоллбэка.

Запуск: python3 -m mcp_agent.servers.web_search_server
"""
import asyncio
import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web_search")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")


def _format_results(items: list[dict], title_key: str, url_key: str, snippet_key: str) -> str:
    lines = []
    for i, item in enumerate(items[:6], 1):
        title = item.get(title_key, "Untitled")
        url = item.get(url_key, "")
        snippet = (item.get(snippet_key) or "")[:250]
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")
    return "\n\n".join(lines)


async def _search_ddg(query: str) -> str:
    from duckduckgo_search import DDGS

    def _run() -> list[dict]:
        return list(DDGS().text(query, max_results=6))

    results = await asyncio.get_event_loop().run_in_executor(None, _run)
    if not results:
        return "No results found"
    return "[DDG] " + _format_results(results, "title", "href", "body")


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the internet using the local SearXNG instance (falls back to
    DuckDuckGo if SearXNG is unavailable). Returns titles, URLs, snippets."""
    query = query.strip()
    if not query:
        return "Error: search query not specified"

    # searxng_reachable/unresponsive_engines — different failure mode than
    # "SearXNG is down": SearXNG can answer 200 OK with 0 results because
    # ALL its upstream engines are rate-limited/CAPTCHA'd at once (named in
    # the response's own "unresponsive_engines", e.g. brave/duckduckgo/
    # startpage together) — that must not be reported with the SAME
    # "SearXNG is not running" message as a real connection failure, since
    # SearXNG was never down at all.
    searxng_reachable = False
    unresponsive_engines: list = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "language": "auto"},
            )
            r.raise_for_status()
            data = r.json()
            searxng_reachable = True
            results = data.get("results", [])
            unresponsive_engines = data.get("unresponsive_engines", [])
        if results:
            return _format_results(results, "title", "url", "content")
    except (httpx.ConnectError, httpx.TimeoutException):
        pass  # SearXNG genuinely unreachable — fall through to DDG below
    except Exception as e:
        return f"Search error: {e}"

    try:
        return await _search_ddg(query)
    except ImportError:
        if searxng_reachable:
            engines_note = (
                " Engines currently blocked: "
                + ", ".join(f"{name} ({reason})" for name, reason in unresponsive_engines)
                if unresponsive_engines else ""
            )
            return (
                "SearXNG is running and responded, but found 0 results for "
                f"this query.{engines_note} duckduckgo-search fallback isn't "
                "installed either — try a different/more specific query, or "
                "wait a while (rate-limited/CAPTCHA'd engines usually "
                "recover on their own)."
            )
        return (
            f"Search unavailable: SearXNG is not running ({SEARXNG_URL}), "
            "duckduckgo-search is not installed."
        )
    except Exception as e:
        return f"DuckDuckGo error: {e}"


if __name__ == "__main__":
    mcp.run()
