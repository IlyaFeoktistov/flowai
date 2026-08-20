"""
web_read — fetch a URL and answer ONE specific question about it via a
fresh, isolated call to the SAME already-resident chat model, instead of
returning the raw page into the main conversation. A single documentation
page can easily run past this project's whole OLLAMA_NUM_CTX budget
(model_config.py) on its own — dumping it into the live conversation risks
losing the actual task context to a compaction pass triggered by one tool
result. Same "reuse the resident model, one call, no second load"
constraint as delegate_tool.py/self_heal.py's judge call justify — simpler
than delegate() even: one plain model.ainvoke(), no create_agent/tools/
checkpointer, since there's no multi-step investigation here, just one page
and one question.

Reuses mcp_server_fetch's own HTML->markdown extraction and robots.txt
check (mcp-server-fetch is already an installed dependency — it backs the
raw `fetch` MCP tool, see config.py:build_mcp_connections) by importing its
functions directly instead of going through its stdio/MCP transport — same
rules about what's allowed to be fetched as the raw `fetch` tool, just
processed differently once fetched.
"""
import time

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from mcp_server_fetch.server import (
    DEFAULT_USER_AGENT_AUTONOMOUS,
    check_may_autonomously_fetch_url,
    fetch_url,
)

from mcp_agent.message_utils import _tool_text

# The raw page never reaches the main conversation — only the summarizer
# call below sees it — so this can be far more generous than
# TOOL_OUTPUT_CHAR_CAP (model_config.py, 20000 chars for a normal tool
# result reaching the main model). Still bounded: an unbounded page fed
# whole into one model.ainvoke() would blow past OLLAMA_NUM_CTX on its own.
_MAX_PAGE_CHARS = 60_000

# Self-cleaning cache — asking a second question about the same page (or a
# retry) shouldn't re-fetch and re-extract every time. Keyed by URL only,
# not by question — the fetch/extraction step is identical regardless of
# what's later asked about it.
_CACHE_TTL_SECONDS = 15 * 60
_page_cache: dict[str, tuple[float, str]] = {}


async def _fetch_page_content(url: str) -> str:
    cached = _page_cache.get(url)
    if cached is not None:
        fetched_at, content = cached
        if time.monotonic() - fetched_at <= _CACHE_TTL_SECONDS:
            return content
        del _page_cache[url]

    await check_may_autonomously_fetch_url(url, DEFAULT_USER_AGENT_AUTONOMOUS)
    content, prefix = await fetch_url(url, DEFAULT_USER_AGENT_AUTONOMOUS)
    content = prefix + content
    if len(content) > _MAX_PAGE_CHARS:
        content = content[:_MAX_PAGE_CHARS] + "\n\n...[page truncated at the fetch step]"
    _page_cache[url] = (time.monotonic(), content)
    return content


def build_web_read_tool(model):
    """Closure over this turn's already-built chat model — no second model
    load, same convention as build_delegate_tool (delegate_tool.py)."""

    @tool
    async def web_read(url: str, question: str) -> str:
        """Fetch a URL and get a concise answer to a SPECIFIC question about
        its content, instead of the raw page. Prefer this over the raw
        `fetch` tool by default — use `fetch` instead only when you need an
        exact quote/code snippet verbatim rather than a distilled answer.
        The page itself never enters this conversation, only the answer to
        `question` does, so write `question` as a complete, specific ask
        (e.g. "what are the constructor's required parameters and their
        default values", not just "summarize this page")."""
        try:
            content = await _fetch_page_content(url)
        except Exception as e:
            return f"Error fetching {url!r}: {e}"

        prompt = (
            f"Web page content fetched from {url}:\n---\n{content}\n---\n\n"
            f"Question: {question}\n\n"
            "Answer concisely, based only on the content above. If the "
            "content doesn't actually answer the question, say so plainly "
            "instead of guessing."
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return _tool_text(response.content) or "(model returned an empty answer)"

    return web_read
