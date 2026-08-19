"""
Generic get-or-build-with-freshness-key cache — the same hand-rolled shape
(a dict of built values, a parallel dict of the settings-tuple each one was
built from, compare-then-rebuild-under-lock) was independently implemented
4 times before this module existed: mcp_agent/agent_builder.py's
_tools_cache/_agent_cache/_role_agent_cache, mcp_agent/dnd_agent.py's
_dnd_agent_cache, and mcp_agent/router.py's _casual_agent_cache. Single home
for that shape now — each call site keeps its own idea of what the
"freshness key" tuple contains (chat_model/voice_mode/repo_path/... — see
each _get_*/_build_* pair for that), this module only owns the caching
mechanics around it.
"""
import asyncio
from typing import Awaitable, Callable


class BuildCache:
    """`key` identifies WHAT is being cached (pass a single fixed value,
    e.g. None, for a single-slot cache like a bare `_agent_cache` used to
    be). `freshness` is compared by `==` on every call — a mismatch means
    "something the caller cares about changed since this was built,
    rebuild it" (the caller decides what freshness contains; this class
    never inspects it). `build()` is only awaited while holding this
    cache's own lock, so two concurrent callers racing to build the same
    NEW key/stale entry only pay for one build, not two.

    `on_stale(old_freshness, new_freshness)` — awaited AFTER a stale entry
    is replaced, only when there WAS a previous entry for this key. Hook
    for side effects the cache itself has no business knowing about (e.g.
    evicting the OLD Ollama model when the model tag inside freshness
    changed) — receives the raw freshness values, not the built value,
    since the built value (a LangGraph agent) carries no reference back to
    what it was built from."""

    def __init__(self):
        self._values: dict = {}
        self._freshness: dict = {}
        self._lock = asyncio.Lock()

    async def get_or_build(
        self, key, freshness, build: Callable[[], Awaitable],
        *, on_stale: Callable[[object, object], Awaitable] | None = None,
    ):
        if key in self._values and self._freshness.get(key) == freshness:
            return self._values[key]
        async with self._lock:
            old_freshness = self._freshness.get(key)
            if key not in self._values or old_freshness != freshness:
                self._values[key] = await build()
                self._freshness[key] = freshness
                if old_freshness is not None and on_stale is not None:
                    await on_stale(old_freshness, freshness)
        return self._values[key]

    def values(self):
        return self._values.values()

    def __bool__(self) -> bool:
        return bool(self._values)

    def clear(self) -> None:
        self._values = {}
        self._freshness = {}
