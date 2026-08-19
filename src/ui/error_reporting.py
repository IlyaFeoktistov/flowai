"""install_background_exception_handler — routes exceptions from orphaned
asyncio background tasks (nobody directly awaits them, so they'd otherwise
never surface anywhere a human would see them) through the same console
the rest of the app already prints through, instead of asyncio's default
handler dumping a raw traceback straight to real stderr.

Kept in its own module, not inline in cli.py: cli.py rewires sys.stdout/
sys.stderr to UTF-8-safe TextIOWrappers at import time (line ~16), which
conflicts with pytest's own stdout/stderr capture machinery and breaks
capture teardown ("ValueError: I/O operation on closed file") the moment
a test imports cli.py — this function has no such baggage and can be
exercised directly.

Live incident: an MCP server's stdio background reader task raised a
pydantic ValidationError parsing a malformed JSON-RPC frame long after the
tool call that started that connection had already returned. The user saw
the raw traceback flash on screen and then vanish — the TUI's next redraw
overwrote it before anyone could read past the first few lines, and
nothing about it was recorded anywhere."""
import asyncio
import traceback

from mcp_agent.debug_log import log_event
from ui.console import console


def install_background_exception_handler() -> None:
    loop = asyncio.get_running_loop()

    def _handler(loop: asyncio.AbstractEventLoop, context: dict) -> None:
        exc = context.get("exception")
        detail = (
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if exc is not None
            else context.get("message", "unknown error")
        )
        log_event("background_exception", detail=detail)
        console.print(f"\n[red] ✗ Необработанная фоновая ошибка (см. лог): {detail.strip().splitlines()[-1]}[/]")

    loop.set_exception_handler(_handler)
