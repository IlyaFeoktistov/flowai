import io
import sys
from rich.console import Console


class _AppProxy:
    """File-like object that forwards writes to FlowAIApp or stdout."""

    def __init__(self) -> None:
        self._app = None

    def connect(self, app) -> None:
        self._app = app

    def write(self, text: str) -> None:
        if self._app is not None:
            # Strip carriage returns and OSC title sequences — they don't render
            # in the output pane and cause corruption if passed through.
            import re as _re
            text = _re.sub(r"\x1b\][^\x07]*\x07", "", text)  # OSC sequences
            text = text.replace("\r", "")                     # carriage returns
            if text:
                self._app.write(text)
        else:
            try:
                sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
                sys.stdout.buffer.flush()
            except Exception:
                pass

    def flush(self) -> None:
        pass

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")

    def isatty(self) -> bool:
        return True


_proxy = _AppProxy()
console = Console(file=_proxy, force_terminal=True, highlight=False)


def connect_app(app) -> None:
    """Wire the Rich console to a FlowAIApp instance."""
    _proxy.connect(app)


def get_app():
    """Return the connected FlowAIApp, or None if running in plain CLI mode."""
    return _proxy._app


def safe_write(text: str) -> None:
    _proxy.write(text)


def debug_print(msg: str) -> None:
    """console.print for a DEBUG-only diagnostic line (compaction/
    agent_builder/delegate_tool/self_heal/agent.py traces) — always starts
    on its own fresh line first. These fire independently of the live
    streamed answer text (written via safe_write, tracked by
    ui/stream.py's own _last_written_was_newline) — without a leading
    newline here, a debug line printed while the cursor is still mid-line
    (an unflushed streamed answer with no trailing newline yet) visually
    glues onto its tail instead of starting on its own line."""
    safe_write("\n")
    console.print(msg)


def set_title(text: str) -> None:
    if _proxy._app is None:
        safe_write(f"\033]0;{text}\007")


def fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms}мс"
    return f"{ms / 1000:.1f}с"


def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}с"
    return f"{s // 60}м {s % 60}с"
