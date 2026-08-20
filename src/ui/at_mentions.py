"""@path mentions in plain chat text. ui/app.py's _FileSearchCompleter
already helps TYPE one (fuzzy substring autocomplete while typing) — this
is the other half: once the message is actually submitted, resolve_at_mentions
below replaces each @path that resolves to a real file (relative to
os.getcwd(), same root _FileSearchCompleter searches) with the file's real
content, inline, right where the mention was. Same "resolve before it
reaches stream_chat" pattern as ui/paste_store.py:resolve_pastes/
ui/images.py:resolve_image_paths — the model gets the real content directly
in its own message, instead of a bare path it has to decide to call
read_file on itself (and might not bother).

A token that ISN'T a real file (an email, an @handle, plain prose) is left
completely untouched — the only signal used is os.path.isfile(), no syntax
restriction on what counts as a "mention" beyond that.
"""
import os
import re

from mcp_agent.model_config import TOOL_OUTPUT_CHAR_CAP

_AT_TOKEN_RE = re.compile(r"@([^\s@]+)")
# Trailing punctuation a sentence naturally glues on ("...smth in @foo.py.")
# that's almost never actually part of the path itself.
_TRAILING_PUNCT = ".,;:!?)"


def _is_binary(path: str) -> bool:
    """Same signal as file_ops_server.py:_is_binary_file — a NUL byte in the
    first chunk reliably marks a compiled/image/binary file; kept as its own
    tiny copy here rather than importing that MCP-server module (a separate
    subprocess's code, not meant to be imported into the main process)."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return False
    return b"\x00" in chunk


def resolve_at_mentions(text: str) -> str:
    """Replaces every "@relative/path" that names a real, readable text file
    with that file's content, inline. Mirrors file_ops_server.py:read_file's
    own size guard (TOOL_OUTPUT_CHAR_CAP) — same reasoning: past that, the
    model's own tool-result cap would just truncate it anyway, so attaching
    the whole thing here only wastes context on a partial dump it can't act
    on as a whole. Over that size, leaves the mention as plain text instead
    (the model can still read it deliberately, with offset/limit, if it
    actually needs to) rather than silently cutting it — a cut-off file is
    worse than no attachment, since a partial file can look complete."""
    cwd = os.getcwd()

    def _sub(m: re.Match) -> str:
        token = m.group(1).rstrip(_TRAILING_PUNCT)
        rel = token
        path = os.path.join(cwd, rel)
        if not os.path.isfile(path):
            return m.group(0)
        if _is_binary(path):
            return f"{m.group(0)} [binary file — content not attached]"
        try:
            size = os.path.getsize(path)
            if size > TOOL_OUTPUT_CHAR_CAP:
                return (
                    f"{m.group(0)} [{size} bytes — too large to attach whole; "
                    "ask to read a specific part via read_file instead]"
                )
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            return f"{m.group(0)} [failed to read: {e}]"
        return f"{m.group(0)}\n--- {rel} ---\n{content}\n---"

    return _AT_TOKEN_RE.sub(_sub, text)
