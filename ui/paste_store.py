import re

# Threshold for converting a paste to a placeholder
_LINES_THRESHOLD = 3
_CHARS_THRESHOLD = 200

_store: dict[int, str] = {}
_counter = 0

# Format: [📄 42стр·#1]  —  line-count + unique id so multiple pastes with same length don't collide
_RE = re.compile(r'\[📄 (\d+)стр·#(\d+)\]')


def is_large(text: str) -> bool:
    return text.count('\n') >= _LINES_THRESHOLD or len(text) >= _CHARS_THRESHOLD


def store_paste(text: str) -> str:
    global _counter
    _counter += 1
    _store[_counter] = text
    lines = text.count('\n') + 1
    return f"[📄 {lines}стр·#{_counter}]"


def resolve_pastes(text: str) -> str:
    """Replace all paste placeholders with their actual content."""
    def _sub(m: re.Match) -> str:
        paste_id = int(m.group(2))
        return _store.get(paste_id, m.group(0))
    return _RE.sub(_sub, text)


def placeholder_before_cursor(text_before: str) -> str | None:
    """Return the placeholder string if the cursor is right after one."""
    m = _RE.search(text_before)
    if m and m.end() == len(text_before):
        return m.group()
    return None


def placeholder_after_cursor(text_after: str) -> str | None:
    """Return the placeholder string if the cursor is right before one."""
    m = _RE.match(text_after)
    return m.group() if m else None


def clear_store() -> None:
    global _counter
    _store.clear()
    _counter = 0
