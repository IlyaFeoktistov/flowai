import difflib
import json
import re
from typing import Any


def unified_diff_at(old_lines: list[str], new_lines: list[str], path: str, start_line: int) -> str:
    """Like difflib.unified_diff, but hunk headers ("@@ -N,M +N,M @@") use
    real 1-indexed file line numbers instead of always starting at 1 —
    difflib.unified_diff has no offset parameter, and a hunk that always
    reads "@@ -1 +1 @@" regardless of whether the edit was at line 5 or
    line 549 forces whoever's reading the diff to re-derive the real
    location themselves. get_grouped_opcodes gives (tag, i1, i2, j1, j2)
    per hunk with i/j indexed into old_lines/new_lines (0-based, relative
    to start_line) — offsetting by start_line converts those into the
    file's actual line numbers."""
    sm = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    groups = list(sm.get_grouped_opcodes(3))
    if not groups:
        return ""
    out = [f"--- {path} (before)\n", f"+++ {path} (after)\n"]
    for group in groups:
        old_start = start_line + group[0][1]
        old_len = group[-1][2] - group[0][1]
        new_start = start_line + group[0][3]
        new_len = group[-1][4] - group[0][3]
        old_range = f"{old_start}" if old_len == 1 else f"{old_start},{old_len}"
        new_range = f"{new_start}" if new_len == 1 else f"{new_start},{new_len}"
        out.append(f"@@ -{old_range} +{new_range} @@\n")
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                out.extend(" " + l for l in old_lines[i1:i2])
                continue
            if tag in ("replace", "delete"):
                out.extend("-" + l for l in old_lines[i1:i2])
            if tag in ("replace", "insert"):
                out.extend("+" + l for l in new_lines[j1:j2])
    return "".join(out)


def strip_json_comments(text: str) -> str:
    """Removes // line comments and /* */ block comments from near-JSON text,
    WITHOUT touching // that appears inside a string value (e.g. https://...).
    LLMs asked for "strict JSON" occasionally still add explanatory comments —
    a plain regex on // would also mangle URLs, so this walks the string
    tracking whether we're inside a quoted string."""
    out = []
    i, n = 0, len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            nl = text.find("\n", i)
            i = nl if nl != -1 else n
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = (end + 2) if end != -1 else n
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def extract_json_object(text: str) -> str:
    """Extracts the first balanced {...} object from text, ignoring braces
    inside string literals. Tolerates markdown code fences and stray prose
    before/after the JSON, which strict json.loads() does not."""
    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return text[start:]


def _drop_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_json_loose(text: str) -> Any | None:
    """Best-effort JSON parsing for LLM output that is *supposed* to be pure
    JSON but sometimes isn't: inline comments, markdown fences, leading/
    trailing prose, or a trailing comma. Returns None if nothing works —
    callers must have their own fallback, this never raises."""
    if not text:
        return None

    candidates = [text]
    cleaned = extract_json_object(strip_json_comments(text))
    candidates.append(cleaned)
    candidates.append(_drop_trailing_commas(cleaned))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def dict_or_str_to_args(raw_args) -> dict:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except Exception:
            return {}
    return dict(raw_args)


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains images."""
    return any("images" in msg and msg["images"] for msg in messages)

