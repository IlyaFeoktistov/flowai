"""
Кастомный MCP-сервер: file I/O + поиск по коду. Заменяет разом три вещи:
внешний filesystem MCP-сервер (@modelcontextprotocol/server-filesystem —
read_file/read_text_file/read_multiple_files/list_directory/directory_tree/
search_files/get_file_info/write_file/edit_file/create_directory/move_file),
code_search_server.py (search_code/read_file_range/find_files_by_name/
search_symbols/project_tree) и fs_extra_server.py (replace_lines/insert_lines/
copy_lines остаются позади — delete_path/restore_deleted_path/list_deleted_paths
переехали сюда без изменений).

Модель — read_file/write_file/edit_file/grep_search/glob_search: пять
несовпадающих по назначению тулов вместо дюжины почти-дублей. Никакого
отдельного тула для листинга директории/move/mkdir: glob_search("**/*",
path) закрывает браузинг структуры проекта, write_file сама создаёт
недостающие родительские директории — так что явный "создай папку" тул не
нужен, а move — это либо write_file(new_path, <прочитанное>)+delete_path
(old_path), либо bash.

Особенности реализации:
  - write_file/edit_file строят настоящий diff через utils/parsing.py:
    unified_diff_at (difflib.SequenceMatcher) — уже написанный в этом
    проекте генератор построчного diff'а, а не подделку вроде "весь старый
    файл как удалённые строки, весь новый — как добавленные". Модели он не
    возвращается (см. _text_result) — уходит только в structuredContent
    (CallToolResult), откуда его берёт исключительно ui/stream.py для
    рендера человеку; модель получает одну строку-подтверждение.
  - write_file/edit_file требуют свежего read_file по этому пути перед
    записью (_require_fresh_read, _last_read_mtime) — отказ, если путь
    вообще не читался в этой сессии или изменился на диске с момента
    чтения, чтобы не перезаписать вслепую то, чего модель не видела.
  - read_file/write_file/edit_file держат бинарный/размерный гард ПРЯМО
    внутри тула (NUL-байт в первых 8KB, 10MB потолок) — раньше это было
    внешней обёрткой в tool_wrappers.py, потому что чтение шло через
    сторонний npm-пакет; теперь тул свой, гард можно держать в естественном
    месте, а не снаружи.
  - grep_search/glob_search переиспользуют SKIP_DIRS/_HAS_RG/_run/_sh из
    старого code_search_server.py как есть (та же логика игнора venv-tts/
    vendor/, см. их комментарии ниже) — не переизобретены с нуля.

required_permission (TOOL_PERMISSIONS ниже) — декларативный тег
read_only/workspace_write на каждом туле, который roles.py читает при
сборке _PROJECT_READ_TOOLS/_WRITE_TOOLS вместо дублирования этого же
списка руками в двух местах.

Запуск: python3 -m mcp_agent.servers.file_ops_server
"""
import asyncio
import fnmatch
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from storage import connect, data_dir  # noqa: E402
from utils.parsing import unified_diff_at  # noqa: E402
from mcp_agent.servers._lsp_client import _get_client  # noqa: E402

mcp = FastMCP("file_ops")

# Декларативный permission-тег на каждый тул этого сервера (см. модульный
# докстринг): roles.py фильтрует по нему вместо того, чтобы перечислять
# имена тулов дважды.
TOOL_PERMISSIONS = {
    "read_file": "read_only",
    "grep_search": "read_only",
    "glob_search": "read_only",
    "list_deleted_paths": "read_only",
    "write_file": "workspace_write",
    "edit_file": "workspace_write",
    "delete_path": "workspace_write",
    "restore_deleted_path": "workspace_write",
}

# Past this, a single read/write call would dump/accept more raw content
# than any turn should reasonably carry in one tool result.
_MAX_READABLE_FILE_BYTES = 10 * 1024 * 1024
_MAX_WRITABLE_CONTENT_BYTES = 10 * 1024 * 1024

# Mirrors TOOL_OUTPUT_CHAR_CAP (mcp_agent/model_config.py, same env var) without
# importing that module — it pulls in `settings`, which probes for a CUDA GPU
# via torch at import time (settings.py:38), a multi-second cost this
# lightweight I/O subprocess has no reason to pay just to read one constant.
#
# A whole-file read (no limit) past this cap would today still be read in
# full and only THEN truncated by agent_builder.py's _cap_tool_output —
# paying the read cost for a sandwich-truncated result that's mostly filler.
# Rejecting it before the read, with a nudge toward offset/limit, is both
# cheaper and more token-economical: a throw costs a ~100-byte error message,
# while letting it through and truncating afterward costs a full cap's worth
# of mostly-useless tokens for the same overflow.
_MAX_READ_CHARS_UNBOUNDED = int(os.getenv("TOOL_OUTPUT_CHAR_CAP", "20000"))

# code_search_server.py's own list, live-bug-driven (venv-tts/venv-build-tools/
# venv-uv beyond the standard .venv — "venv*" glob, not the literal name
# "venv" — none of the standard exclusion lists in the wild catch this
# project's own extra venvs, see the original comment this was copied from).
SKIP_DIRS = ["node_modules", ".git", "vendor", "__pycache__", ".cache", "dist", "build", ".venv", "venv*", ".tox"]
_HAS_RG = shutil.which("rg") is not None
MAX_RESULTS = 60
MAX_LINE_LEN = 300
DEFAULT_TIMEOUT = 15
MAX_TIMEOUT = 120


def _is_binary_file(path: str) -> bool:
    """NUL byte in the first chunk is a reliable binary signal — real text
    files essentially never contain one, while compiled binaries/images/
    databases almost always do within the first few KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
    except OSError:
        return False
    return b"\x00" in chunk


# path (as passed by the model, unnormalized — matches how read_file/write_file/
# edit_file already key everything else in this module) -> mtime at the moment
# read_file last successfully looked at it. Lives for this subprocess's whole
# lifetime (same pattern as rag_server.py's own mtime caches), not reset per
# turn — a file read several turns ago should still count as "read" now,
# same as a human editor doesn't forget a file it opened earlier.
#
# Used by write_file/edit_file (see _require_fresh_read below) to refuse
# touching a path the model hasn't actually looked at, or that changed on
# disk since it did — so a write/edit can never blindly clobber content
# the model never saw.
_last_read_mtime: dict[str, float] = {}


def _require_fresh_read(path: str) -> str | None:
    """Guard shared by write_file/edit_file. Returns a bare error message
    (no "Error: " prefix — callers pass it through _text_error, which adds
    that) if the write should be refused, None if it's clear to proceed. A
    path that doesn't exist yet is always fine (nothing to have read) —
    this only guards against blindly overwriting/editing content never
    seen, or seen before it changed underneath the model (another process,
    a linter, the user)."""
    if not os.path.exists(path):
        return None
    last = _last_read_mtime.get(path)
    if last is None:
        return (
            f"{path!r} exists and hasn't been read yet this session — "
            "call read_file on it first so you're not overwriting content "
            "you've never actually seen."
        )
    try:
        current = os.path.getmtime(path)
    except OSError as e:
        return str(e)
    if current > last:
        return (
            f"{path!r} has changed on disk since you last read it "
            "(another process, a linter, or the user may have modified it) "
            "— call read_file again before writing."
        )
    return None


def _text_result(
    text: str,
    *,
    diff: str | None = None,
    diagnostics_text: str | None = None,
    diagnostics: list | None = None,
) -> CallToolResult:
    """Success shape for write_file/edit_file: a short confirmation for the
    model (it already knows what it just wrote — echoing the diff back would
    only cost tokens for no new information), with the full diff (if any)
    carried in structuredContent instead. That field doesn't reach the
    model's context at all — langchain-mcp-adapters exposes it as the
    ToolMessage's `artifact` — but the UI (ui/stream.py) reads it from there
    to still render the real diff for the human.

    diagnostics_text (if given) IS appended to the model-facing text — unlike
    the diff, new LSP diagnostics are exactly the kind of thing the model
    needs to see and react to in the same turn (see _diagnostics_summary).
    diagnostics (the structured list backing that text) rides in
    structuredContent alongside diff, for the UI to render separately."""
    if diagnostics_text:
        text = f"{text}\n\n{diagnostics_text}"
    structured = {}
    if diff:
        structured["diff"] = diff
    if diagnostics:
        structured["diagnostics"] = diagnostics
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        structuredContent=(structured or None),
    )


def _text_error(message: str) -> CallToolResult:
    return _text_result(f"Error: {message}")


async def _get_lsp_client_quiet(path: str):
    """Best-effort LSP client for `path`'s extension, or None if none is
    configured/installed, or it fails to start — the diagnostics-after-edit
    feature below is purely additive, it must never turn a write/edit that
    would otherwise succeed into an error."""
    try:
        return await _get_client(path)
    except Exception:
        return None


async def _diagnostics_snapshot(client, path: str) -> list[dict] | None:
    """None means "unknown" (no client, or the server didn't respond in
    time) — callers must not treat that as "no issues"."""
    if client is None:
        return None
    try:
        return await client.diagnostics_for(path)
    except Exception:
        return None


_SEVERITY_NAMES = {1: "error", 2: "warning", 3: "info", 4: "hint"}
_MAX_DIAGNOSTIC_LINES = 10


def _diag_key(d: dict) -> tuple:
    start = ((d.get("range") or {}).get("start")) or {}
    return (start.get("line"), start.get("character"), d.get("severity"), d.get("message"))


def _format_diagnostic(path: str, d: dict) -> str:
    start = ((d.get("range") or {}).get("start")) or {}
    line = (start.get("line") or 0) + 1
    char = (start.get("character") or 0) + 1
    severity = _SEVERITY_NAMES.get(d.get("severity"), "info")
    source = d.get("source")
    suffix = f" ({source})" if source else ""
    return f"{path}:{line}:{char} {severity}: {(d.get('message') or '').strip()}{suffix}"


def _diagnostics_summary(path: str, before: list[dict] | None, after: list[dict] | None) -> tuple[str, list] | None:
    """New diagnostics introduced by this write/edit — after minus before,
    compared by position+severity+message (see _diag_key; a diagnostic that
    merely shifted lines from an unrelated change above it can misread as
    "new" — an inherent limit of positional diffing, not fixed here). None
    if either side is unknown (can't safely diff) or nothing new showed up."""
    if before is None or after is None:
        return None
    seen_before = {_diag_key(d) for d in before}
    new_diags = [d for d in after if _diag_key(d) not in seen_before]
    if not new_diags:
        return None
    shown = new_diags[:_MAX_DIAGNOSTIC_LINES]
    lines = [_format_diagnostic(path, d) for d in shown]
    if len(new_diags) > len(shown):
        lines.append(f"... and {len(new_diags) - len(shown)} more")
    text = f"Found {len(new_diags)} new diagnostic issue(s):\n  " + "\n  ".join(lines)
    return text, new_diags


@mcp.tool()
async def read_file(path: str, offset: int = 0, limit: int | None = None) -> str:
    """Read a text file, optionally windowed by line — offset (0-indexed,
    how many lines to skip) and limit (how many lines to return after that).
    Omit both to read the whole file. Content comes back AS-IS, no per-line
    number prefix on it — a header states the line range instead (e.g.
    "(lines 1-50 of 320 total)"), so count from that (or use grep_search's
    own path:line:content output) to name a specific line. This keeps the
    returned text byte-for-byte identical to the file's real content — edit_file's
    old_string must match that exactly, and an embedded "N\\t" prefix
    accidentally copied into it would silently make old_string never match.
    Rejects binary files (images, compiled artifacts, .db, ...) and anything
    over 10MB outright — narrow with offset/limit or use bash's grep/head/
    tail on a huge file instead of trying to read it whole. A whole-file read
    (no limit given) is also rejected up front if the file is large enough
    that the result would just come back truncated — pass limit= for large
    files instead of omitting it."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    try:
        st = os.stat(path)
        size = st.st_size
    except OSError as e:
        return f"Error: {e}"
    if size > _MAX_READABLE_FILE_BYTES:
        return (
            f"Error: {path!r} is {size / (1024 * 1024):.1f}MB — over the "
            f"{_MAX_READABLE_FILE_BYTES // (1024 * 1024)}MB limit for a single "
            "read. Narrow with offset/limit, or use bash with grep/head/tail."
        )
    if limit is None and size > _MAX_READ_CHARS_UNBOUNDED:
        return (
            f"Error: {path!r} is {size} bytes — reading it whole would exceed "
            f"the {_MAX_READ_CHARS_UNBOUNDED}-character output cap and come "
            "back truncated from the middle anyway. Pass limit= (optionally "
            "with offset=) to read it in a bounded window, or use grep_search "
            "to jump straight to the relevant lines."
        )
    if _is_binary_file(path):
        return (
            f"Error: {path!r} looks like a binary file, not text — reading "
            "it here would only produce garbage/replacement characters."
        )

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: {e}"
    _last_read_mtime[path] = st.st_mtime

    total = len(lines)
    start = max(0, offset)
    end = total if limit is None else min(total, start + max(0, limit))
    selected = "".join(lines[start:end])
    if not selected:
        return f"(empty — file has {total} lines, offset={offset} is past the end)" if start >= total else "(empty file)"
    result = f"(lines {start + 1}-{end} of {total} total)\n{selected}"
    if end < total:
        result += f"\n... ({total - end} more lines — increase limit or offset to see the rest)"
    return result


@mcp.tool()
async def write_file(path: str, content: str) -> CallToolResult:
    """Write (create or fully overwrite) a text file — creates any missing
    parent directories automatically. Requires the file (if it already
    exists) to have been read via read_file first, with nothing changing it
    on disk since — refuses otherwise, to avoid blindly overwriting content
    never actually seen. Returns a short confirmation only, not the diff —
    the change is already known to you (you just wrote it) and is shown to
    the user separately. Rejects content over 10MB — for a file that large,
    this is very likely the wrong tool for the job.

    If a language server is configured for this file's extension (Python,
    Go, TypeScript/JavaScript, PHP — see _lsp_client.py), the result also
    reports any NEW diagnostic issues (errors/warnings) this write
    introduced, e.g. "Found 1 new diagnostic issue: path.py:12:5 error: ...".
    That's a REAL error from the project's actual language server, not a
    guess — treat it the same as a failed test or a compiler error: fix it
    with edit_file before moving on, don't just note it and continue as if
    the write succeeded cleanly. Silent if there's nothing new, or no
    language server is available for this extension — don't rely on its
    absence to mean the file is clean."""
    path = path.strip()
    if not path:
        return _text_error("path is required")
    if len(content.encode("utf-8", errors="replace")) > _MAX_WRITABLE_CONTENT_BYTES:
        return _text_error(
            f"content is {len(content) / (1024 * 1024):.1f}MB — over "
            f"the {_MAX_WRITABLE_CONTENT_BYTES // (1024 * 1024)}MB limit for "
            "a single write."
        )
    guard_error = _require_fresh_read(path)
    if guard_error:
        return _text_error(guard_error)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
        is_new = False
    except OSError:
        original = None
        is_new = True

    lsp_client = await _get_lsp_client_quiet(path)
    # A brand-new file has no "before" to diff against — skip that LSP
    # round-trip entirely, this is the common, latency-sensitive case.
    before_diags = [] if is_new else await _diagnostics_snapshot(lsp_client, path)

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return _text_error(str(e))
    _last_read_mtime[path] = os.path.getmtime(path)

    after_diags = await _diagnostics_snapshot(lsp_client, path)
    diag = _diagnostics_summary(path, before_diags, after_diags)
    diag_text, diag_list = diag if diag else (None, None)

    if is_new:
        return _text_result(
            f"Created {path!r} ({len(content.splitlines())} lines).",
            diagnostics_text=diag_text, diagnostics=diag_list,
        )
    diff = unified_diff_at((original or "").splitlines(keepends=True), content.splitlines(keepends=True), path, 1)
    if not diff:
        return _text_result(
            f"Wrote {path!r} (content unchanged).",
            diagnostics_text=diag_text, diagnostics=diag_list,
        )
    return _text_result(f"Updated {path!r}.", diff=diff, diagnostics_text=diag_text, diagnostics=diag_list)


@mcp.tool()
async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> CallToolResult:
    """Replace an exact, VERBATIM substring in a file — old_string must
    match the file's CURRENT content byte-for-byte (read the file first,
    don't reconstruct old_string from memory). By default replaces only the
    FIRST occurrence — old_string must therefore include enough surrounding
    context (a few lines) to be unique in the file, or the call fails
    asking you to narrow it; pass replace_all=true to replace every
    occurrence instead (e.g. renaming a variable used many times). Requires
    the file to have been read via read_file first, with nothing changing
    it on disk since — refuses otherwise, to avoid blindly editing content
    never actually seen. Returns a short confirmation only, not the diff —
    the change is already known to you (you specified old_string/new_string
    yourself) and is shown to the user separately.

    If a language server is configured for this file's extension (Python,
    Go, TypeScript/JavaScript, PHP — see _lsp_client.py), the result also
    reports any NEW diagnostic issues (errors/warnings) this edit
    introduced, e.g. "Found 1 new diagnostic issue: path.py:12:5 error: ...".
    That's a REAL error from the project's actual language server, not a
    guess — treat it the same as a failed test or a compiler error: fix it
    with another edit_file call before moving on, don't just note it and
    continue as if the edit succeeded cleanly. Silent if there's nothing
    new, or no language server is available for this extension — don't
    rely on its absence to mean the file is clean."""
    path = path.strip()
    if not path:
        return _text_error("path is required")
    if old_string == new_string:
        return _text_error("old_string and new_string must differ")
    guard_error = _require_fresh_read(path)
    if guard_error:
        return _text_error(guard_error)
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except OSError as e:
        return _text_error(str(e))

    count = original.count(old_string)
    if count == 0:
        return _text_error(f"old_string not found in {path!r} — re-read the file, it may not match byte-for-byte.")
    if not replace_all and count > 1:
        return _text_error(
            f"old_string appears {count} times in {path!r} — it must be "
            "unique for a single replacement. Include more surrounding context "
            "in old_string to disambiguate, or pass replace_all=true if you "
            "really want every occurrence replaced."
        )

    lsp_client = await _get_lsp_client_quiet(path)
    before_diags = await _diagnostics_snapshot(lsp_client, path)

    updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    except OSError as e:
        return _text_error(str(e))
    _last_read_mtime[path] = os.path.getmtime(path)

    after_diags = await _diagnostics_snapshot(lsp_client, path)
    diag = _diagnostics_summary(path, before_diags, after_diags)
    diag_text, diag_list = diag if diag else (None, None)

    diff = unified_diff_at(original.splitlines(keepends=True), updated.splitlines(keepends=True), path, 1)
    replacements = count if replace_all else 1
    msg = f"Edited {path!r} ({replacements} replacement{'s' if replace_all and count != 1 else ''})."
    return _text_result(msg, diff=diff, diagnostics_text=diag_text, diagnostics=diag_list)


def _sh(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


async def _run(cmd: str, timeout: int = DEFAULT_TIMEOUT) -> tuple[str, int]:
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode(errors="replace"), proc.returncode
    except asyncio.TimeoutError:
        return "(timeout)", 1
    except Exception as e:
        return f"(error: {e})", 1


def apply_limit(items: list, limit: int | None, offset: int = 0) -> tuple[list, bool, int | None, int | None]:
    """Shared count-based pagination for grep_search/glob_search, default
    limit 250. Returns (items, truncated, applied_limit, applied_offset) —
    an explicit truncated flag beats guessing from a character-count text
    marker whether there's more the model isn't seeing."""
    offset = max(0, offset)
    windowed = items[offset:]
    effective_limit = 250 if limit is None else limit
    if effective_limit == 0:
        return windowed, False, None, (offset or None)
    truncated = len(windowed) > effective_limit
    return windowed[:effective_limit], truncated, (effective_limit if truncated else None), (offset or None)


@mcp.tool()
async def grep_search(
    pattern: str,
    path: str = ".",
    glob: str = "",
    output_mode: str = "files_with_matches",
    before: int = 0,
    after: int = 0,
    context: int = 0,
    case_insensitive: bool = False,
    file_type: str = "",
    head_limit: int | None = None,
    offset: int = 0,
    multiline: bool = False,
) -> str:
    """Search file CONTENTS with a regex pattern (ripgrep-style) — the ONE
    search tool, three modes via output_mode:
      - "files_with_matches" (default) — just the list of matching file paths.
      - "content" — matching lines themselves, with before/after/context
        lines of surrounding code (use INSTEAD OF a follow-up read_file call
        when a match's own line isn't enough — free context on the same call).
      - "count" — number of matching files + total match count, no content.
    glob restricts by filename pattern (e.g. "*.py"); file_type by extension
    without the dot (e.g. "py"). head_limit/offset paginate results (default
    250 per page). multiline lets '.' match across line breaks."""
    pattern = pattern.strip()
    if not pattern:
        return "Error: pattern is required"
    path = path.strip() or "."
    effective_after = after or context
    effective_before = before or context

    if _HAS_RG:
        flags = ["rg", "--line-number", "--with-filename", "--color=never"]
        if case_insensitive:
            flags.append("--ignore-case")
        if multiline:
            flags.append("--multiline")
        if glob:
            flags += ["--glob", glob]
        if file_type:
            flags += ["--glob", f"*.{file_type}"]
        if output_mode == "content" and (effective_before or effective_after):
            if effective_before == effective_after:
                flags += ["--context", str(effective_before)]
            else:
                flags += ["--before-context", str(effective_before), "--after-context", str(effective_after)]
        flags += [f"--glob={_sh('!' + d + '/**')}" for d in SKIP_DIRS]
        cmd = " ".join(flags) + f" -- {_sh(pattern)} {_sh(path)}"
    else:
        # -E (ERE), not plain BRE: the docstring advertises ripgrep-style
        # patterns, and ripgrep's regex syntax treats `|`/`+`/`?`/`{}` as
        # metacharacters natively — GNU grep's BRE default does NOT (a
        # pattern like "Controller|controller" would search for that
        # literal string, pipe character included, and silently match
        # nothing real). Without ripgrep installed, every alternation
        # pattern the model naturally writes was returning a false
        # "no matches" instead of erroring, indistinguishable from an
        # honest empty result.
        flags = "grep -rnE --color=never"
        if case_insensitive:
            flags += " -i"
        if file_type:
            flags += f" --include={_sh('*.' + file_type)}"
        if glob:
            # GNU grep's --include does a plain fnmatch on the basename
            # only — it has no concept of ripgrep's recursive "**" glob
            # syntax, so a leading "**/" (the "any depth" prefix the
            # docstring's own example, '*.{{ts,tsx}}', and ripgrep's real
            # --glob both expect) matches no real filename here, silently
            # returning zero results. -r already recurses on its own, so
            # the trailing pattern alone (e.g. "*.php") is the correct
            # equivalent for this fallback.
            plain_glob = glob[3:] if glob.startswith("**/") else glob
            flags += f" --include={_sh(plain_glob)}"
        if output_mode == "content" and (effective_before or effective_after):
            flags += f" -B {effective_before} -A {effective_after}"
        skip = " ".join(f"--exclude-dir={d}" for d in SKIP_DIRS)
        cmd = f"{flags} {skip} -- {_sh(pattern)} {_sh(path)}"

    out, rc = await _run(cmd)
    if rc > 1:
        return f"Search error (rc={rc}). Pattern: {pattern!r}"
    raw_lines = [ln for ln in out.splitlines() if ln.strip()]
    if not raw_lines:
        return f"No matches for {pattern!r} in {path}"

    if output_mode == "count":
        files = sorted({ln.split(":", 1)[0] for ln in raw_lines if ":" in ln})
        return f"{len(files)} file(s), {len(raw_lines)} matching line(s) for {pattern!r}"

    if output_mode == "files_with_matches":
        files = sorted({ln.split(":", 1)[0] for ln in raw_lines if ":" in ln})
        files, truncated, applied_limit, applied_offset = apply_limit(files, head_limit, offset)
        result = "\n".join(files)
        if truncated:
            result += f"\n... (showing {applied_limit} of more results — narrow pattern/glob/path for the rest)"
        return result

    # output_mode == "content"
    lines, truncated, applied_limit, applied_offset = apply_limit(raw_lines, head_limit, offset)
    result = "\n".join(ln if len(ln) <= MAX_LINE_LEN else ln[:MAX_LINE_LEN] + "…" for ln in lines)
    if truncated:
        result += f"\n... (showing {applied_limit} lines — narrow pattern/glob/path, or page with offset= for the rest)"
    return result


def _expand_braces(pattern: str) -> list[str]:
    """Python's glob module doesn't support brace expansion ({a,b,c}) at
    all — a pattern like "*.{ts,tsx}" would otherwise match nothing.
    Handles ONE brace group (good enough for the common case); nested/
    multiple groups aren't expanded, the literal pattern is used as-is for
    those."""
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    options = m.group(1).split(",")
    return [pattern[:m.start()] + opt + pattern[m.end():] for opt in options]


def _is_skipped(rel_parts: tuple[str, ...]) -> bool:
    for part in rel_parts:
        for skip in SKIP_DIRS:
            if fnmatch.fnmatch(part, skip):
                return True
    return False


@mcp.tool()
async def glob_search(pattern: str, path: str = "") -> str:
    """Find files by NAME/path pattern (e.g. "*.ts", "src/**/*.py",
    "**/*.{ts,tsx}") — supports "**" for recursive matching and brace
    groups. This is ALSO how to browse a directory's contents (there's no
    separate listing tool) — e.g. glob_search("**/*", "src/some_module")
    lists everything under it. Results are sorted by most-recently-modified
    first (usually the most relevant to whatever's currently being worked
    on), capped at 100 with the rest omitted — narrow the pattern/path for
    more specific results instead of paging through all of them."""
    pattern = pattern.strip()
    if not pattern:
        return "Error: pattern is required"
    base = Path(path.strip() or ".").resolve()

    seen: set[Path] = set()
    for expanded in _expand_braces(pattern):
        try:
            for candidate in base.glob(expanded):
                if candidate.is_file():
                    rel = candidate.relative_to(base) if candidate.is_relative_to(base) else candidate
                    if not _is_skipped(rel.parts):
                        seen.add(candidate)
        except (ValueError, OSError):
            continue

    if not seen:
        return f"No files matching {pattern!r} under {base}"

    matches = sorted(seen, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    truncated = len(matches) > 100
    shown = matches[:100]
    result = "\n".join(str(p) for p in shown)
    if truncated:
        result += f"\n... (showing 100 of {len(matches)} matches, most-recently-modified first — narrow the pattern for the rest)"
    return result


_TRASH_DIR = data_dir() / "trash"


def _conn():
    conn = connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS trash ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, original_path TEXT NOT NULL, "
        "trash_path TEXT NOT NULL, ts TEXT NOT NULL)"
    )
    conn.commit()
    return conn


@mcp.tool()
async def delete_path(path: str) -> str:
    """Delete a file OR a directory (recursively, one call for the whole
    tree) — the ONLY delete tool; write_file/edit_file have no delete at
    all. SAFE BY DESIGN: instead of a permanent rm, this MOVES the target
    into a local trash folder — recoverable via restore_deleted_path if the
    deletion turns out to be wrong. Always prefer this over bash's rm/rm
    -rf, which is permanent and gives no way back."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    if not os.path.exists(path):
        return f"Error: {path!r} does not exist"

    _TRASH_DIR.mkdir(parents=True, exist_ok=True)
    trash_name = f"{uuid.uuid4().hex[:12]}__{os.path.basename(path.rstrip('/')) or 'root'}"
    trash_path = str(_TRASH_DIR / trash_name)

    try:
        shutil.move(path, trash_path)
    except OSError as e:
        return f"Error: {e}"

    conn = _conn()
    conn.execute(
        "INSERT INTO trash (original_path, trash_path, ts) VALUES (?, ?, ?)",
        (path, trash_path, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    trash_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return f"Deleted {path!r} (moved to trash, id={trash_id}) — call restore_deleted_path({trash_id}) to undo."


@mcp.tool()
async def list_deleted_paths(limit: int = 20) -> str:
    """List recently deleted (trashed) files/directories, most recent
    first — use this to find the right id before calling
    restore_deleted_path; never guess an id."""
    limit = max(1, min(limit, 100))
    conn = _conn()
    rows = conn.execute(
        "SELECT id, original_path, ts FROM trash ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    if not rows:
        return "Trash is empty — nothing has been deleted via delete_path."
    return "\n".join(f"id={r[0]}  deleted_at={r[2]}  was={r[1]}" for r in rows)


@mcp.tool()
async def restore_deleted_path(trash_id: int) -> str:
    """Restore a file/directory previously removed by delete_path, back to
    its original location. Call list_deleted_paths() first to find the
    right trash_id. Fails loudly (does not overwrite) if something already
    exists at the original path."""
    conn = _conn()
    row = conn.execute(
        "SELECT original_path, trash_path FROM trash WHERE id = ?", (trash_id,)
    ).fetchone()
    if row is None:
        return f"Error: no trash entry with id={trash_id} — call list_deleted_paths() to see valid ids."
    original_path, trash_path = row
    if os.path.exists(original_path):
        return f"Error: {original_path!r} already exists — move/remove it first if you really want to restore here."
    if not os.path.exists(trash_path):
        return f"Error: trash contents for id={trash_id} are missing on disk (already restored or purged)."
    try:
        shutil.move(trash_path, original_path)
    except OSError as e:
        return f"Error: {e}"
    conn.execute("DELETE FROM trash WHERE id = ?", (trash_id,))
    conn.commit()
    return f"Restored {original_path!r} from trash."


if __name__ == "__main__":
    mcp.run()
