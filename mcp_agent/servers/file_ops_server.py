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
  - write_file возвращает настоящий diff через utils/parsing.py:
    unified_diff_at (difflib.SequenceMatcher) — уже написанный в этом
    проекте генератор построчного diff'а, а не подделку вроде "весь старый
    файл как удалённые строки, весь новый — как добавленные".
  - read_file/write_file/edit_file держат бинарный/размерный гард ПРЯМО
    внутри тула (NUL-байт в первых 8KB, 10MB потолок) — раньше это было
    внешней обёрткой в tool_wrappers.py, потому что чтение шло через
    сторонний npm-пакет; теперь тул свой, гард можно держать в естественном
    месте, а не снаружи.
  - grep_search/glob_search переиспользуют SKIP_DIRS/_HAS_RG/_run/_sh из
    старого code_search_server.py как есть (тот же живой опыт с venv-tts/
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

from storage import connect, data_dir  # noqa: E402
from utils.parsing import unified_diff_at  # noqa: E402

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
    tail on a huge file instead of trying to read it whole."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return f"Error: {e}"
    if size > _MAX_READABLE_FILE_BYTES:
        return (
            f"Error: {path!r} is {size / (1024 * 1024):.1f}MB — over the "
            f"{_MAX_READABLE_FILE_BYTES // (1024 * 1024)}MB limit for a single "
            "read. Narrow with offset/limit, or use bash with grep/head/tail."
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
async def write_file(path: str, content: str) -> str:
    """Write (create or fully overwrite) a text file — creates any missing
    parent directories automatically. Returns a real diff of what changed
    against the previous content (empty if the file is new). Rejects
    content over 10MB — for a file that large, this is very likely the
    wrong tool for the job."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    if len(content.encode("utf-8", errors="replace")) > _MAX_WRITABLE_CONTENT_BYTES:
        return (
            f"Error: content is {len(content) / (1024 * 1024):.1f}MB — over "
            f"the {_MAX_WRITABLE_CONTENT_BYTES // (1024 * 1024)}MB limit for "
            "a single write."
        )

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
        is_new = False
    except OSError:
        original = None
        is_new = True

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: {e}"

    if is_new:
        return f"Created {path!r} ({len(content.splitlines())} lines)."
    diff = unified_diff_at((original or "").splitlines(keepends=True), content.splitlines(keepends=True), path, 1)
    return f"Updated {path!r}.\n\n{diff}" if diff else f"Wrote {path!r} (content unchanged)."


@mcp.tool()
async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact, VERBATIM substring in a file — old_string must
    match the file's CURRENT content byte-for-byte (read the file first,
    don't reconstruct old_string from memory). By default replaces only the
    FIRST occurrence — old_string must therefore include enough surrounding
    context (a few lines) to be unique in the file, or the call fails
    asking you to narrow it; pass replace_all=true to replace every
    occurrence instead (e.g. renaming a variable used many times). Returns
    a real diff of what changed."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    if old_string == new_string:
        return "Error: old_string and new_string must differ"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()
    except OSError as e:
        return f"Error: {e}"

    count = original.count(old_string)
    if count == 0:
        return f"Error: old_string not found in {path!r} — re-read the file, it may not match byte-for-byte."
    if not replace_all and count > 1:
        return (
            f"Error: old_string appears {count} times in {path!r} — it must be "
            "unique for a single replacement. Include more surrounding context "
            "in old_string to disambiguate, or pass replace_all=true if you "
            "really want every occurrence replaced."
        )

    updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
    except OSError as e:
        return f"Error: {e}"

    diff = unified_diff_at(original.splitlines(keepends=True), updated.splitlines(keepends=True), path, 1)
    return f"Edited {path!r} ({count if replace_all else 1} replacement{'s' if replace_all and count != 1 else ''}).\n\n{diff}"


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
        flags = "grep -rn --color=never"
        if case_insensitive:
            flags += " -i"
        if file_type:
            flags += f" --include={_sh('*.' + file_type)}"
        if glob:
            flags += f" --include={_sh(glob)}"
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
