"""
Кастомный MCP-сервер: поиск по содержимому файлов (grep/rg), поиск файлов по
имени, поиск определений символов.

filesystem MCP-сервер (@modelcontextprotocol/server-filesystem) даёт
search_files только по ИМЕНИ файла — искать "TODO/FIXME по коду" им нельзя.
Готового content-search сервера с таким же поведением (rg/grep, exclude
node_modules/.venv/.git и т.п.) в реестре нет — переиспользуем логику
tools/code_search.py как есть, просто через MCP-транспорт.

Запуск: python3 -m mcp_agent.servers.code_search_server
"""
import asyncio
import os
import re
import shutil
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from utils.parsing import line_hash  # noqa: E402

mcp = FastMCP("code_search")

_HAS_RG = shutil.which("rg") is not None
MAX_RESULTS = 60
MAX_LINE_LEN = 300
SKIP_DIRS = ["node_modules", ".git", "vendor", "__pycache__", ".cache", "dist", "build", ".venv", "venv", ".tox"]
DEFAULT_TIMEOUT = 15
# Потолок для the timeout= argument exposed on search_code/find_files_by_name/
# search_symbols below — same idea as bash_exec_server.py's MAX_TIMEOUT: a
# call this long still blocks the whole synchronous tool call/turn, so it's
# meant for "I already know this path is a large/external tree", not a
# substitute for narrowing path/file_pattern.
MAX_TIMEOUT = 120


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


# rg/grep with --with-filename prints the FULL path on every single line,
# match or context: "path:42:match", "path-43-context", "path-44-context",
# ... — asked a user about this exact output once context_lines pulled in
# a dozen lines of one file per match: the same ~70-char path repeated on
# every one of them, pure waste once it's already established which file
# a block belongs to. Matches rg's own "-" (context) vs ":" (match)
# separator convention, so a match line keeps ":" and a context line keeps
# "-" even after the path is dropped.
_GREP_LINE_RE = re.compile(r"^(.+?)([:-])(\d+)([:-])(.*)$")


def _collapse_repeated_paths(lines: list[str]) -> list[str]:
    """Collapses consecutive lines from the SAME file down to one path
    header followed by bare "line:content"/"line-context" lines — same
    idea as `rg --heading`, applied after the fact since we already build
    the plain rg/grep command elsewhere. Resets the header on a `--`
    block-separator line (rg/grep's own marker for a gap between non-
    adjacent matches — a new block deserves its own path) and on any line
    that doesn't parse as path+line+content (left untouched, unchanged)."""
    out: list[str] = []
    last_path: str | None = None
    for line in lines:
        if line == "--":
            out.append(line)
            last_path = None
            continue
        m = _GREP_LINE_RE.match(line)
        if not m:
            out.append(line)
            last_path = None
            continue
        path, _sep1, lineno, sep2, rest = m.groups()
        if path == last_path:
            out.append(f"{lineno}{sep2}{rest}")
        else:
            out.append(line)
            last_path = path
    return out


# Live bug: search_code(query="fmt\\.", case_sensitive=True) (regex left
# at its False default) returned "No matches" against a file that
# literally contains "fmt.Println(..." — correct behavior for a FIXED
# STRING search (the file has no literal backslash before the dot), but
# the model clearly intended '\.' as a regex-escaped dot and never
# realized its "no matches" was actually "you forgot regex=true", not
# "this text doesn't exist". A real backslash character showing up in a
# code-search query is rare outside Windows paths; '.*' is a regex idiom
# that's essentially never a literal target either. Neither check fires
# on an ordinary literal query like "config.go" or "*.env" (no backslash,
# no '.*').
_REGEX_HINT_MARKERS = ("\\", ".*")


def _looks_like_intended_regex(query: str) -> bool:
    return any(marker in query for marker in _REGEX_HINT_MARKERS)


def _trim(lines: list[str], limit: int = MAX_RESULTS) -> str:
    if len(lines) > limit:
        shown = lines[:limit]
        shown.append(f"... (showing {limit} of {len(lines)} results, narrow the search)")
        lines = shown
    return "\n".join(ln if len(ln) <= MAX_LINE_LEN else ln[:MAX_LINE_LEN] + "…" for ln in lines)


@mcp.tool()
async def search_code(
    query: str,
    path: str = ".",
    regex: bool = False,
    case_sensitive: bool = False,
    file_pattern: str = "",
    context_lines: int = 0,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Search for a text/regex pattern INSIDE file contents (e.g. TODO,
    FIXME, a function call, an error string). Use this instead of guessing
    file paths — file_pattern e.g. "*.py" restricts by extension.
    context_lines (0-20) shows N lines of surrounding code around each match
    — use it INSTEAD OF a follow-up read_file call when a match's own line
    isn't enough to tell what's going on; it's free context on the same
    call, not a second round trip. Default timeout is 15s — if `path` is a
    large/external tree (a big repo outside this session's own working
    directory, or an unfamiliar monorepo/subsystem) and you already expect
    this to legitimately take longer, pass a bigger timeout= (up to 120s)
    instead of retrying the same call against the default; for anything
    still too broad even at 120s, narrow path/file_pattern instead."""
    query = query.strip()
    if not query:
        return "Error: query is required"
    # Модель иногда явно передаёт path="" вместо того, чтобы просто не
    # указывать аргумент — пустая строка идёт в rg/grep буквально и даёт
    # ошибку (rc=2), а не "текущая директория", как можно было ожидать.
    path = path.strip() or "."
    context_lines = max(0, min(context_lines, 20))
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))

    if _HAS_RG:
        flags = ["rg", "--line-number", "--with-filename", "--max-count=5", "--color=never"]
        if not case_sensitive:
            flags.append("--ignore-case")
        if not regex:
            flags.append("--fixed-strings")
        if file_pattern:
            flags += ["--glob", file_pattern]
        if context_lines:
            flags += ["--context", str(context_lines)]
        cmd = " ".join(flags) + f" -- {_sh(query)} {_sh(path)}"
    else:
        flags = "grep -rn --color=never"
        if not case_sensitive:
            flags += " -i"
        if not regex:
            flags += " -F"
        if file_pattern:
            flags += f" --include={_sh(file_pattern)}"
        if context_lines:
            flags += f" -C {context_lines}"
        skip = " ".join(f"--exclude-dir={d}" for d in SKIP_DIRS)
        cmd = f"{flags} {skip} -- {_sh(query)} {_sh(path)}"

    out, rc = await _run(cmd, timeout=effective_timeout)
    if rc > 1:
        return f"Search error (rc={rc}). Query: {query!r}"
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        no_match = f"No matches for {query!r} in {path}"
        if not regex and _looks_like_intended_regex(query):
            # Live bug: the tool description already warns that regex
            # defaults to False (see _REGEX_WARNING, mcp_agent/
            # tool_wrappers.py) — a purely static, before-the-call warning
            # that this exact query still walked straight past. A query
            # with '\.'/'\d' etc. or '.*' is almost never meant to be
            # searched as a LITERAL string (a real backslash character in
            # source code is rare); appended ONLY when a plain match
            # actually came back empty, right where it's most likely to be
            # read and acted on, not buried in the tool description before
            # the mistake even happens.
            no_match += (
                " — searched as a LITERAL string (regex=False, the "
                "default): any backslashes/dots/asterisks in the query "
                "were matched as plain characters, not regex syntax. This "
                f"query ({query!r}) looks like it was meant as a regex "
                "pattern — if so, retry with regex=true; if you really "
                "did mean a literal string, this is a genuine miss, not a "
                "tool bug."
            )
        return no_match
    return _trim(_collapse_repeated_paths(lines))


MAX_RANGE_LINES = 400


@mcp.tool()
async def read_file_range(path: str, start_line: int, end_line: int) -> str:
    """Read an EXACT line range [start_line, end_line] (1-indexed, inclusive)
    from a file. Use this INSTEAD OF read_file's head/tail whenever you
    already know roughly where the relevant code is — a line number from
    search_code/search_symbols, a stack trace, or a diff hunk header — so you
    don't have to guess a head/tail size and re-read overlapping chunks of
    the same file to home in on it.

    Each line is shown as "N [HHHH] content" — N is the line number, HHHH
    (inside the brackets, 4 hex chars, nothing else) is a short content
    hash. Copy ONLY what's inside the brackets into replace_lines/
    insert_lines/copy_lines' expected_first_hash/expected_last_hash/
    expected_hash — those tools check it instead of the full line text, so
    a stale edit (file changed, or an earlier edit this turn shifted every
    line number after it) fails loudly with a hash mismatch instead of
    silently landing on the wrong line. Do NOT include "N ", the brackets
    themselves, or the line content — just the 4 characters between them."""
    if start_line < 1 or end_line < start_line:
        return "Error: start_line must be >= 1 and end_line >= start_line"
    if end_line - start_line + 1 > MAX_RANGE_LINES:
        return f"Error: range too large (max {MAX_RANGE_LINES} lines) — narrow start_line/end_line"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: {e}"

    total = len(lines)
    if start_line > total:
        return f"Error: file has only {total} lines, start_line={start_line} is past the end"

    selected = lines[start_line - 1:end_line]
    numbered = [
        f"{start_line + i} [{line_hash(line)}] {line.rstrip(chr(10))}"
        for i, line in enumerate(selected)
    ]
    result = "\n".join(numbered)
    if end_line > total:
        result += f"\n\n[end_line {end_line} is past the end — file has {total} lines, showing through EOF]"
    return result


@mcp.tool()
async def find_files_by_name(pattern: str, path: str = ".", max_results: int = 80, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Find files by NAME pattern (e.g. '*.ts', '*controller*') — not
    content. Supports REAL glob wildcards ('*', '?', '[abc]'), unlike the
    filesystem server's own search_files, which only does a plain
    substring match on the name and silently returns nothing for a pattern
    containing '*'/'?'. Use THIS tool whenever the pattern needs a
    wildcard. Default timeout is 15s — for a large/external tree you
    already expect to take longer, pass a bigger timeout= (up to 120s)."""
    path = path.strip() or "."
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    cmd = (
        f"find {_sh(path)} -type f -name {_sh(pattern)} "
        + " ".join(f"-not -path '*/{d}/*'" for d in SKIP_DIRS)
        + f" | head -{max_results}"
    )
    out, _ = await _run(cmd, timeout=effective_timeout)
    lines = [l for l in out.splitlines() if l.strip()]
    if not lines:
        return f"No files matching '{pattern}' in {path}"
    return "\n".join(sorted(lines))


FILE_LIST_THRESHOLD = 300


@mcp.tool()
async def project_tree(path: str = ".", max_entries: int = 500, timeout: int = DEFAULT_TIMEOUT) -> str:
    """List directories recursively under path — a flat, sorted list of
    relative paths — excluding vendor/dependency directories (node_modules,
    vendor, .venv, venv, .git, __pycache__, dist, build, .cache, .tox) that
    would otherwise bury the project's own structure in noise. Call this
    ONCE at the start of an investigation instead of list_directory on each
    subdirectory one at a time — live bug: 13 sequential list_directory
    calls walked one module's subdirectories individually before a single
    file got read, burning most of a round's step budget on structure
    alone. The official directory_tree tool has no exclusion list and would
    dump vendor/node_modules wholesale on a real project — use this instead
    whenever you need the PROJECT's own layout, not directory_tree.

    If the project has 300 files or fewer (after the same exclusions),
    files are included in the same flat list alongside directories, so a
    small project can often be fully mapped in this one call. Bigger
    projects get directories only, plus a note with the omitted file count
    — narrow path and call again, or use find_files_by_name/search_code to
    locate specific files. Default timeout is 15s per `find` call (two per
    invocation) — for a large/external tree you already expect to take
    longer, pass a bigger timeout= (up to 120s)."""
    path = path.strip() or "."
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))
    prune = " -o ".join(f"-name {_sh(d)}" for d in SKIP_DIRS)
    exclude_path = " ".join(f"-not -path '*/{d}/*'" for d in SKIP_DIRS)

    dir_cmd = f"find {_sh(path)} -type d \\( {prune} \\) -prune -o -type d -print"
    dir_out, rc = await _run(dir_cmd, timeout=effective_timeout)
    if rc > 1:
        return f"Error listing directories (rc={rc})"
    dirs = [l for l in dir_out.splitlines() if l.strip()]
    if not dirs:
        return f"No directories found under {path}"

    file_cmd = f"find {_sh(path)} -type f {exclude_path}"
    file_out, _ = await _run(file_cmd, timeout=effective_timeout)
    files = [l for l in file_out.splitlines() if l.strip()]

    note = ""
    if len(files) <= FILE_LIST_THRESHOLD:
        entries = sorted(dirs + files)
    else:
        entries = sorted(dirs)
        note = (
            f"\n... ({len(files)} files omitted, above the "
            f"{FILE_LIST_THRESHOLD}-file threshold for listing them all — "
            "narrow path or use find_files_by_name/search_code)"
        )

    truncated = len(entries) > max_entries
    if truncated:
        entries = entries[:max_entries]
    result = "\n".join(entries)
    if truncated:
        result += f"\n... (showing {max_entries} entries, narrow path for the rest)"
    result += note
    return result


@mcp.tool()
async def search_symbols(query: str, path: str = ".", timeout: int = DEFAULT_TIMEOUT) -> str:
    """Find function/class/interface DEFINITIONS by name across common
    languages (Python, JS/TS, Go, Java, PHP, Kotlin, Ruby). Runs several
    patterns in sequence, each with its own timeout (default 15s) — for a
    large/external tree you already expect to take longer, pass a bigger
    timeout= (up to 120s), applied per pattern, not to the whole call."""
    query = query.strip()
    if not query:
        return "Error: query is required"
    path = path.strip() or "."
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))

    patterns = [
        f"def {query}", f"class {query}", f"function {query}", f"func {query}",
        f"func .*{query}", f"interface {query}", f"type {query} ", f"const {query} =",
        f"let {query} =", f"var {query} ", f"public.*function {query}",
        f"private.*function {query}", f"protected.*function {query}",
        f"public.*{query}\\(", f"fun {query}\\(", f"sub {query}", f"def self\\.{query}",
    ]

    results: list[str] = []
    seen: set[str] = set()

    for pat in patterns:
        if _HAS_RG:
            cmd = f"rg -n --with-filename --color=never -e {_sh(pat)} {_sh(path)}"
        else:
            skip = " ".join(f"--exclude-dir={d}" for d in SKIP_DIRS)
            cmd = f"grep -rn -E --color=never {skip} -- {_sh(pat)} {_sh(path)}"

        out, _ = await _run(cmd, timeout=effective_timeout)
        for line in out.splitlines():
            if line.strip() and line not in seen:
                seen.add(line)
                results.append(line)
        if len(results) >= MAX_RESULTS:
            break

    if not results:
        return f"No symbol definitions matching '{query}' found in {path}"
    return _trim(_collapse_repeated_paths(results))


if __name__ == "__main__":
    mcp.run()
