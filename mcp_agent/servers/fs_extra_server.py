"""
Кастомный MCP-сервер: недостающие filesystem-операции.

@modelcontextprotocol/server-filesystem умеет read/write/create_directory/
move_file, но НЕ умеет удалять — ни файл, ни папку, никак. Единственный
способ удалить что-то сейчас — bash_exec с rm/rm -rf, а это необратимо и
ничего не оставляет на восстановление, если модель удалила не то.
delete_path закрывает этот пробел тем же принципом обратимости, что уже
есть у git_restore_file/restore_file_snapshot: вместо permanent rm —
перенос в локальную "корзину", откуда можно вернуть обратно.

replace_lines закрывает другой пробел: edit_file требует byte-for-byte
oldText — модели приходится дважды генерировать один и тот же кусок кода
(старый текст для сопоставления + новый текст), это дорого по токенам и
именно оттуда взялись все живые баги с невалидным JSON/галлюцинациями в
oldText (см. _normalize_edit_file_args в mcp_agent/agent.py). Модель почти
всегда уже знает номера строк (search_code/read_file_range их возвращают)
— значит сопоставление по тексту вообще не нужно, достаточно диапазона.

copy_lines закрывает третий пробел: до него единственный способ перенести
код между файлами был read целиком + write целиком — модель пересочиняла
уже существующий текст по памяти вместо копирования, что на локальной
модели медленно (см. живой прогон: 13 минут на файл 356 строк) и рискует
незаметно исказить логику при "пересказе". copy_lines копирует диапазон
строк verbatim в другой файл, replace_lines(new_content='') следом чистит
источник — итого честный move без единого лишнего токена генерации.

(Живой прогон также выявил, что модель звала search_files (filesystem MCP
server, плоский substring-матч, '*'/'?' не поддерживаются) с
glob-паттерном "*agent*.py" и получила пустой результат — хотя в проекте
уже есть свой find_files_by_name с настоящим glob (code_search_server.py),
который эту же задачу решает правильно. Фикс там — в agent.py, description
search_files дописывается предупреждением, а не новый тул здесь.)

Запуск: python3 -m mcp_agent.servers.fs_extra_server
"""
import os
import re
import shutil
import sys
import uuid
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

from storage import connect, data_dir  # noqa: E402
from utils.parsing import line_hash, unified_diff_at  # noqa: E402

mcp = FastMCP("fs_extra")

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
    tree) — the ONLY delete tool; the filesystem tools (read_file,
    write_file, create_directory, move_file, ...) have no delete at all.
    SAFE BY DESIGN: instead of a permanent rm, this MOVES the target into a
    local trash folder — recoverable via restore_deleted_path if the
    deletion turns out to be wrong. Always prefer this over bash_exec with
    rm/rm -rf, which is permanent and gives no way back."""
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


_HASH_RE = re.compile(r"^[0-9a-f]{4}$")


def _hash_guard(*named_values: tuple[str, str]) -> str:
    """Catches an obviously-wrong expected_*_hash value early — anything
    that isn't a bare 4-hex-char token (exactly what's inside the brackets
    in read_file_range's "N [HHHH] ..." output) is either a full line of
    text left over from before this tool switched to hash-based
    verification, or something typed/guessed instead of copied. Failing
    here with a specific reason beats the generic hash-mismatch error
    below, which would otherwise send the model chasing the wrong cause
    (recounting line numbers) instead of the real one (wrong argument
    shape) — the same live-bug pattern that used to hit the old
    expected_*_line text field.

    Live run (20260812, XOR-in-Go task): even with the bracket format, the
    model twice passed "155:d8e9|" / "170:c8a0|" — the whole old
    colon/pipe-glued token verbatim, not just the hash — so the format
    itself was changed (brackets, not colon+pipe) to make the boundary of
    "just the hash" visually unambiguous, on top of this guard."""
    for name, value in named_values:
        if value and not _HASH_RE.match(value):
            return (
                f"Error: {name} must be the exact 4-character hash "
                f"read_file_range showed in brackets next to that line "
                f"(e.g. \"a3f2\" from \"N [a3f2] ...\"), not the line's text, "
                f"its line number, or the brackets themselves — copy ONLY "
                f"what's between the brackets, from that tool's most recent "
                f"output for this exact line."
            )
    return ""


def _suggest_range(lines: list[str], expected_first_hash: str, expected_last_hash: str) -> str:
    """Best-effort: if a line matching the (first, last) hash pair the model
    expects still exists elsewhere in the CURRENT file, say where. Turns a
    bare mismatch into an actionable correction instead of forcing another
    blind read+recount round-trip — the file already contains the answer,
    no need to make the model guess again."""
    starts = [i for i, l in enumerate(lines) if line_hash(l) == expected_first_hash]
    if not starts:
        return ""
    if expected_first_hash == expected_last_hash:
        return f" Found a line with that hash at line {starts[0] + 1} instead — did you mean start_line={starts[0] + 1}?"
    for i in starts:
        for j in range(i, min(i + 500, len(lines))):
            if line_hash(lines[j]) == expected_last_hash:
                return f" Found a matching line pair at lines {i + 1}-{j + 1} instead — did you mean start_line={i + 1}, end_line={j + 1}?"
    return ""


def _duplicate_block_warning(lines: list[str], new_lines: list[str], insert_at: int, window: int = 80, min_run: int = 3) -> str:
    """Live incident: insert_lines was called with the CORRECT target line
    (right before an existing `const TokenTtl = 15`), so expected_hash
    matched and the call succeeded — but new_content had been "reconstructed"
    from memory as a whole desired end-state (import block + two new
    functions + that same const) instead of ONLY the two new functions,
    duplicating an 8-line import block that already existed a few lines
    above. expected_hash only checks the ONE anchor line, so it cannot catch
    this class of mistake at all. Scans a small window of the file around the
    insertion point for a run of >= min_run consecutive lines that's already
    identical inside new_content — a near-certain sign of exactly that
    mistake — and rejects the write instead of silently corrupting the file."""
    lo = max(0, insert_at - window)
    hi = min(len(lines), insert_at + window)
    nearby = [l.rstrip("\n") for l in lines[lo:hi]]
    new_stripped = [l.rstrip("\n") for l in new_lines]
    n = len(new_stripped)
    for run in range(n, min_run - 1, -1):
        for start in range(0, n - run + 1):
            chunk = new_stripped[start:start + run]
            if sum(len(c.strip()) for c in chunk) < 15:
                continue  # too short/trivial (blank lines, lone braces) to be a meaningful signal
            for j in range(0, len(nearby) - run + 1):
                if nearby[j:j + run] == chunk:
                    file_line = lo + j + 1
                    return (
                        f"Error: new_content line(s) {start + 1}-{start + run} are IDENTICAL to "
                        f"{run} line(s) already sitting at {file_line}-{file_line + run - 1} in "
                        f"the file, right near your insertion point. insert_lines only ADDS "
                        f"lines — it never touches what's already there, so if new_content "
                        f"repeats existing code, the result is a duplicate, not a replacement. "
                        f"This usually means new_content was retyped from memory as 'what the "
                        f"file should look like after my change' instead of 'only the new lines "
                        f"to splice in'. Remove the duplicated part from new_content (keep only "
                        f"what's genuinely new) and retry — or use replace_lines if you actually "
                        f"meant to replace that existing block."
                    )
    return ""


def _suggest_line(lines: list[str], expected_hash: str) -> str:
    matches = [i for i, l in enumerate(lines) if line_hash(l) == expected_hash]
    if not matches:
        return ""
    return f" Found a line with that hash at line {matches[0] + 1} instead — did you mean line={matches[0] + 1}?"


@mcp.tool()
async def replace_lines(
    path: str,
    start_line: int,
    end_line: int,
    new_content: str,
    expected_first_hash: str = "",
    expected_last_hash: str = "",
) -> str:
    """Replace an EXACT [start_line, end_line] range (1-indexed, inclusive)
    with new_content — the PREFERRED way to edit an existing file whenever
    you already know the line range (from read_file_range/search_code,
    which both report line numbers in their output). Unlike edit_file, you
    NEVER need to reproduce the file's existing content — only the NEW
    lines — cutting the code you generate roughly in half and eliminating
    the byte-for-byte oldText matching that regularly fails on quoting/
    escaping. Pass expected_first_hash/expected_last_hash — ONLY the 4-char
    hash read_file_range showed in brackets next to those two lines
    ("N [HHHH] ..."), not the brackets/line number/line text — to catch a
    stale line range: the
    file changed since you read it, or an earlier edit in this same turn
    shifted line numbers; the call then fails loudly with the file's ACTUAL
    current content at that range instead of silently overwriting the wrong
    lines. Leave them empty only when you just read this exact range in
    this same turn. Pass new_content='' to delete the range outright.
    Returns a diff of what actually changed — check it matches your intent
    before reporting success."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    if start_line < 1 or end_line < start_line:
        return "Error: start_line must be >= 1 and end_line >= start_line"
    guard = _hash_guard(("expected_first_hash", expected_first_hash), ("expected_last_hash", expected_last_hash))
    if guard:
        return guard

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: {e}"

    total = len(lines)
    if end_line > total:
        return f"Error: file has only {total} lines, end_line={end_line} is past the end"

    current_first = lines[start_line - 1].rstrip("\n")
    current_last = lines[end_line - 1].rstrip("\n")

    if expected_first_hash and line_hash(current_first) != expected_first_hash:
        return (
            f"Error: line {start_line} doesn't match expected_first_hash — the file "
            f"changed since you last read it, or an earlier edit shifted line numbers."
            f"{_suggest_range(lines, expected_first_hash, expected_last_hash)}\n"
            f"Expected hash: {expected_first_hash!r}\n"
            f"Actual line {start_line}: {current_first!r} (hash {line_hash(current_first)!r})\n"
            f"Re-read the current range (read_file_range) before retrying."
        )
    if expected_last_hash and line_hash(current_last) != expected_last_hash:
        return (
            f"Error: line {end_line} doesn't match expected_last_hash — the file "
            f"changed since you last read it, or an earlier edit shifted line numbers."
            f"{_suggest_range(lines, expected_first_hash, expected_last_hash)}\n"
            f"Expected hash: {expected_last_hash!r}\n"
            f"Actual line {end_line}: {current_last!r} (hash {line_hash(current_last)!r})\n"
            f"Re-read the current range (read_file_range) before retrying."
        )

    old_block = lines[start_line - 1:end_line]
    new_block_lines = new_content.splitlines()
    new_lines = [line + "\n" for line in new_block_lines]
    # Сохраняем "нет финального переноса строки" у EOF, если заменяемый
    # блок был концом файла и сам не заканчивался переносом.
    if new_lines and end_line == total and not lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1].rstrip("\n")

    updated = lines[:start_line - 1] + new_lines + lines[end_line:]

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(updated)
    except OSError as e:
        return f"Error: {e}"

    diff = unified_diff_at(old_block, new_lines, path, start_line)
    return f"Replaced lines {start_line}-{end_line} in {path!r}.\n\n{diff}"


@mcp.tool()
async def insert_lines(
    path: str,
    line: int,
    new_content: str,
    expected_hash: str = "",
) -> str:
    """Insert new_content as new lines BEFORE `line` (1-indexed) in path,
    WITHOUT touching or replacing any existing line — the tool for adding
    code (a new import, a new method, a new block) next to existing code
    where nothing should be removed. Pass line=0 (or total_lines+1) to
    append at the end of the file instead of inserting before a specific
    line. Unlike replace_lines, you never need to know or reproduce the
    content of the line you're inserting next to — only WHERE; that's the
    difference this tool exists for — faking an insertion via
    replace_lines(start_line=end_line=N, new_content=<line N verbatim> +
    '\\n' + <new code>) requires reproducing line N byte-for-byte, and
    getting that wrong either duplicates or silently drops it. Pass
    expected_hash — ONLY the 4-char hash read_file_range showed in brackets
    next to the line you intend to insert before ("N [HHHH] ..."), not the
    brackets/line number/line text — to catch a stale line number: an
    earlier edit in this same turn shifted every line number after it, and
    a mismatch here fails loudly with the file's actual content instead of
    inserting at the wrong
    spot. Leave it empty only for line=0 (append) or when you just read
    this exact line in this same turn. Returns a diff of what was inserted
    — check it lands where you intended."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    if line < 0:
        return "Error: line must be >= 0 (0 = append at end of file)"
    guard = _hash_guard(("expected_hash", expected_hash))
    if guard:
        return guard

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: {e}"

    total = len(lines)
    if line == 0:
        insert_at = total
    else:
        if line > total + 1:
            return f"Error: {path!r} has {total} lines, line must be between 1 and {total + 1} (or 0 to append)"
        insert_at = line - 1
        if insert_at < total:
            current = lines[insert_at].rstrip("\n")
            if expected_hash and line_hash(current) != expected_hash:
                return (
                    f"Error: line {line} doesn't match expected_hash — the file "
                    f"changed since you last read it, or an earlier edit in this "
                    f"same turn shifted line numbers."
                    f"{_suggest_line(lines, expected_hash)}\n"
                    f"Expected hash: {expected_hash!r}\n"
                    f"Actual line {line}: {current!r} (hash {line_hash(current)!r})\n"
                    f"Re-read the current range (read_file_range) before retrying."
                )

    new_block_lines = new_content.splitlines()
    new_lines = [l + "\n" for l in new_block_lines]

    dup_error = _duplicate_block_warning(lines, new_lines, insert_at)
    if dup_error:
        return dup_error

    if insert_at == total and lines and not lines[-1].endswith("\n"):
        # Файл заканчивался без переноса строки — сначала закрыть последнюю
        # существующую строку, иначе она слипнется с первой новой.
        lines[-1] = lines[-1] + "\n"

    updated = lines[:insert_at] + new_lines + lines[insert_at:]

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(updated)
    except OSError as e:
        return f"Error: {e}"

    where = "the end of" if line == 0 else f"before line {line} in"
    diff = unified_diff_at([], new_lines, path, insert_at + 1)
    return f"Inserted {len(new_lines)} line(s) {where} {path!r}.\n\n{diff}"


@mcp.tool()
async def copy_lines(
    source_path: str,
    start_line: int,
    end_line: int,
    dest_path: str,
    dest_line: int = 0,
    create_dest: bool = False,
    expected_first_hash: str = "",
    expected_last_hash: str = "",
) -> str:
    """Copy an EXACT [start_line, end_line] range (1-indexed, inclusive)
    from source_path into dest_path VERBATIM, byte-for-byte — no need to
    retype/regenerate the content on either end. THE way to move logic
    between files (e.g. splitting a function out of a big file into a new
    module): read_file_range/search_code give you the range, this copies it
    exactly instead of you reading it and writing it back out from memory
    (slow on a local model, and risks silently changing the code while
    "porting" it). source_path is left UNCHANGED — for a MOVE (not a copy),
    follow up with replace_lines(source_path, start_line, end_line,
    new_content='') to delete the range from the source once the copy is
    confirmed in place. dest_line=0 (default) appends to the end of
    dest_path; any other value inserts BEFORE that line (1-indexed) in
    dest_path. Set create_dest=True to create dest_path if it doesn't exist
    yet (e.g. the first extraction into a brand-new module); without it, a
    missing dest_path is an error, to catch a typo'd path before it silently
    creates garbage. Pass expected_first_hash/expected_last_hash (as in
    replace_lines — the short hash read_file_range showed, not the line
    text) to catch a stale source range instead of silently copying the
    wrong lines."""
    source_path = source_path.strip()
    dest_path = dest_path.strip()
    if not source_path or not dest_path:
        return "Error: source_path and dest_path are required"
    if start_line < 1 or end_line < start_line:
        return "Error: start_line must be >= 1 and end_line >= start_line"
    guard = _hash_guard(("expected_first_hash", expected_first_hash), ("expected_last_hash", expected_last_hash))
    if guard:
        return guard

    try:
        with open(source_path, "r", encoding="utf-8") as f:
            source_lines = f.readlines()
    except OSError as e:
        return f"Error reading source_path: {e}"

    total = len(source_lines)
    if end_line > total:
        return f"Error: {source_path!r} has only {total} lines, end_line={end_line} is past the end"

    current_first = source_lines[start_line - 1].rstrip("\n")
    current_last = source_lines[end_line - 1].rstrip("\n")
    if expected_first_hash and line_hash(current_first) != expected_first_hash:
        return (
            f"Error: line {start_line} of {source_path!r} doesn't match expected_first_hash "
            f"— the file changed since you last read it, or an earlier edit shifted line "
            f"numbers.{_suggest_range(source_lines, expected_first_hash, expected_last_hash)}\n"
            f"Expected hash: {expected_first_hash!r}\nActual: {current_first!r} (hash {line_hash(current_first)!r})\n"
            "Re-read the current range (read_file_range) before retrying."
        )
    if expected_last_hash and line_hash(current_last) != expected_last_hash:
        return (
            f"Error: line {end_line} of {source_path!r} doesn't match expected_last_hash "
            f"— the file changed since you last read it, or an earlier edit shifted line "
            f"numbers.{_suggest_range(source_lines, expected_first_hash, expected_last_hash)}\n"
            f"Expected hash: {expected_last_hash!r}\nActual: {current_last!r} (hash {line_hash(current_last)!r})\n"
            "Re-read the current range (read_file_range) before retrying."
        )

    block = source_lines[start_line - 1:end_line]
    if block and not block[-1].endswith("\n"):
        block[-1] = block[-1] + "\n"

    if os.path.exists(dest_path):
        try:
            with open(dest_path, "r", encoding="utf-8") as f:
                dest_lines = f.readlines()
        except OSError as e:
            return f"Error reading dest_path: {e}"
    elif create_dest:
        dest_lines = []
    else:
        return f"Error: {dest_path!r} does not exist — pass create_dest=True to create it"

    dest_total = len(dest_lines)
    if dest_line == 0:
        insert_at = dest_total
        if dest_lines and not dest_lines[-1].endswith("\n"):
            dest_lines[-1] = dest_lines[-1] + "\n"
    else:
        if dest_line < 1 or dest_line > dest_total + 1:
            return f"Error: {dest_path!r} has {dest_total} lines, dest_line must be between 1 and {dest_total + 1}"
        insert_at = dest_line - 1

    updated = dest_lines[:insert_at] + block + dest_lines[insert_at:]

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            f.writelines(updated)
    except OSError as e:
        return f"Error writing dest_path: {e}"

    where = "the end of" if dest_line == 0 else f"before line {dest_line} in"
    return (
        f"Copied lines {start_line}-{end_line} ({len(block)} lines) from {source_path!r} "
        f"to {where} {dest_path!r}. source_path is UNCHANGED — if this was a MOVE, now call "
        f"replace_lines({source_path!r}, {start_line}, {end_line}, new_content='') to delete "
        "it from the source."
    )


if __name__ == "__main__":
    mcp.run()
