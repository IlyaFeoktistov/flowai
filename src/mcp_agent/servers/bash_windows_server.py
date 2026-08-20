"""
Windows-native counterpart to bash_server.py — a SEPARATE implementation,
not the same code with if-branches, because the two OSes need genuinely
different process-management primitives, not just a different shell
syntax. Selected once, at server-registration time, by
mcp_agent/config.py:build_mcp_connections (sys.platform == "win32" picks
this module's path instead of bash_server.py's) — same public tool surface
(bash/bash_bg/bash_bg_check/bash_bg_list, same names/docstrings/output
shape), so roles.py/self_heal.py/the system prompt need no OS-aware
branching of their own: the model sees one "bash" tool regardless of host
OS. bash_server.py itself is untouched by this file's existence.

Commands run through a real POSIX-compatible shell (Git for Windows' MSYS
bash — see _resolve_bash below), not cmd.exe. The whole agent's prompts/
self-heal logic assume grep/ls/cat/&&-chaining/quoting the way bash
provides them (see bash_server.py's own _is_non_error_exit, roles.py's
tool descriptions) — cmd.exe has none of that, and Python's own
asyncio.create_subprocess_shell would silently pick cmd.exe as the shell
on Windows if used here, which is exactly what this file avoids by calling
create_subprocess_exec with an explicit bash path instead.

Process lifecycle differs from bash_server.py by OS necessity:
  - No start_new_session=True — Windows' subprocess.Popen doesn't support
    it (raises ValueError). CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS is
    the Windows equivalent for "outlive this process, don't inherit its
    console" that bash_bg needs.
  - No os.kill(pid, 0)/os.killpg — neither exists in a usable form on
    Windows. Liveness uses psutil.pid_exists (psutil is already a
    requirements.txt dependency, no new install needed); tree-kill shells
    out to the native `taskkill /T /F /PID`, which kills the whole process
    tree in one OS call — bash_server.py's Linux side needs its own /proc
    walk (utils/proc.py) purely because POSIX doesn't give you that for
    free the way taskkill does.
  - bash_bg's timeout ceiling (BG_TIMEOUT) is enforced by a second,
    independently-detached watchdog process (`powershell Start-Sleep` +
    `taskkill`) rather than wrapping the command in the POSIX `timeout`
    coreutil the Linux side uses — the watchdog targets the real Windows
    PID directly via taskkill, sidestepping any question of whether
    signals sent from inside Git Bash's MSYS environment reach a
    Windows-native PID correctly.

This is new, Windows-specific process-management code with no equivalent
prior art in this codebase — it has NOT been exercised on a real Windows
machine yet. Test bash/bash_bg/bash_bg_check/bash_bg_list for real before
relying on this in day-to-day use.

Run standalone: python -m mcp_agent.servers.bash_windows_server
"""
import asyncio
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Тот же приём, что bash_server.py/memory_server.py — запускается по
# абсолютному пути (см. config.py:_own_server), не через "-m", так что
# sys.path не содержит корень проекта сам по себе.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import storage  # noqa: E402

mcp = FastMCP("bash")

MAX_OUTPUT = 5000
TIMEOUT = 60
MAX_TIMEOUT = 600
BG_TIMEOUT = 1800
MAX_BG_JOBS = 30

_JOBS_DIR = storage.data_dir() / "bg_jobs"
_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Голова получает больше бюджета, чем хвост — тот же приём и то же
# обоснование, что в bash_server.py._sandwich_truncate (продублировано, а
# не импортировано — этот файл намеренно не зависит от bash_server.py, см.
# module docstring).
_TRUNCATE_HEAD_RATIO = 0.6

_GIT_BASH_CANDIDATES = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
]


def _resolve_bash() -> str | None:
    """FLOWAI_WINDOWS_BASH (written into .env by windows/setup.bat once it
    locates Git for Windows) wins; otherwise PATH, then well-known install
    locations. None means no POSIX-compatible shell is available — callers
    return a clear, actionable error instead of silently falling back to
    cmd.exe (which would just make every command fail in a more confusing
    way, one grep/git-status/&&-chain at a time)."""
    override = os.getenv("FLOWAI_WINDOWS_BASH")
    if override and os.path.isfile(override):
        return override
    found = shutil.which("bash")
    if found:
        return found
    for candidate in _GIT_BASH_CANDIDATES:
        if os.path.isfile(candidate):
            return candidate
    return None


_BASH = _resolve_bash()
_NO_BASH_ERROR = (
    "Error: no POSIX-compatible shell found (Git for Windows' bash.exe). "
    "Install Git for Windows (https://git-scm.com/download/win, default "
    "options are fine) or re-run windows\\setup.bat, then restart flowai."
)


def _is_non_error_exit(command: str) -> bool:
    cmd = command.lower()
    return any(x in cmd for x in ["grep", "find", "rg", "head", "tail", "cat"])


def _sandwich_truncate(text: str, max_chars: int) -> str:
    head_budget = int(max_chars * _TRUNCATE_HEAD_RATIO)
    tail_budget = max_chars - head_budget
    omitted = len(text) - max_chars
    marker = (
        f"\n...[TRUNCATED: {omitted} chars omitted from the middle "
        f"(showing first {head_budget} and last {tail_budget} of {len(text)} "
        "total) — narrow the command (grep/head/tail/-- <path>) if you need "
        "the omitted part]...\n"
    )
    return text[:head_budget] + marker + text[-tail_budget:]


def _kill_tree(pid: int) -> None:
    """taskkill /T kills the whole process tree in one native OS call —
    unlike the Linux side (utils/proc.py), no manual /proc walk needed.
    Best-effort: a pid that's already gone just makes taskkill exit
    non-zero, which is fine to ignore here."""
    subprocess.run(
        ["taskkill", "/F", "/T", "/PID", str(pid)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _is_pid_alive(pid: int) -> bool:
    import psutil
    return psutil.pid_exists(pid)


@mcp.tool()
async def bash(command: str, timeout: int = TIMEOUT) -> str:
    """Execute a bash command on the local machine and return its output.
    Use for: system info, scripts, git, installing packages, any shell command.
    Default timeout is 60s. If you already have good reason to expect THIS
    specific command to legitimately take longer (a known-slow DB query, a
    big test file, a large build step) — not a broad/unscoped command that
    should be narrowed instead — pass a bigger timeout= (up to 600s)
    instead of retrying the same call against the default and hitting the
    same wall. For anything open-ended or likely to run past a few
    minutes, use bash_bg instead — it doesn't block this turn at
    all.

    There is no real controlling terminal here (stdin is closed, no
    allocated tty at all) — an interactive TUI/game/ncurses-style program
    that reads live keypresses WILL fail here or hang waiting for input
    that will never arrive. That failure is about this environment lacking
    a tty, not evidence the program's own logic is broken — don't chase it
    as a code bug. Verify this kind of program by building it, running its
    own non-interactive test suite, or reading the code path in question —
    not by trying to run the interactive program directly."""
    if not command.strip():
        return "Error: no command specified"
    if _BASH is None:
        return _NO_BASH_ERROR
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))

    try:
        # stdin=DEVNULL — same reasoning as bash_server.py's bash: without
        # it a command that reads stdin blocks on this process's real
        # stdin (the MCP stdio transport) for the whole timeout instead of
        # getting an immediate EOF.
        proc = await asyncio.create_subprocess_exec(
            _BASH, "-c", command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except Exception as e:
        return f"Error: {e}"

    timed_out = False

    async def _watchdog() -> None:
        nonlocal timed_out
        await asyncio.sleep(effective_timeout)
        if proc.returncode is None:
            timed_out = True
            _kill_tree(proc.pid)

    watchdog = asyncio.create_task(_watchdog())
    try:
        stdout_b, stderr_b = await proc.communicate()
    except Exception as e:
        watchdog.cancel()
        return f"Error: {e}"
    watchdog.cancel()

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    stderr = stderr_b.decode("utf-8", errors="replace").strip()
    output = "\n".join(x for x in [stdout, stderr] if x)

    if timed_out:
        if effective_timeout < MAX_TIMEOUT:
            retry_hint = (
                f"If this specific command is known to legitimately need "
                f"more time (not just broad/unscoped), retry it with a "
                f"bigger timeout= (up to {MAX_TIMEOUT}s) instead of the "
                "same default. Otherwise narrow it to the exact file(s)/"
                "path/query you actually need."
            )
        else:
            retry_hint = (
                f"Already at the {MAX_TIMEOUT}s cap for this tool — if it "
                "still isn't enough, this belongs in bash_bg instead "
                "(doesn't block this turn at all), not a bigger timeout= "
                "here."
            )
        partial = f" Output captured before it was killed:\n{output}" if output else ""
        return (
            f"Error: command timed out after {effective_timeout}s and was "
            "killed. This can mean the command itself was too broad/slow "
            "(e.g. scanning the whole repo or vendor/ instead of specific "
            "files), that it's a single command that genuinely needs more "
            "time, OR that it was waiting on input it was never going to "
            "get (stdin is closed for this tool) — if so, this is not an "
            "interactive shell; rerun the underlying program with its "
            f"input passed non-interactively instead. {retry_hint}{partial}"
        )

    if proc.returncode == 0 or _is_non_error_exit(command):
        result = output or "(no output)"
    else:
        result = f"Error (exit {proc.returncode}): {output}"

    if len(result) > MAX_OUTPUT:
        result = _sandwich_truncate(result, MAX_OUTPUT)
    return result


def _jobs_conn():
    conn = storage.connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bg_jobs ("
        "job_id TEXT PRIMARY KEY, command TEXT NOT NULL, pid INTEGER NOT NULL, "
        "started_at REAL NOT NULL, output_path TEXT NOT NULL, exit_path TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _prune_finished_bg_jobs(conn) -> None:
    rows = conn.execute(
        "SELECT job_id, output_path, exit_path FROM bg_jobs ORDER BY started_at"
    ).fetchall()
    finished = [r for r in rows if os.path.exists(r[2])]
    excess = len(finished) - MAX_BG_JOBS
    if excess <= 0:
        return
    for job_id, output_path, exit_path in finished[:excess]:
        conn.execute("DELETE FROM bg_jobs WHERE job_id = ?", (job_id,))
        for p in (output_path, exit_path):
            try:
                os.remove(p)
            except OSError:
                pass
    conn.commit()


@mcp.tool()
async def bash_bg(command: str) -> str:
    """Start a bash command in the BACKGROUND and return a job id
    immediately, instead of blocking this turn until it finishes. Use for
    anything that legitimately takes longer than bash's 60s timeout —
    a test suite, a build, a long migration/import script. Check on it with
    bash_bg_check(job_id) later — there's no automatic notification
    when it's done, so don't call bash_bg_check in a tight loop right
    after starting it; do other useful work (or finish your response) and
    check back after a plausible amount of time has passed. Same no-tty
    environment as bash — don't use this to run an interactive TUI/game
    "in the background" expecting it to work, it will just fail the same
    way (see bash's own docstring)."""
    if not command.strip():
        return "Error: no command specified"
    if _BASH is None:
        return _NO_BASH_ERROR

    conn = _jobs_conn()
    try:
        rows = conn.execute("SELECT pid, exit_path FROM bg_jobs").fetchall()
        running = sum(1 for pid, exit_path in rows if not os.path.exists(exit_path) and _is_pid_alive(pid))
        if running >= MAX_BG_JOBS:
            return f"Error: {MAX_BG_JOBS} background jobs already running — check bash_bg_list and wait for one to finish"

        _prune_finished_bg_jobs(conn)

        job_id = uuid.uuid4().hex[:10]
        output_path = _JOBS_DIR / f"{job_id}.out"
        exit_path = _JOBS_DIR / f"{job_id}.exit"

        # command goes in RAW (not quoted) — it's the script body itself,
        # same as bash() above passes it to create_subprocess_exec (may
        # contain its own &&/pipes/redirects).
        wrapped = (
            f"{command} "
            f"> {shlex.quote(str(output_path))} 2>&1; "
            f"echo $? > {shlex.quote(str(exit_path))}"
        )
        # CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS — Windows' equivalent
        # of start_new_session=True (unavailable on Windows, see module
        # docstring): survives this MCP-server process dying right after
        # this function returns, same reason bash_server.py's Linux side
        # needs a real detached OS process rather than an asyncio task.
        proc = subprocess.Popen(
            [_BASH, "-c", wrapped],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        # BG_TIMEOUT ceiling — own independently-detached watchdog rather
        # than wrapping the command in the POSIX `timeout` coreutil (see
        # module docstring for why): targets proc.pid directly via
        # taskkill, so it works regardless of whether the job itself is a
        # bash builtin, a native .exe, or a whole pipeline.
        subprocess.Popen(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"Start-Sleep -Seconds {BG_TIMEOUT}; "
                f"taskkill /F /T /PID {proc.pid} 2>$null",
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        conn.execute(
            "INSERT INTO bg_jobs (job_id, command, pid, started_at, output_path, exit_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, command, proc.pid, time.time(), str(output_path), str(exit_path)),
        )
        conn.commit()
        return f'Started as job "{job_id}". Check progress/result with bash_bg_check("{job_id}").'
    finally:
        conn.close()


@mcp.tool()
async def bash_bg_check(job_id: str) -> str:
    """Check a background job started by bash_bg: still running, or
    finished with its output (same truncation as bash). Safe to call
    repeatedly — checking a finished job doesn't clear its stored result."""
    conn = _jobs_conn()
    try:
        row = conn.execute(
            "SELECT command, pid, exit_path, output_path FROM bg_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return f"Error: no such job '{job_id}' (never started, or evicted after {MAX_BG_JOBS} more recent jobs finished)"

    command, pid, exit_path, output_path = row
    if not os.path.exists(exit_path):
        if _is_pid_alive(pid):
            return f'{job_id} is still running (command: {command!r})'
        return (
            f"{job_id}: the process is gone but never wrote its exit "
            "marker — it was likely killed unexpectedly (OOM, manual kill, "
            "system shutdown, or the bash_bg timeout watchdog) rather than "
            f"finishing normally. Whatever output it produced before that "
            f"is in {output_path!r} if you need to inspect it directly."
        )

    try:
        exit_code = int(Path(exit_path).read_text().strip())
    except (OSError, ValueError):
        exit_code = None
    try:
        output = Path(output_path).read_text(errors="replace").strip()
    except OSError:
        output = ""
    if len(output) > MAX_OUTPUT:
        output = _sandwich_truncate(output, MAX_OUTPUT)

    if exit_code == 0 or _is_non_error_exit(command):
        return f"{job_id} finished — {output or '(no output)'}"
    return f"{job_id} finished — Error (exit {exit_code}): {output}"


@mcp.tool()
async def bash_bg_list() -> str:
    """List background jobs started with bash_bg (most recent first,
    up to the retention cap), with their command and current status — check
    this before starting a new background command if you're not sure what's
    already running."""
    conn = _jobs_conn()
    try:
        rows = conn.execute(
            "SELECT job_id, command, pid, exit_path FROM bg_jobs ORDER BY started_at DESC"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "No background jobs started"
    lines = []
    for job_id, command, pid, exit_path in rows:
        if os.path.exists(exit_path):
            status = "done"
        elif _is_pid_alive(pid):
            status = "running"
        else:
            status = "gone (killed unexpectedly, no exit code recorded)"
        lines.append(f"{job_id}: {status} — {command}")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
