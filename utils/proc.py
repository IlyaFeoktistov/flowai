"""
Process-tree kill helper shared by cli.py's "!command" and
mcp_agent/servers/bash_exec_server.py's bash_exec — both spawn a command via
`asyncio.create_subprocess_shell` (= `sh -c ...`), and asyncio.subprocess.
Process.kill() only signals that immediate `sh -c` wrapper, not whatever it
spawned. Live-verified (20260812): even a SINGLE literal command with no
pipes/`;` (e.g. "sleep 5") forks a real child on this system's /bin/sh
rather than exec'ing directly into it — killing only the wrapper leaves
that child alive, orphaned, still holding its own inherited copy of the
wrapper's stdout/stderr pipe file descriptors. A pipe's read end (what
proc.communicate() is waiting on) only sees EOF once EVERY process holding
a copy of the write end closes it — so for a command that was killed
specifically because it never exits on its own, communicate() called right
after kill() can still hang forever: the kill "succeeds" against the
wrapper, but the caller never gets its result back, and the real target
process (the whole reason for killing anything) keeps running regardless.

killpg() was tried for exactly this and reverted elsewhere in this codebase
(see bash_exec_server.py's own docstring) — signalling a whole process
GROUP risks reaching unrelated processes that happen to share it (shell
job control, this same terminal session, ...). This instead walks
/proc/<pid>/task/*/children (Linux-only, no subprocess spawn/shell needed)
to find and kill ONLY this specific process's own descendants — scoped to
"the tree we ourselves spawned", not "everything sharing its session/pgrp".
"""
import os
import signal


def _children(pid: int) -> list[int]:
    kids: list[int] = []
    task_dir = f"/proc/{pid}/task"
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return kids
    for tid in tids:
        try:
            with open(f"{task_dir}/{tid}/children") as f:
                kids.extend(int(p) for p in f.read().split())
        except OSError:
            continue
    return kids


def kill_process_tree(pid: int, sig: int = signal.SIGKILL) -> None:
    """Kills pid AND all of its descendants, children BEFORE parent — so a
    child is never left alive even briefly in the gap between its parent
    dying and us getting around to it. Best-effort: a process that's
    already gone (ProcessLookupError) or not ours to signal
    (PermissionError) is silently skipped, same tolerance as
    bash_exec_server.py's own _is_pid_alive."""
    for child in _children(pid):
        kill_process_tree(child, sig)
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass
