"""
Кастомный MCP-сервер: git-операции, которых нет в стороннем mcp-server-git.

Живой инцидент: модель решила откатить свою же правку файла и не нашла для
этого инструмента — git_checkout из mcp-server-git переключает ВЕТКИ
(GitTools.CHECKOUT -> git_checkout(repo, branch_name), см. mcp_server_git/
server.py), а не отдельные файлы; git_reset только снимает файлы со stage,
содержимого не трогает. За неимением настоящего "верни файл как в git" тула
модель попыталась восстановить содержимое сама через write_file — и
переписала файл на 1654 строки почти пустым огрызком, потому что не могла
дословно удержать в памяти большой файл. Этот тул закрывает именно этот
пробел: реальный `git checkout <ref> -- <path>`, гарантированно
байт-в-байт, а не реконструкция по памяти.

Запуск: python3 -m mcp_agent.servers.git_extra_server
"""
import asyncio
import shlex

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("git_extra")

TIMEOUT = 15


async def _run_git(args: list[str]) -> tuple[str, int]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        out = (stdout_b.decode(errors="replace") + stderr_b.decode(errors="replace")).strip()
        return out, proc.returncode
    except asyncio.TimeoutError:
        return "(timeout)", 1
    except Exception as e:
        return f"(error: {e})", 1


@mcp.tool()
async def git_restore_file(path: str, ref: str = "HEAD") -> str:
    """Restore ONE file to its exact byte-for-byte content from git — the
    ONLY correct way to undo/revert edits to a file. Runs `git checkout <ref>
    -- <path>`, which discards ALL uncommitted changes (staged and unstaged)
    to that file and replaces it with the version from <ref> (default HEAD =
    last commit). Use this any time you or the user want to undo a file
    edit. NEVER try to reconstruct original content yourself with
    write_file from memory — you cannot reliably recall a large file
    byte-for-byte, and a wrong reconstruction silently replaces good code
    with a broken guess instead of failing loudly like this tool does on a
    bad path/ref. This is destructive and irreversible for anything not
    committed to <ref> — only call it when you're sure the file's current
    uncommitted state should be thrown away."""
    path = path.strip()
    if not path:
        return "Error: path is required"
    ref = ref.strip() or "HEAD"

    out, rc = await _run_git(["checkout", ref, "--", path])
    if rc != 0:
        return f"Error (exit {rc}): {out or 'git checkout failed'}"
    return f"Restored {path!r} to its content at {ref!r}." + (f"\n{out}" if out else "")


if __name__ == "__main__":
    mcp.run()
