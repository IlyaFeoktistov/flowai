"""
Кастомный MCP-сервер: bash_exec.

Готового shell/exec-сервера с нужной permission-гранулярностью в
официальном/community MCP-реестре нет (проверено: только filesystem, git,
fetch, everything, pdf) — поэтому этот наш. Сам permission-гейт НЕ здесь:
подтверждение — забота клиента (HumanInTheLoopMiddleware в mcp_agent/agent.py),
как и для всех остальных MCP-серверов в этой миграции — единая точка,
а не логика внутри каждого тула (как было в tools/bash_exec.py + tools/confirm.py).

bash_exec_bg/bash_exec_bg_check/bash_exec_bg_list — то же самое, но не
блокируя ход: старт возвращает job_id сразу, результат забирается позже
отдельным вызовом. Нужны для команд, которые реально долго идут (тест-сьют,
сборка, миграция) — иначе только bash_exec с TIMEOUT=60s, то есть либо
уложиться в минуту, либо получить "command timed out" на честно работающей
команде.

Живой баг (найден по докладу пользователя "кажется, бекграунд не работает"):
раньше состояние job'ов держалось в module-level dict (_BG_JOBS) и сам
процесс запускался через asyncio.create_subprocess_shell +
asyncio.create_task — ровно тот же приём, что и у обычного bash_exec, просто
без ожидания. Расчёт был на то, что "MCP-сервер поднимается один раз на
сессию flowai и живёт до её конца" (тот же принцип, что у lsp_server.py) —
но это предположение НЕВЕРНО для stdio-транспорта в установленной версии
langchain-mcp-adapters (0.3.0): её MultiServerMCPClient.get_tools() создаёт
НОВУЮ сессию (= новый подпроцесс этого файла с нуля) НА КАЖДЫЙ вызов тула,
а не одну на всю сессию (см. её собственный докстринг: "A new session will
be created for each tool call"). Из этого следует два независимых провала
старой схемы:
  1. _BG_JOBS — пустой dict в СВЕЖЕМ процессе на каждый вызов: job,
     запущенный в bash_exec_bg, физически не существовал для процесса,
     обслуживающего следующий bash_exec_bg_check — тот всегда отвечал "no
     such job", что старый код даже предвидел в тексте своей же ошибки
     ("...or the server was restarted") — здесь этот случай происходит
     ВСЕГДА, не как редкий крайний случай.
  2. asyncio.create_task(...) — фоновая задача внутри ТЕКУЩЕГО event loop,
     который сам вот-вот остановится (процесс завершается сразу после
     возврата ответа от bash_exec_bg) — сама команда либо не успевала
     запуститься, либо обрывалась вместе с процессом-родителем, а не
     продолжала жить как реальный фоновый процесс ОС.

Исправление — состояние переживает смену процесса ДВУМЯ способами разом:
  - job-реестр (job_id/command/pid/пути к файлам) — в общей SQLite
    (storage.py:connect(), тот же файл, что settings/memory/knowledge), не
    в памяти процесса.
  - сама команда — НАСТОЯЩИЙ отсоединённый процесс ОС (subprocess.Popen с
    start_new_session=True, тот же приём, что nohup/setsid: новая сессия
    ОС, не привязанная к процессу-родителю, переживает его смерть и не
    получает его сигналов), с выводом в файл (не в pipe — держать pipe
    открытым через границу процессов невозможно) и отдельным
    файлом-маркером exit-кода, который пишет сама обёрнутая команда, когда
    реально завершится. bash_exec_bg_check/bash_exec_bg_list читают эти
    файлы + пробуют databaseNexus PID (os.kill(pid, 0)) — никакого
    ожидания в асинхронном тасте, которому неоткуда пережить свой процесс.

Настоящего "разбуди при готовности" всё ещё нет — flowai это синхронный REPL
без idle-цикла, слушать который можно было бы вне хода. Разбудить агент
ПРЯМО когда фоновая команда закончится — отдельная работа над ui/ (нужен
асинхронный хук в prompt_toolkit, а не просто MCP-тул).

Запуск (обычно через MultiServerMCPClient, но можно и вручную):
    python3 -m mcp_agent.servers.bash_exec_server
"""
import asyncio
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# Тот же приём, что memory_server.py/knowledge_server.py — storage.py лежит
# в корне проекта, а этот файл запускается по абсолютному пути (см.
# config.py:_own_server), не через "-m", так что sys.path не содержит
# корень проекта сам по себе.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import storage  # noqa: E402

mcp = FastMCP("bash_exec")

MAX_OUTPUT = 5000
TIMEOUT = 60
# Потолок для bash_exec's own timeout= argument (see bash_exec below) — a
# call this long still blocks the WHOLE synchronous tool call/turn, unlike
# bash_exec_bg (which doesn't block at all). Past this, bash_exec_bg is the
# right tool, not a bigger timeout= here.
MAX_TIMEOUT = 600

# Фоновые команды не относятся 60-секундным TIMEOUT'ом — это специально для
# команд, которые дольше секунд, но всё равно нужен потолок, чтобы
# зависший/забытый процесс не копился в системе вечно. Обеспечивается самой
# ОС-командой `timeout` (обёртка запускаемой команды, см. bash_exec_bg) —
# раньше это был asyncio.wait_for, который здесь неприменим (нечему больше
# ждать в умирающем процессе).
BG_TIMEOUT = 1800
# Сколько ЗАВЕРШЁННЫХ job'ов держим в реестре одновременно — без этого
# история фоновых запусков за долгую жизнь ~/.local/share/flowai/flowai.db
# росла бы неограниченно. Job'ы, которые ещё выполняются, эту чистку не
# затрагивают никогда.
MAX_BG_JOBS = 30

_JOBS_DIR = storage.data_dir() / "bg_jobs"
_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# Голова получает больше бюджета, чем хвост — см. _sandwich_truncate в
# mcp_agent/agent.py (тот же приём, продублирован здесь, потому что этот
# кап срабатывает раньше и на меньшем пороге, чем общая обёртка
# _cap_tool_output, и раньше топил хвост вывода целиком).
_TRUNCATE_HEAD_RATIO = 0.6


# Живой прогон (mail-server, 20260707-163337-11afd268): `php -l` on a file
# with a genuine syntax error exits non-zero AND prints its "Errors parsing
# FILE / PHP Parse error: ..." message to STDOUT (not stderr) — the old
# fallback `return bool(stdout.strip())` treated "printed anything to
# stdout" as proof of success for ANY command, so this real, non-zero-exit
# failure got returned as if it succeeded (no "Error (exit N):" prefix at
# all). self_heal.py's execution-failure check (see
# _execution_evidence_shows_failure) never had a chance to catch it — this
# tool had already lied about the exit code before that check ever ran.
# Most compilers/linters/test-runners (php -l, tsc, phpunit, pytest, ...)
# print their failure output to stdout, not stderr, so this wasn't a narrow
# edge case — it silently defeated exit-code checking for most real
# failures, not just this one.
#
# The grep/find/rg/head/tail/cat allowlist stays: THOSE specific commands
# legitimately exit non-zero for reasons that aren't a real failure (grep/rg
# exit 1 on "no matches", not "grep is broken") — every other command's
# non-zero exit is trusted as a real failure regardless of what it printed.
def _is_non_error_exit(command: str) -> bool:
    cmd = command.lower()
    return any(x in cmd for x in ["grep", "find", "rg", "head", "tail", "cat"])


def _sandwich_truncate(text: str, max_chars: int) -> str:
    """Итог команды (тест-раннера, линтера, git diff) почти всегда в конце
    вывода — чистая head-обрезка топила именно ту часть, ради которой
    команда вообще запускалась."""
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


@mcp.tool()
async def bash_exec(command: str, timeout: int = TIMEOUT) -> str:
    """Execute a bash command on the local machine and return its output.
    Use for: system info, scripts, git, installing packages, any shell command.
    Default timeout is 60s. If you already have good reason to expect THIS
    specific command to legitimately take longer (a known-slow DB query, a
    big test file, a large build step) — not a broad/unscoped command that
    should be narrowed instead — pass a bigger timeout= (up to 600s)
    instead of retrying the same call against the default and hitting the
    same wall. For anything open-ended or likely to run past a few
    minutes, use bash_exec_bg instead — it doesn't block this turn at
    all."""
    if not command.strip():
        return "Error: no command specified"
    effective_timeout = max(1, min(int(timeout), MAX_TIMEOUT))

    try:
        # stdin=DEVNULL — БЕЗ этого create_subprocess_shell оставляет
        # дочернему процессу stdin ЭТОГО (bash_exec_server'а) процесса как
        # есть, то есть настоящий терминальный stdin (stdio-транспорт MCP,
        # см. модульный docstring). Живой инцидент (20260812): скомпилированный
        # Go-бинарник читал ввод через bufio.NewReader(os.Stdin) — bash_exec
        # его честно запустил и завис на 60-секундном таймауте... кроме
        # того, что таймаут не успел сработать (родительский процесс
        # оборвался раньше), и осиротевший бинарник остался висеть НАПРЯМУЮ
        # на реальном терминале пользователя (pts/3), блокируя его вплоть до
        # ручного kill -9. С DEVNULL любое чтение stdin получает
        # МГНОВЕННЫЙ EOF вместо зависания — интерактивная программа тут же
        # падает с понятной ошибкой (модель это видит и может поправить
        # программу или передать вход через pipe/`<file`), а не блокируется
        # незаметно на неопределённый срок, воруя чужой терминал в придачу.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=effective_timeout)
        stdout = stdout_b.decode("utf-8", errors="replace").strip()
        stderr = stderr_b.decode("utf-8", errors="replace").strip()
        output = "\n".join(x for x in [stdout, stderr] if x)

        if proc.returncode == 0 or _is_non_error_exit(command):
            result = output or "(no output)"
        else:
            result = f"Error (exit {proc.returncode}): {output}"

        if len(result) > MAX_OUTPUT:
            result = _sandwich_truncate(result, MAX_OUTPUT)
        return result

    except asyncio.TimeoutError:
        # Live run: a repo-wide `find . -exec php -l {} \;` (no path/name
        # narrowing) hit this timeout, and the model retried essentially
        # the SAME broad scan again later in the same turn instead of
        # narrowing it — the message now says outright not to repeat it.
        #
        # Tried (and reverted): killing the whole process group via
        # start_new_session=True + os.killpg() to also clean up children a
        # shell pipeline/`find -exec` spawns, since proc.kill() only kills
        # the `/bin/sh -c` wrapper. Verified live in this sandboxed
        # environment that killpg's blast radius wasn't reliably scoped to
        # just this subprocess's own tree — too risky here. Left as plain
        # proc.kill() (only the immediate wrapper) — an orphaned grandchild
        # process is a smaller problem than accidentally signaling
        # processes outside this tool call.
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        # Two different retries make sense depending on WHY it timed out —
        # narrowing helps a broad/unscoped scan, a bigger timeout= helps a
        # single, genuinely slow command — so name both instead of only
        # ever telling the model to narrow (a real live query it needs to
        # wait longer for isn't "too broad", it's just legitimately slow).
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
                "still isn't enough, this belongs in bash_exec_bg instead "
                "(doesn't block this turn at all), not a bigger timeout= "
                "here."
            )
        return (
            f"Error: command timed out after {effective_timeout}s. This "
            "can mean the command itself was too broad/slow (e.g. scanning "
            "the whole repo or vendor/ instead of specific files) — that "
            "says nothing about whether any particular file is correct — "
            f"or that it's a single command that genuinely needs more time. {retry_hint}"
        )
    except Exception as e:
        return f"Error: {e}"


def _jobs_conn():
    """Свежее sqlite-соединение на каждый вызов, не кешируется на уровне
    модуля (в отличие от snapshots.py:_snapshot_conn) — этот процесс живёт
    ровно один вызов тула (см. модульный docstring), кеш пережил бы этот
    же самый вызов и ничего не выиграл бы."""
    conn = storage.connect()
    conn.execute(
        "CREATE TABLE IF NOT EXISTS bg_jobs ("
        "job_id TEXT PRIMARY KEY, command TEXT NOT NULL, pid INTEGER NOT NULL, "
        "started_at REAL NOT NULL, output_path TEXT NOT NULL, exit_path TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists, just not ours to signal — still alive
    return True


def _prune_finished_bg_jobs(conn) -> None:
    """Держит не больше MAX_BG_JOBS ЗАВЕРШЁННЫХ записей (exit_path уже
    существует) — старейшие по started_at отбрасываются вместе с их
    output/exit файлами. Job'ы, которые ещё выполняются, никогда не
    трогаются, сколько бы их ни накопилось."""
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
async def bash_exec_bg(command: str) -> str:
    """Start a bash command in the BACKGROUND and return a job id
    immediately, instead of blocking this turn until it finishes. Use for
    anything that legitimately takes longer than bash_exec's 60s timeout —
    a test suite, a build, a long migration/import script. Check on it with
    bash_exec_bg_check(job_id) later — there's no automatic notification
    when it's done, so don't call bash_exec_bg_check in a tight loop right
    after starting it; do other useful work (or finish your response) and
    check back after a plausible amount of time has passed."""
    if not command.strip():
        return "Error: no command specified"

    conn = _jobs_conn()
    try:
        rows = conn.execute("SELECT pid, exit_path FROM bg_jobs").fetchall()
        running = sum(1 for pid, exit_path in rows if not os.path.exists(exit_path) and _is_pid_alive(pid))
        if running >= MAX_BG_JOBS:
            return f"Error: {MAX_BG_JOBS} background jobs already running — check bash_exec_bg_list and wait for one to finish"

        _prune_finished_bg_jobs(conn)

        job_id = uuid.uuid4().hex[:10]
        output_path = _JOBS_DIR / f"{job_id}.out"
        exit_path = _JOBS_DIR / f"{job_id}.exit"

        # timeout — тот же BG_TIMEOUT-потолок, что раньше был на
        # asyncio.wait_for, только теперь его обеспечивает сама ОС-команда,
        # а не наш процесс (у которого нет шанса дождаться). echo $? —
        # exit-код именно обёрнутой команды (timeout вернёт 124, если сам
        # её убил по таймауту — bash_exec_bg_check это различает).
        # start_new_session=True — новая сессия ОС (как nohup/setsid):
        # процесс переживает смерть ЭТОГО (родительского) процесса, которая
        # наступит сразу после того, как эта функция вернёт ответ — см.
        # модульный docstring про то, почему это критично, не опционально.
        wrapped = (
            f"(timeout {BG_TIMEOUT} sh -c {shlex.quote(command)}) "
            f"> {shlex.quote(str(output_path))} 2>&1; "
            f"echo $? > {shlex.quote(str(exit_path))}"
        )
        proc = subprocess.Popen(
            ["sh", "-c", wrapped],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        conn.execute(
            "INSERT INTO bg_jobs (job_id, command, pid, started_at, output_path, exit_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, command, proc.pid, time.time(), str(output_path), str(exit_path)),
        )
        conn.commit()
        return f'Started as job "{job_id}". Check progress/result with bash_exec_bg_check("{job_id}").'
    finally:
        conn.close()


@mcp.tool()
async def bash_exec_bg_check(job_id: str) -> str:
    """Check a background job started by bash_exec_bg: still running, or
    finished with its output (same truncation as bash_exec). Safe to call
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
            "system shutdown) rather than finishing normally. Whatever "
            f"output it produced before that is in {output_path!r} if you "
            "need to inspect it directly."
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

    if exit_code == 124:
        return f"{job_id} timed out after {BG_TIMEOUT}s and was killed. Output so far:\n{output or '(no output)'}"
    if exit_code == 0 or _is_non_error_exit(command):
        return f"{job_id} finished — {output or '(no output)'}"
    return f"{job_id} finished — Error (exit {exit_code}): {output}"


@mcp.tool()
async def bash_exec_bg_list() -> str:
    """List background jobs started with bash_exec_bg (most recent first,
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
