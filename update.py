"""
/update — автообновление flowAI поверх git, а не пакетного менеджера.

flowAI не ставится как pip-пакет (`pip show flowai` ничего не находит) — ./flowai
(лаунчер в ~/.local/bin — его же копия) эксит .venv/bin/python прямо из ЭТОЙ
директории репозитория (см. flowAI/CLAUDE.md). Репозиторий И ЕСТЬ живое
приложение, так что "обновление" здесь — `git fetch` + `git pull --ff-only` в
этой же рабочей копии, никакого отдельного релиза/канала доставки не нужно.

Тот же best-effort/fail-open принцип, что и в doctor.py: каждый шаг
самодостаточен, сетевой сбой одной проверки не должен маскировать остальной
отчёт. НО, в отличие от doctor.py (только диагностика), run_update() реально
меняет файлы на диске — отсюда одна асимметрия: pull делается ТОЛЬКО если
рабочее дерево чистое (`git status --porcelain` пуст). Это тот же самый
репозиторий, где пользователь ведёт свою повседневную разработку (это же
самое дерево, в котором работает Claude Code в текущем чате) — тянуть поверх
незакоммиченных правок молча означало бы рисковать потерять или молча
смержить чужую и свою правку одного файла без единого предупреждения.

refresh_cache() — дешёвая, ТОЛЬКО ЧТЕНИЕ (git fetch + rev-list, без pull) —
вызывается в фоне при каждом старте cli.py, но реальный git fetch делает не
чаще раза в _CHECK_INTERVAL (см. settings.py:last_update_check) чтобы не
дёргать GitHub на каждый релонч и не задерживать старт. Найденный результат
только сохраняется в settings (last_update_check/update_commits_behind) —
шапка (ui/header.py) читает его и показывает бейдж, само обновление — только
по явной команде /update (см. cli.py), никогда не применяется само.
"""
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import settings

_REPO_ROOT = Path(__file__).resolve().parent
_GIT_TIMEOUT = 30
_PIP_TIMEOUT = 180
_CHECK_INTERVAL = timedelta(hours=6)


def _git(*args: str, timeout: int = _GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def _current_branch() -> str | None:
    r = _git("rev-parse", "--abbrev-ref", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def _commits_behind(branch: str) -> int | None:
    r = _git("rev-list", "--count", f"HEAD..origin/{branch}")
    return int(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip().isdigit() else None


def _is_dirty() -> bool:
    r = _git("status", "--porcelain")
    return r.returncode != 0 or bool(r.stdout.strip())


def refresh_cache() -> bool:
    """Обновляет settings' last_update_check/update_commits_behind не чаще
    раза в _CHECK_INTERVAL. Возвращает True, если бейдж в шапке должен
    перерисоваться (число коммитов позади реально изменилось за этот вызов)
    — вызывающий (cli.py) решает, звать ли print_header заново. Никогда не
    бросает исключение — вызывается из fire-and-forget background task
    (см. cli.py:main), необработанное исключение там не роняет процесс, но
    может засорить stderr трейсбеком поверх TUI."""
    try:
        if not (_REPO_ROOT / ".git").exists():
            return False
        last_check = settings.get("last_update_check")
        if last_check:
            elapsed = datetime.now() - datetime.fromisoformat(last_check)
            if elapsed < _CHECK_INTERVAL:
                return False

        fetch = _git("fetch", "origin")
        settings.set_value("last_update_check", datetime.now().isoformat(timespec="seconds"))
        if fetch.returncode != 0:
            return False

        branch = _current_branch()
        if not branch:
            return False
        behind = _commits_behind(branch)
        if behind is None:
            return False

        previous = settings.get("update_commits_behind") or 0
        settings.set_value("update_commits_behind", behind)
        return behind != previous
    except Exception:
        return False


async def run_update() -> str:
    """Полный /update: fetch -> сравнение -> (если чисто) pull --ff-only ->
    (если requirements.txt изменился) переустановка зависимостей в .venv.
    Возвращает Rich-markup отчёт, тот же стиль, что doctor.py:run_doctor."""
    if not (_REPO_ROOT / ".git").exists():
        return "[red]✗[/] это не git-репозиторий — автообновление недоступно"

    fetch = _git("fetch", "origin")
    settings.set_value("last_update_check", datetime.now().isoformat(timespec="seconds"))
    if fetch.returncode != 0:
        return f"[red]✗[/] `git fetch origin` не удался: {fetch.stderr.strip() or fetch.stdout.strip()}"

    branch = _current_branch()
    if not branch:
        return "[red]✗[/] не удалось определить текущую ветку (`git rev-parse --abbrev-ref HEAD`)"

    behind = _commits_behind(branch)
    if behind is None:
        return f"[red]✗[/] не удалось сравнить с origin/{branch} — есть ли такая ветка на origin?"

    if behind == 0:
        settings.set_value("update_commits_behind", 0)
        head = _git("rev-parse", "--short", "HEAD").stdout.strip()
        return f"[green]✓[/] уже последняя версия ({branch} @ {head})"

    if _is_dirty():
        return (
            f"[yellow]⚠[/] origin/{branch} впереди на {behind} коммит(ов), но в рабочей копии "
            "есть незакоммиченные правки — обновление пропущено, чтобы их не потерять. "
            "Закоммить или застэшить (`git stash`) и повтори /update."
        )

    req_path = _REPO_ROOT / "requirements.txt"
    req_before = req_path.read_text() if req_path.is_file() else ""

    pull = _git("pull", "--ff-only", "origin", branch, timeout=_GIT_TIMEOUT)
    if pull.returncode != 0:
        return f"[red]✗[/] `git pull --ff-only` не удался: {pull.stderr.strip() or pull.stdout.strip()}"

    settings.set_value("update_commits_behind", 0)
    lines = [f"[green]✓[/] подтянуто {behind} коммит(ов) из origin/{branch}"]

    req_after = req_path.read_text() if req_path.is_file() else ""
    if req_after != req_before:
        pip = _REPO_ROOT / ".venv" / "bin" / "pip"
        try:
            install = subprocess.run(
                [str(pip), "install", "-r", str(req_path)],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=_PIP_TIMEOUT,
            )
            if install.returncode == 0:
                lines.append("[green]✓[/] requirements.txt изменился — зависимости в .venv обновлены")
            else:
                lines.append(
                    "[red]✗[/] requirements.txt изменился, но `pip install -r requirements.txt` "
                    f"не удался: {install.stderr.strip()[-500:]}"
                )
        except subprocess.TimeoutExpired:
            lines.append(
                f"[red]✗[/] requirements.txt изменился, но переустановка зависимостей не "
                f"уложилась в {_PIP_TIMEOUT}с — запусти `.venv/bin/pip install -r requirements.txt` вручную"
            )

    lines.append("[dim]перезапусти flowai (exit + заново), чтобы новый код применился[/]")
    return "\n".join(lines)
