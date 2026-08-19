"""Пример per-project хука — post_file_edit/pre_commit ТОЛЬКО для этого
проекта (без манифеста: любой .py в .flowai/hooks/, ищутся функции с этими
именами, см. mcp_agent/plugins.py's docstring про discover_project_hooks).
Те же сигнатуры и контракт возврата, что у плагинных хуков
(examples/plugins/hello-world/hooks.py) — файл волен определить и то, и
другое, или только нужное. Оба могут быть sync или async."""
import re

_SECRET_RE = re.compile(r"(api[_-]?key|secret|password)\s*=\s*['\"][^'\"]{8,}", re.IGNORECASE)


def post_file_edit(path, repo_path):
    """После успешной правки — грубая эвристика на случайно вписанный
    секрет; только предупреждает, не блокирует (post_file_edit не может
    отменить уже сделанную правку)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return
    if _SECRET_RE.search(text):
        from ui.console import console
        console.print(f"[yellow]  ⚠ похоже на секрет в {path} — проверь перед коммитом[/]")


def pre_commit(command, repo_path):
    """Перед git commit — та же эвристика по незакоммиченному диффу;
    непустая строка ЗАБЛОКИРУЕТ коммит и уйдёт модели как причина."""
    import subprocess
    diff = subprocess.run(
        ["git", "diff", "--cached"], cwd=repo_path,
        capture_output=True, text=True, timeout=10,
    ).stdout
    if _SECRET_RE.search(diff):
        return "В застейдженном диффе похоже на секрет (api_key/secret/password=...) — проверь руками перед коммитом."
    return None
