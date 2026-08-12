"""
Полный, всегда включённый (не под DEBUG) лог происходящего за один запуск
flowai — один JSONL-файл на процесс в системном temp, для быстрого чтения
человеком/другим инструментом (grep/tail/cat) вместо ad-hoc SQL-запросов к
episodic_messages или скроллбека терминала, где сейчас теряются DEBUG-принты
(console.print(f"[dim][...] ...")) из mcp_agent/agent.py и self_heal.py.

Файл в /tmp, а не в постоянном data_dir() — это диагностический выхлоп, а не
данные, которые нужно хранить бессрочно; retention (_prune_old_logs) чистит
его сам, той же логикой, что уже применена к MCP-подпроцессным логам
(config.py:_LOG_DIR), но с добавленным TTL, которого там нет.

Использует mcp_agent.snapshots._SESSION_ID (уже существующий per-process
uuid, см. его собственный докстринг) вместо отдельного идентификатора —
один и тот же процесс не должен иметь два разных "session id" для двух
разных диагностических механизмов.
"""
import json
import os
import tempfile
import time
from datetime import datetime

from mcp_agent.snapshots import _SESSION_ID

_LOG_DIR = os.path.join(tempfile.gettempdir(), "flowai-run-logs")
_LOG_PATH = os.path.join(_LOG_DIR, f"{_SESSION_ID}.jsonl")

_RETENTION_DAYS = 3

_pruned_once = False


def _prune_old_logs() -> None:
    """Раз за процесс (перед первой записью) удаляет файлы старше
    _RETENTION_DAYS — единственный способ, которым эти логи вообще
    исчезают, ни ОС, ни что-либо другое в проекте их не подчищает (в
    отличие от файлов /tmp, которые переживают до перезагрузки, а не
    "какое-то время")."""
    global _pruned_once
    if _pruned_once:
        return
    _pruned_once = True
    cutoff = time.time() - _RETENTION_DAYS * 86400
    try:
        names = os.listdir(_LOG_DIR)
    except OSError:
        return
    for name in names:
        path = os.path.join(_LOG_DIR, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            continue


def log_event(event_kind: str, **fields) -> None:
    """Дописывает одну JSON-строку. Никогда не поднимает исключение наружу —
    диагностический сайд-канал не должен ронять реальный ход работы агента
    из-за, например, заполненного /tmp.

    Параметр называется event_kind, а не kind: verdict-словари и событие
    self_heal_reject уже несут свой собственный содержательный ключ "kind"
    (например "execution_failure") — вызовы вида log_event("verdict",
    **verdict) распаковывают его в **fields, и совпадение имени параметра с
    ключом из fields упало бы TypeError "multiple values for argument"."""
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _prune_old_logs()
        line = json.dumps(
            {"ts": datetime.now().isoformat(timespec="seconds"), "event": event_kind, **fields},
            ensure_ascii=False, default=str,
        )
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
    except OSError:
        pass
