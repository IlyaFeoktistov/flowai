"""
Полный, всегда включённый (не под DEBUG) лог происходящего за один запуск
flowai — один JSONL-файл на процесс, для быстрого чтения человеком/другим
инструментом (grep/tail/cat) вместо ad-hoc SQL-запросов к episodic_messages
или скроллбека терминала, где сейчас теряются DEBUG-принты
(console.print(f"[dim][...] ...")) из mcp_agent/agent.py и self_heal.py.

storage.data_dir() (~/.local/share/flowai/), не системный temp — /tmp не
гарантирует переживание перезагрузки: systemd's systemd-tmpfiles-clean
обычно чистит /tmp при каждом старте, retention здесь ни при чём (TTL
защищает от накопления, а не от исчезновения при перезагрузке), так что
лог, к которому может понадобиться вернуться уже после перезагрузки
машины, там ненадёжен. data_dir() — тот же каталог, что уже хранит
flowai.db, переживает перезагрузки по определению и не требует root (в
отличие от /var/log, куда обычный пользователь писать не может — 0775
root:syslog на большинстве систем).

Использует mcp_agent.snapshots._SESSION_ID (уже существующий per-process
uuid, см. его собственный докстринг) вместо отдельного идентификатора —
один и тот же процесс не должен иметь два разных "session id" для двух
разных диагностических механизмов.
"""
import json
import os
import time
from datetime import datetime

import storage
from mcp_agent.snapshots import _SESSION_ID

_LOG_DIR = str(storage.data_dir() / "run-logs")
_LOG_PATH = os.path.join(_LOG_DIR, f"{_SESSION_ID}.jsonl")

_RETENTION_DAYS = 3

_pruned_once = False


def _prune_old_logs() -> None:
    """Раз за процесс (перед первой записью) удаляет файлы старше
    _RETENTION_DAYS — единственный способ, которым эти логи вообще
    исчезают: data_dir() постоянный (переживает перезагрузки), ни ОС, ни
    что-либо другое в проекте его не подчищает само."""
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
    из-за, например, заполненного диска.

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
