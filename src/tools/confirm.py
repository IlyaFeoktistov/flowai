import asyncio
import select
import sys
import termios
import tty
from ui.console import console as _console
_always_approve = False
_approved_actions: set[str] = set()
_app = None  # FlowAIApp instance, set via connect_app()

# С параллельным executor'ом несколько tool_calls могут одновременно попросить
# подтверждение — без лока это гонка за raw-mode stdin/TUI-диалог. Лок
# сериализует только сам диалог, не выполнение инструмента.
_permission_lock = asyncio.Lock()

# Сколько permission-диалогов сейчас реально ждут ответа человека.
# executor.py читает это через has_pending_prompt(), чтобы НЕ засчитывать
# время ожидания подтверждения в TOOL_TIMEOUT — иначе долгое размышление
# пользователя над Y/N превращается в "инструмент не ответил за 20с" и
# tool_call молча улетает в status=error без права на ответ.
_pending_prompts = 0

# Авто-approve по первому слову команды ("bash:cd") небезопасен, как
# только в команде есть цепочка/пайп — "cd /tmp && rm -rf ~" и "cd /tmp"
# начинаются одним и тем же первым словом, но вторая половина первой команды
# пользователем не проверяется. Тот же класс проблемы уже пофикшен для
# git_* тулов (см. ask_user_tool.py:_action_and_detail — у каждого своя
# action-ключ вместо общего "bash" с матчем по первому слову) — тот фикс не
# покрывает сам bash, где команда — не фиксированное имя тула, а
# произвольная строка, которая ЛЕГКО дополняется после уже одобренного
# префикса. Когда в команде есть эти маркеры, гранулярность "первое слово"
# отключается — approve/remember работает только по ТОЧНОМУ совпадению всей
# строки команды, не по префиксу.
_SHELL_CHAIN_MARKERS = ("&&", "||", ";", "|", "`", "$(")


def _bash_auto_approve_key(detail: str) -> str | None:
    """None — небезопасно приближать/запоминать по первому слову (есть
    цепочка/пайп/подстановка); вызывающий код должен в этом случае матчить
    ТОЛЬКО полную строку команды, не префикс."""
    stripped = detail.strip()
    if not stripped or any(marker in stripped for marker in _SHELL_CHAIN_MARKERS):
        return None
    return stripped.split()[0]


def _remember_bash_approval(detail: str) -> str:
    """Общая логика для всех трёх мест, где пользователь жмёт "A" (всегда) —
    возвращает текст для вывода в консоль. Составная команда запоминается
    целиком (не даёт разрешения на "любую команду с тем же первым словом"),
    простая — по первому слову, как и раньше."""
    stripped = detail.strip()
    cmd_name = _bash_auto_approve_key(detail)
    if cmd_name is not None:
        _approved_actions.add(f"bash:{cmd_name}")
        return f"[dim]  → {cmd_name} авто-одобрен на эту сессию[/]\n"
    _approved_actions.add(stripped)
    return "[dim]  → эта составная команда авто-одобрена на эту сессию (точное совпадение, не префикс)[/]\n"


def has_pending_prompt() -> bool:
    return _pending_prompts > 0


# Ответ пользователя на ask_user, который сам по себе — просьба прервать
# ход, не содержательный ответ по теме заданного вопроса. Живой инцидент:
# Planner (mcp_agent/stage_runner.py — punt-to-user rescue) заблудился,
# спросил бессмысленный уточняющий вопрос про путь к файлу, пользователь
# ответил "остановись" — и это ушло дальше как ОБЫЧНЫЙ текст в guidance для
# следующей попытки той же стадии: self-heal исправно продолжил ретраить
# (Planner обязан вызвать ask_user перед завершением), полностью
# игнорируя то, что пользователь просил прекратить, — Planner дошёл до
# max_attempts, так и не остановившись. Точное совпадение ПОСЛЕ normalize,
# не substring — иначе легитимный ответ вроде "останови на варианте 2"
# ложно сработал бы как стоп-команда.
_STOP_PHRASES = frozenset({
    "стоп", "стой", "останов", "остановись", "останови", "остановите",
    "хватит", "прекрати", "прекратите", "отмена", "отмени", "отставить",
    "stop", "cancel", "abort", "quit",
})


def _is_stop_intent(text: str) -> bool:
    return text.strip().lower().strip(" !.?…\n\t") in _STOP_PHRASES

_ACTION_LABELS: dict[str, str] = {
    "bash":         "bash-команды",
    "read_file":    "чтение файлов",
    "write_file":   "запись файлов",
    "patch_file":   "редактирование файлов",
    "append_file":  "добавление в файлы",
    "delete_file":  "удаление файлов",
    "list_dir":     "просмотр директорий",
    # mcp_agent/ask_user_tool.py:_OutOfProjectWriteApprovalMiddleware —
    # запись тулом file_ops_server.py (write_file/edit_file) по пути вне
    # repo_path текущего хода.
    "write_outside_project": "запись вне текущего проекта",
}


def connect_app(app) -> None:
    global _app
    _app = app


def _reset_session():
    global _always_approve, _approved_actions
    _always_approve = False
    _approved_actions = set()


def _read_key_sync() -> str:
    """Reads one keypress without Enter (raw mode). Fallback — readline."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline().strip().lower()
        return line[:1] if line else "n"
    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except Exception:
        line = sys.stdin.readline().strip().lower()
        return line[:1] if line else "n"
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Drain escape sequences (arrow keys etc.)
        if ch == "\x1b":
            while select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _ask_permission_sync(action: str, detail: str) -> bool:
    """Synchronous dialog — called via run_in_terminal when TUI is active."""
    global _always_approve

    label = _ACTION_LABELS.get(action, action)

    _console.print()
    _console.print(f"[bold yellow]  ⚠  Запрос разрешения[/]")
    _console.print(f"[bright_black]  Действие:[/] [cyan]{action}[/]")
    for i, line in enumerate(detail.splitlines()):
        prefix = "  Команда: " if i == 0 else "           "
        _console.print(f"[bright_black]{prefix}[/] [white]{line}[/]")

    while True:
        _console.print(
            f"\n"
            f"  [bold green][Y][/] [green]да[/]   "
            f"[bold blue][A][/] [blue]да, всегда[/]   "
            f"[bold red][N][/] [red]нет[/]  › ",
            end="",
        )
        sys.stdout.flush()

        ch = _read_key_sync()
        ch_lower = ch.lower()

        _console.print(ch.upper() if ch.isprintable() else "")

        if ch_lower in ("y", "\r", "\n", ""):
            return True
        elif ch_lower == "a":
            if action == "bash" and detail.strip():
                _console.print(_remember_bash_approval(detail))
            else:
                _approved_actions.add(action)
                _console.print(f"[dim]  → {label} авто-одобрены на эту сессию[/]\n")
            return True
        elif ch_lower in ("n", "\x1b", "\x03"):
            _console.print("[dim]  → отклонено[/]\n")
            return False
        else:
            _console.print(f"  [dim]нажми Y, A или N[/]")


async def ask_user_question(
    question: str, options: list[dict], recommended: str | None = None
) -> str:
    """Блокирует до реального ответа пользователя — либо выбранный вариант
    из options (каждый — {"label": ..., "description": ...}), либо свободный
    текст. recommended — label варианта, который стоит выделить как
    рекомендуемый (может быть None). В отличие от ask_permission, здесь нет
    авто-approve: вопрос всегда требует настоящего ответа.

    Если ответ — стоп-слово (см. _is_stop_intent), поднимаем
    asyncio.CancelledError вместо того, чтобы вернуть его как обычный текст
    — тот же тип исключения, что уже кидает Ctrl+C (ui/app.py:_interrupt ->
    task.cancel()), так что весь СУЩЕСТВУЮЩИЙ путь обработки отмены
    (cli.py:602 "stopped = True", mcp_agent/pipeline.py:401 предупреждение
    про незавершённые правки перед re-raise) подхватывает это без единой
    новой строчки там — раз это BaseException, а не Exception, ничто по
    пути (ask_user_tool.py:_ToolErrorGuardMiddleware в частности) не
    перехватывает и не гасит его как "тул упал"."""
    global _pending_prompts

    async with _permission_lock:
        _pending_prompts += 1
        try:
            if _app is not None:
                answer = await _app.show_ask_user_dialog(question, options, recommended)
                if answer is None:
                    return "(user dismissed the question without answering)"
                if _is_stop_intent(answer):
                    raise asyncio.CancelledError("user asked to stop via ask_user answer")
                return answer

            # Fallback: no TUI — plain terminal prompt (e.g. mcp_agent/run_cli.py).
            loop = asyncio.get_event_loop()
            _console.print()
            _console.print(f"[bold yellow]  ❓ {question}[/]")
            for i, opt in enumerate(options, 1):
                label = opt.get("label", "")
                desc = opt.get("description", "")
                star = " [green](рекомендуется)[/]" if label == recommended else ""
                _console.print(f"  [cyan][{i}][/] [bold]{label}[/]{star}")
                if desc:
                    _console.print(f"      [dim]{desc}[/]")
            _console.print(f"  [dim]Номер варианта или свой ответ:[/] ", end="")
            sys.stdout.flush()
            line = (await loop.run_in_executor(None, sys.stdin.readline)).strip()
            if _is_stop_intent(line):
                raise asyncio.CancelledError("user asked to stop via ask_user answer")
            if line.isdigit() and 1 <= int(line) <= len(options):
                return options[int(line) - 1].get("label", "")
            return line or "(user gave no answer)"
        finally:
            _pending_prompts -= 1


async def ask_permission(action: str, detail: str) -> bool:
    global _always_approve, _pending_prompts

    # Глобальный выключатель (/settings, "спрашивать разрешения") — ЕДИНАЯ
    # точка входа для обоих путей подтверждения (HITL-интеррапт графа через
    # _ask_decisions, и _execute_leaked_tool_call для утёкших вызовов), так
    # что проверка здесь отключает диалоги полностью, а не только один из
    # путей. Ничего не логируем/не печатаем при выключенном режиме — если
    # человек явно попросил не спрашивать, повторяющаяся строка "разрешение
    # не требуется" на каждый bash была бы тем же шумом, от которого он
    # и пытался избавиться.
    import settings
    if not settings.get("ask_permissions"):
        return True

    def _is_auto_approved() -> bool:
        if _always_approve or action in _approved_actions:
            return True
        # bash approvals are per-command-name (e.g. "bash:pwd") ТОЛЬКО для
        # простой, нецепочечной команды — см. _bash_auto_approve_key.
        # Составная (cd X && Y, ls | grep) матчится по ПОЛНОЙ строке, иначе
        # одобрение "cd" один раз тихо одобрило бы любое "cd X && <что угодно>".
        if action == "bash" and detail.strip():
            cmd_name = _bash_auto_approve_key(detail)
            if cmd_name is not None:
                return f"bash:{cmd_name}" in _approved_actions
            return detail.strip() in _approved_actions
        return False

    if _is_auto_approved():
        short = detail.splitlines()[0][:70]
        _console.print(f"[dim]  ✓ авто: {short}[/]")
        return True

    label = _ACTION_LABELS.get(action, action)

    # Диалог/raw-mode stdin ниже — не потокобезопасно и не гонко-безопасно
    # для нескольких параллельных tool_calls, поэтому сериализуем его тут.
    async with _permission_lock:
        # Другой параллельный tool_call мог уже авто-одобрить это действие,
        # пока мы ждали лок — перепроверяем перед показом диалога.
        if _is_auto_approved():
            short = detail.splitlines()[0][:70]
            _console.print(f"[dim]  ✓ авто: {short}[/]")
            return True

        _pending_prompts += 1
        try:
            if _app is not None:
                result = await _app.show_permission_dialog(action, detail)
                if result == "a":
                    if action == "bash" and detail.strip():
                        _console.print(_remember_bash_approval(detail))
                    else:
                        _approved_actions.add(action)
                        _console.print(f"[dim]  → {label} авто-одобрены на эту сессию[/]\n")
                return result in ("y", "a")

            # Fallback: no TUI active — use async executor
            loop = asyncio.get_event_loop()

            while True:
                _console.print(
                    f"\n"
                    f"  [bold green][Y][/] [green]да[/]   "
                    f"[bold blue][A][/] [blue]да, всегда[/]   "
                    f"[bold red][N][/] [red]нет[/]  › ",
                    end="",
                )
                sys.stdout.flush()

                ch = await loop.run_in_executor(None, _read_key_sync)
                ch_lower = ch.lower()

                _console.print(ch.upper() if ch.isprintable() else "")

                if ch_lower in ("y", "\r", "\n", ""):
                    return True
                elif ch_lower == "a":
                    if action == "bash" and detail.strip():
                        _console.print(_remember_bash_approval(detail))
                    else:
                        _approved_actions.add(action)
                        _console.print(f"[dim]  → {label} авто-одобрены на эту сессию[/]\n")
                    return True
                elif ch_lower in ("n", "\x1b", "\x03"):
                    _console.print("[dim]  → отклонено[/]\n")
                    return False
                else:
                    _console.print(f"  [dim]нажми Y, A или N[/]")
        finally:
            _pending_prompts -= 1
