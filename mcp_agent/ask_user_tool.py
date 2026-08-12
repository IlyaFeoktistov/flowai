"""
ask_user — интерактивный тул, которым модель задаёт пользователю уточняющий
вопрос вместо того, чтобы решать самой или отфутболивать выбор обычным
текстом. Не MCP-тул (нужен прямой доступ к TUI через tools/confirm.py),
поэтому определён здесь, а не в mcp_agent/servers/*.

Сюда же — всё, что напрямую обслуживает permission-диалог и HITL-цикл:
_action_and_detail/_ask_decisions (пересказ MCP-тула в понятный
ask_permission() action/detail) и две guard-мидлвари
(_ToolErrorGuardMiddleware, _AskUserGuardMiddleware), не позволяющие сбою
одного тула уронить весь ход и не позволяющие модели действовать, не
дождавшись ответа на свой же ask_user.
"""
import os
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from tools.confirm import ask_permission, ask_user_question

# Пересказ MCP-тула в (action, detail) для tools/confirm.py:ask_permission()
# — тот же словарь _ACTION_LABELS и та же по-командная auto-approve
# гранулярность для bash, что и в текущем приложении.
_ACTION_MAP = {
    "bash_exec": "bash",
    "write_file": "write_file",
    "edit_file": "patch_file",
    "create_directory": "write_file",
    "move_file": "write_file",
    "update_memory": "write_file",
}


def _action_and_detail(name: str, args: dict) -> tuple[str, str]:
    if name in ("bash_exec", "bash_exec_bg"):
        # Тот же action "bash", что и у синхронного bash_exec — одна и та же
        # команда одинаково опасна независимо от того, ждём мы её результат
        # синхронно или запускаем в фоне; auto-approve по первому слову
        # команды должен работать одинаково для обеих форм.
        return "bash", str(args.get("command", ""))
    if name.startswith("git_"):
        # У каждой git-операции — свой action (== имя MCP-тула), а НЕ общий
        # "bash" с git-командой в тексте. Раньше git_checkout/git_commit/
        # git_add/git_reset/git_create_branch все шли через action="bash" с
        # авто-approve по первому слову detail'а — а первое слово там всегда
        # "git", так что "да, всегда" на git_add тихо разрешал бы и
        # git_checkout, и git_reset — операции с совершенно разным уровнем
        # риска (checkout меняет рабочее дерево реального репозитория
        # пользователя, add почти безвреден). Раздельные action-ключи дают
        # раздельный auto-approve на каждую конкретную операцию.
        return name, str(args)
    action = _ACTION_MAP.get(name, name)
    return action, str(args)


async def _ask_decisions(hitl_request: dict) -> dict:
    decisions = []
    for action_req in hitl_request["action_requests"]:
        action, detail = _action_and_detail(action_req["name"], action_req["args"])
        allowed = await ask_permission(action, detail)
        decisions.append({"type": "approve" if allowed else "reject"})
    return {"decisions": decisions}


class AskUserOption(BaseModel):
    label: str = Field(description="Short option name, e.g. 'React' — this exact text is returned as the answer if the user picks this option.")
    description: str = Field(default="", description="One short sentence: this option's actual tradeoff/rationale, specific to the current task — not a generic definition of what the option is.")


@tool
async def ask_user(
    question: str,
    options: list[AskUserOption | str] | None = None,
    recommended: str | None = None,
) -> str:
    """Ask the user a clarifying question when there's a genuine judgment
    call to make (see the 'judgment call' rule above) — do NOT silently pick
    an option yourself. Put the concrete options you're weighing in `options`
    with a real one-sentence explanation each (the UI always lets the user
    type a free-form custom answer too, so don't add an 'other'/'something
    else' entry yourself — leave `options` empty only for a fully open-ended
    question). If you have an actual recommendation, put that option's exact
    `label` in `recommended` — the UI highlights it; leave `recommended`
    empty if you genuinely have no preference, don't pick one just to fill
    the field. This blocks and returns the user's actual answer as plain
    text. Call it ALONE — no other tool calls in the same turn — and wait
    for its result before doing anything else; never guess ahead or start
    implementing while this is pending."""
    # Живой прогон: модель не всегда следует схеме {label, description} и
    # иногда подставляет options как голые строки (как раньше, до этой
    # правки) — `AskUserOption | str` в аннотации принимает оба варианта
    # без ошибки валидации, здесь просто нормализуем к одному виду вместо
    # того, чтобы ронять весь вызов из-за одного "неправильно" оформленного
    # варианта.
    option_dicts = [
        o.model_dump() if isinstance(o, AskUserOption) else {"label": str(o), "description": ""}
        for o in (options or [])
    ]
    return await ask_user_question(question, option_dicts, recommended)


@tool
async def mark_plan_step_current(step_number: int) -> str:
    """Tell the UI which plan step you're STARTING work on right now —
    step_number is 1-based, matching the approved plan's own numbering
    (the "1. ...", "2. ..." list you were given). Call this ONCE, right
    before you begin locating/reading for that step, for every step in
    order — including the very first one. Without it, the user only finds
    out which steps are done at the very end of the whole round, with no
    indication of what's currently in progress. This is a pure status
    ping with no side effect on the codebase — it never needs approval and
    can't fail in a way that matters, so don't hesitate to call it.
    Not a substitute for your own final report — still describe what
    changed for every step when you're done. Coder-stage only.
    Not MCP-backed — like ask_user, it needs the TUI's plan checklist
    panel directly (ui/app.py:set_plan_current), which a subprocess
    tool has no handle to; the actual UI update happens off the
    tool_start event itself (see ui/stream.py), so this function body
    barely matters — it only needs to exist so the model has something
    to call."""
    return f"Marked step {step_number} as current."


def _sibling_tool_names(state: Any, tool_call_id: str) -> set[str]:
    messages = state["messages"] if isinstance(state, dict) else getattr(state, "messages", [])
    for m in reversed(messages):
        if isinstance(m, AIMessage) and any(tc.get("id") == tool_call_id for tc in (m.tool_calls or [])):
            return {tc["name"] for tc in (m.tool_calls or [])}
    return set()


class _ToolErrorGuardMiddleware(AgentMiddleware):
    """create_agent's built-in ToolNode only converts ToolInvocationError
    (bad args caught by schema validation) into a ToolMessage — every other
    tool-execution exception is re-raised by its default handle_tool_errors
    (see langgraph.prebuilt.tool_node._default_handle_tool_errors) and blows
    up the entire stream_chat call, skipping the retry/guidance logic below
    that expects tool failures to show up as ToolMessages (see the
    `failed_writes` handling). Живой прогон: edit_file получил `edits` как
    голую строку "oldText, newText" вместо массива — _normalize_edit_file_args
    намеренно не трогает не-JSON значения (пусть падает с исходной ошибкой),
    но эта "исходная ошибка" оказалась обычным исключением, а не
    ToolInvocationError/ToolException, и снесла весь ход целиком вместо того
    чтобы стать retryable ToolMessage. Ловим здесь, до ToolNode-обработчика —
    так любой сбой тула (не только этот) становится обычным провалом раунда."""

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            return ToolMessage(
                content=f"Error running `{request.tool_call['name']}`: {e}",
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )


class _AskUserGuardMiddleware(AgentMiddleware):
    """ask_user должен быть ЕДИНСТВЕННЫМ tool_call в своём AIMessage —
    иначе модель успевает выполнить другие действия, так и не дождавшись
    ответа на СВОЙ ЖЕ вопрос (живой баг: модель спросила "какой из способов
    вам больше подходит?" и в том же ходе уже вызвала другой тул с одним из
    вариантов, не дожидаясь реального ответа). Системный промпт просит
    модель не смешивать их, но текстовая инструкция — не гарантия;
    подстраховываемся на уровне выполнения тулов, а не только промптом."""

    async def awrap_tool_call(self, request, handler):
        if request.tool_call["name"] == "ask_user":
            return await handler(request)
        if "ask_user" in _sibling_tool_names(request.state, request.tool_call["id"]):
            return ToolMessage(
                content=(
                    "Skipped: called in the same turn as ask_user, before the "
                    "user answered. Wait for ask_user's result (the user's "
                    "actual answer), then call this again if it's still needed."
                ),
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return await handler(request)


class _AskUserFinalizeMiddleware(AgentMiddleware):
    """Planner-only backstop. Live incident: qwen3-coder:30b, as Planner,
    called ask_user 8 times over ~11 minutes with near-duplicate "готов ли
    я...?" confirmations — the user answered "Да"/"Продолжить"/"Выполнить"
    every single time, but the model kept reading a few more lines and
    asking again instead of finalizing, until the user killed the run by
    hand. planner_verdict (mcp_agent/stages/planner.py) already treats "an
    ask_user call happened this round" as done, and
    _planner_system_prompt (mcp_agent/prompts.py) already tells the model
    to restate its final plan right after ask_user returns and stop — but
    both only take effect once the underlying agent graph run actually
    ends, and nothing stopped the model from calling MORE tools (another
    ask_user, another read_file_range) inside that SAME still-running
    graph execution before it ever does.

    Mechanical backstop instead of relying on the prompt wording alone:
    once one ask_user ToolMessage exists anywhere in the conversation,
    reject every further tool call (ask_user or anything else) with an
    instruction to finalize as plain text right now — forces the round to
    end immediately after the first confirmation, handing off to Coder.

    MUST only count a SUCCESSFUL ask_user call as "the user answered" —
    live bug: the model's first ask_user call had `options` as a bare dict
    instead of a list, which create_agent's ToolNode turns into a
    ToolInvocationError -> ToolMessage(name="ask_user", status="error")
    (see _ToolErrorGuardMiddleware above) WITHOUT ever reaching the user.
    This check used to match on name alone, so it saw that error message,
    believed the user had already confirmed a plan they never saw, and
    blocked the model's very next (correctly-formed) ask_user call from
    ever running — the model then dutifully "restated the confirmed plan"
    that had, in fact, never been shown to anyone. Filtering to
    status != "error" makes only a real, delivered answer count."""

    async def awrap_tool_call(self, request, handler):
        messages = request.state["messages"] if isinstance(request.state, dict) else getattr(request.state, "messages", [])
        if any(
            isinstance(m, ToolMessage) and m.name == "ask_user" and getattr(m, "status", "success") != "error"
            for m in messages
        ):
            return ToolMessage(
                content=(
                    "Skipped: you already called ask_user earlier in this round "
                    "and got the user's answer. Do not call ask_user (or any "
                    "other tool) again — state your FINAL numbered plan as "
                    "plain text right now, the same plan you already "
                    "confirmed, and stop."
                ),
                name=request.tool_call["name"],
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return await handler(request)


# Аргумент(ы)-путь у каждого filesystem-ПИШУЩЕГО тула (читающие — search_files,
# read_file и т.п. — сюда НЕ входят: filesystem-сервер теперь разрешён до "/",
# см. config.py:build_mcp_connections, и по прямому решению пользователя чтение
# ВЕЗДЕ идёт без единого approval-вопроса — риск там намного ниже, чем у записи).
# move_file/copy_lines — по два пути каждый (откуда/куда).
_WRITE_PATH_ARG_NAMES = {
    "write_file": ["path"], "edit_file": ["path"], "create_directory": ["path"],
    "move_file": ["source", "destination"],
    "replace_lines": ["path"], "insert_lines": ["path"],
    "copy_lines": ["source_path", "dest_path"],
    "delete_path": ["path"],
}


class _OutOfProjectWriteApprovalMiddleware(AgentMiddleware):
    """filesystem MCP-сервер (config.py:build_mcp_connections) теперь
    разрешён до "/" — весь диск для ЧТЕНИЯ, по прямому решению пользователя
    ("пускай везде лазит... доступ до корня /"). Но запись вне repo_path —
    отдельный, гораздо более рискованный случай (реально меняет файлы за
    пределами проекта, который пайплайн вообще не собирался трогать) — она
    остаётся под approval, тем же механизмом, что уже есть у bash_exec/
    write_file внутри проекта (tools/confirm.py:ask_permission), просто
    условие срабатывания — путь ВНЕ repo_path, а не имя тула статично.

    Живой инцидент, который стал поводом для всей этой развилки: model
    попыталась выйти в соседний проект, получила жёсткий "Access denied" (в
    старой, узкой конфигурации сервера) и 30 минут бесцельно копалась в
    НЕПРАВИЛЬНОМ репозитории вместо честного "не могу"/вопроса
    пользователю. Отказ approval здесь — обычный ToolMessage с понятной
    причиной, не exception — модель должна его увидеть и остановиться, не
    считать сбоем тула и ретраить тот же путь."""

    def __init__(self, repo_path: str):
        self._repo_path = os.path.realpath(repo_path)

    def _is_outside(self, raw_path: str) -> bool:
        if not raw_path:
            return False
        candidate = raw_path if os.path.isabs(raw_path) else os.path.join(self._repo_path, raw_path)
        resolved = os.path.realpath(candidate)
        return os.path.commonpath([resolved, self._repo_path]) != self._repo_path

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        arg_names = _WRITE_PATH_ARG_NAMES.get(name)
        if not arg_names:
            return await handler(request)

        args = request.tool_call.get("args") or {}
        outside = [
            args[a] for a in arg_names
            if isinstance(args.get(a), str) and self._is_outside(args[a])
        ]
        if not outside:
            return await handler(request)

        allowed = await ask_permission("write_outside_project", f"{name}: {', '.join(outside)}")
        if not allowed:
            return ToolMessage(
                content=(
                    f"Denied: the user did not approve writing outside the "
                    f"current project ({', '.join(outside)}). Do not retry "
                    "this path — report to the user that it's out of reach "
                    "without their explicit approval, or ask them directly "
                    "instead of guessing another path."
                ),
                name=name,
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return await handler(request)
