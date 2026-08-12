# ui/stream.py

import asyncio
import io
import json
import random
import re
import time
from datetime import timedelta
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.markup import escape as _escape_markup
from ui.console import console, safe_write, set_title, fmt_elapsed

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _render_markup(markup: str) -> str:
    """Renders Rich markup to raw ANSI text (no trailing newline), via a
    throwaway Console/StringIO — needed to overwrite ONE already-printed
    logical line in ui.app._OutputControl._lines in place (the tool-call
    status dot: hollow+blinking while pending -> solid white once tool_end
    arrives, see on_event's "tool_start"/"tool_end" below). A brand new
    Console per call (not the shared module-level `console`) avoids any
    risk of interfering with its real file=proxy mid-turn."""
    buf = io.StringIO()
    Console(file=buf, force_terminal=True, highlight=False, width=200).print(markup, end="")
    return buf.getvalue()

# Rich's own color/style SGR codes (same pattern as ui/app.py:_ANSI_ESCAPE,
# just the "m"-only subset — Rich's Markdown rendering never emits cursor-
# movement codes, only color/style) — used by _rerender_markdown to tell a
# genuinely blank line apart from one that merely carries a color reset.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Разбиваем на озвучиваемые куски по ходу стрима (см.
# StreamDisplay._feed_speech) — делим по пробелу/переносу ПОСЛЕ терминатора
# или запятой, а не по самому знаку: пока после "." нет пробела, модель
# может ещё продолжать то же предложение следующим чанком ("3." -> "3.14").
# Запятая — тоже граница (не только конец предложения): ждать полную фразу
# до точки означало заметную задержку первого звука на длинных предложениях.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…,])\s+")

# Если пунктуации всё не видно (длинное предложение без единой запятой),
# принудительно отрезаем кусок через ~7 слов — иначе первый звук откладывался
# бы до конца всей фразы. Кусок при этом может состоять и из 1-2 слов
# (короткие ответы типа "Да." звучат нормально сами по себе — специально не
# склеиваем их с соседями).
_MAX_WORDS_BEFORE_FLUSH = 7


def _split_ready_sentences(buf: str) -> tuple[list[str], str]:
    """Возвращает (готовые куски текста, недописанный остаток)."""
    parts = _SENT_SPLIT_RE.split(buf)
    if len(parts) > 1:
        sentences, remainder = parts[:-1], parts[-1]
    else:
        sentences, remainder = [], buf

    words = remainder.split(" ")
    while len(words) > _MAX_WORDS_BEFORE_FLUSH:
        sentences.append(" ".join(words[:_MAX_WORDS_BEFORE_FLUSH]))
        words = words[_MAX_WORDS_BEFORE_FLUSH:]
    remainder = " ".join(words)

    return sentences, remainder

# Один случайный вариант на фазу вместо фиксированного "думаю"/"обрабатываю"
# — просто для настроения, чисто косметика. Randomизация — на КАЖДЫЙ вход в
# фазу (не на каждый тик спиннера, это было бы мельтешение), см. места
# вызова ниже.
_THINKING_PHRASES = (
    "думаю", "шевелю извилинами", "включаю мозги", "подбираю слова",
    "варю мысли", "советуюсь с нейронами", "гружу идеи",
)
_TOOL_RUNNING_PHRASES = (
    "выполняю тулы", "копаюсь в файлах", "дёргаю рычаги",
    "колдую с инструментами", "жму на кнопки", "кручу гайки",
)
_PROCESSING_PHRASES = (
    "обрабатываю", "перевариваю результат", "раскладываю по полочкам",
    "сверяю показания", "анализирую улов", "изучаю добычу",
)
_GENERATING_PHRASES = (
    "печатаю", "строчу ответ", "накидываю мысль", "выдаю мысль", "пишу",
)
# self_heal.py:_semantic_check — судейский вызов модели ПОСЛЕ того, как
# видимый ответ уже дописан (answer_end), перед решением "готово" vs "ещё
# попытка". Живой баг (репорт пользователя): на медленном локальном железе
# этот вызов сам по себе занимает МИНУТЫ (свой промпт из TASK+TOOL RESULTS+
# ANSWER, та же CPU-тяжёлая модель) — без отдельной фазы футер просто
# показывал последнюю фразу от answer_end/tool_end и молчал, выглядело как
# зависание "уже всё написал, а крутится неизвестно почему".
_VERIFYING_PHRASES = (
    "проверяю ответ", "сверяю с задачей", "перечитываю сам себя",
)

# mcp_agent/pipeline.py эмитит "stage_changed" при переходе между ролями
# (mcp_agent/roles.py) — до этого пользователь видел только generic
# "[MCP-AGENT]" в DEBUG-выводе и не мог понять, какая именно роль сейчас
# работает (живая жалоба: "я везде вижу MCP_AGENT, хочу видеть конкретно
# какой агент — анализатор или кодер"). Ключи — то, что реально шлёт
# pipeline.py (analyzer/planner/coder/verifier/casual), значение — то, что
# видит пользователь в футере и разовой строке перехода.
_STAGE_LABELS = {
    "analyzer": "🔎 Анализатор",
    "planner": "📋 Планировщик",
    "coder": "⌨️ Кодер",
    "verifier": "✅ Верификатор",
    "casual": "💬 Ответ",
    "quick_fix": "⚡ Быстрая правка",
}


# Держать в синхроне с mcp_agent/self_heal.py:_WRITE_TOOL_NAMES вручную —
# та же ситуация, что и mcp_agent/stage_runner.py:_stage_digest (см. его
# комментарий): импортировать оттуда сюда means ui.stream -> mcp_agent.self_heal
# -> ui.console, лишний риск порядка импорта ради одной короткой константы.
_FILE_EDIT_TOOL_NAMES = ("write_file", "edit_file", "replace_lines", "move_file", "copy_lines")

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _format_file_edit_result(name: str, args: dict, result: str) -> tuple[str, list[str]] | None:
    """Compact "Update(path) · +N -M" summary + line-numbered diff body for
    file-edit tool results — replace_lines/fs_extra_server.py's own diff
    (see _unified_diff_at there) and the official filesystem MCP server's
    edit_file both return a standard unified diff with real file line
    numbers in the "@@ -a,b +c,d @@" header, which is exactly what this
    parses. Returns None if `result` isn't diff-shaped (no "@@" hunk header
    at all — e.g. write_file's plain "Successfully wrote to ..." with
    nothing to diff, or a tool error) so the caller falls back to the
    generic tool_end rendering instead of showing an empty/wrong body.

    Live bug motivation (user report): the generic rendering showed the raw
    diff text with only +/- text-color coding and no indication of which
    real file lines changed — reading it required counting from the "@@"
    header by hand. This mirrors Claude Code's own compact tool-result view
    (header + "Added/removed N lines" + numbered body) instead."""
    lines = result.splitlines()
    if not any(_HUNK_HEADER_RE.match(ln) for ln in lines):
        return None

    path = args.get("path") or args.get("destination") or args.get("target") or "?"
    verb = {"write_file": "Write", "move_file": "Move", "copy_lines": "Copy"}.get(name, "Update")
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    header = f"{verb}({path})" + (f" · {' '.join(parts)}" if parts else "")

    body: list[str] = []
    old_no = new_no = 0
    for ln in lines:
        m = _HUNK_HEADER_RE.match(ln)
        if m:
            old_no, new_no = int(m.group(1)), int(m.group(2))
            continue
        if ln.startswith(("+++ ", "--- ")):
            continue
        style = _diff_line_style(ln) or "bright_black"
        content = ln[1:] if ln[:1] in ("+", "-", " ") else ln
        if ln.startswith("+"):
            gutter, new_no = f"{new_no:>5}", new_no + 1
        elif ln.startswith("-"):
            gutter, old_no = f"{old_no:>5}", old_no + 1
        else:
            gutter = f"{new_no:>5}"
            old_no, new_no = old_no + 1, new_no + 1
        body.append(f"[{style}]  {gutter} {_escape_markup(content)}[/]")
    return header, body


def _diff_line_style(line: str) -> str | None:
    """Style for one line of a unified diff (replace_lines/git_diff* already
    return this format, with real file line numbers in the "@@" header —
    see fs_extra_server.py:_unified_diff_at). Same red/green/cyan convention
    as `git diff`, so an edit reads the same way here as it would in a
    terminal git diff. None means "not a diff line, use the plain style"."""
    if line.startswith(("+++ ", "--- ")):
        return "bold"
    if line.startswith("@@"):
        return "bold cyan"
    if line.startswith("+"):
        return "green"
    if line.startswith("-"):
        return "red"
    return None


def _format_duration(seconds: float) -> str:
    """Format duration as Xm Ys or Xs, without fractions."""
    td = timedelta(seconds=int(seconds))
    mins = td.seconds // 60
    secs = td.seconds % 60
    days = td.days
    if days:
        hours = days * 24 + td.seconds // 3600
        rest_secs = (td.seconds % 3600) // 60
        return f"{hours}ч {rest_secs}м {secs}с"
    elif mins:
        return f"{mins}м {secs}с"
    else:
        return f"{secs}с"


def _ai_header(tok: int, elapsed: float, label: str = "") -> str:
    """'[label ]N tok · X.Xs' in raw ANSI — needed for re-render without
    Rich. No "AI ›" prefix here — that marker already exists inline before
    the streamed text itself (see answer_start/handle_chunk); repeating it
    on the footer/stats line just added noise. Used both for the live
    per-tick footer (label = spinner frame + current phase phrase, see
    StreamDisplay._footer_loop) and the final static render once the turn
    is done (label='' — nothing is "in progress" anymore, so no verb)."""
    prefix = f"{label} · " if label else ""
    return f" \033[2m{prefix}{tok} tok · {_format_duration(elapsed)}\033[0m"


class StreamDisplay:
    """Manages display of a single AI response: spinner, thinking, tokens, stats."""

    def __init__(self, session_stats: dict, app=None) -> None:
        self._stats = session_stats
        self._app = app  # FlowAIApp instance or None (fallback to terminal)
        self.pending_stats: dict = {}
        self.full_response: str = ""
        self._first = True
        self._thinking_open = False
        self._tok_approx = 0
        self._t_wall: float = 0.0
        self._footer_task: asyncio.Task | None = None
        self._phase_label: str = ""
        self._current_stage: str = ""  # см. _STAGE_LABELS, "" вне нового пайплайна (легаси-агент/casual ещё не начался)
        self._content_lines: int = 0
        # Статус-точка тула (см. on_event "tool_start"/"tool_end"): индекс
        # строки в _app._output._lines, которую нужно перезаписать при
        # завершении тула (только app-режим — в legacy-терминале переписать
        # уже проскроллившую строку небезопасно, см. комментарий там же).
        # FIFO, не одиночное значение — модель иногда зовёт несколько тулов
        # одним сообщением (несколько tool_start подряд до их tool_end).
        # 4-й элемент — задача мигания этой конкретной точки (см. _blink_tool_dot).
        self._pending_tool_calls: list[tuple[str, dict, int | None, asyncio.Task | None]] = []
        self._speech = None            # SpeechStreamer, создаётся лениво при первом voice_mode-ходе
        self._speech_buf: str = ""
        self._speech_notified: bool = False
        self._last_written_was_newline = True  # см. _collapse_and_write

    def start(self, t_wall: float) -> None:
        self._t_wall = t_wall
        self._first = True
        self._thinking_open = False
        self._tok_approx = 0
        self._content_lines = 0
        self.full_response = ""
        self._last_written_was_newline = True
        self.pending_stats = {}
        self._speech_buf = ""
        self._speech_notified = False
        self._cancel_pending_tool_blinks()
        self._phase_label = f"{random.choice(_THINKING_PHRASES)}..."
        self._current_stage = ""  # сбрасывается на каждый ход — новый ход сам решит, стадийный он или легаси
        self._footer_task = asyncio.create_task(self._footer_loop())

    _TOOL_DOT_BLINK_S = 1.0

    async def _blink_tool_dot(self, line_idx: int, name: str) -> None:
        """Toggles ONE pending tool's status dot between gray and white
        every _TOOL_DOT_BLINK_S seconds. Real ANSI blink (SGR 5) was tried
        first and dropped — live bug (user report): it didn't actually
        blink in their terminal (font/terminal-dependent, and its rate
        isn't controllable) — this drives the same effect explicitly via a
        redraw loop instead, same pattern as _footer_loop's own spinner
        tick, just targeting one specific historical line instead of the
        fixed footer row. App-mode only — see tool_start's own comment on
        why legacy-terminal mode can't do this safely."""
        on = False
        try:
            while True:
                await asyncio.sleep(self._TOOL_DOT_BLINK_S)
                if self._app is None or not (0 <= line_idx < len(self._app._output._lines)):
                    return
                on = not on
                style = "bold white" if on else "bright_black"
                self._app._output._lines[line_idx] = _render_markup(f"[{style}]  ●[/][bright_black] {name}[/]")
                self._app.invalidate()
        except asyncio.CancelledError:
            pass

    def _feed_speech(self, text: str) -> None:
        """Кормит потоковый TTS (см. ui/audio.SpeechStreamer) готовыми
        предложениями по ходу генерации — вызывается из answer_chunk/
        handle_chunk, а НЕ постфактум по всему ответу (см. cli.py, там
        раньше был единственный блокирующий speak() после конца стрима)."""
        import settings
        if not settings.get("voice_mode"):
            return
        if self._speech is None:
            from ui.audio import SpeechStreamer
            self._speech = SpeechStreamer()
        if not self._speech_notified:
            console.print("[dim]  🔊 озвучиваю по ходу...[/]")
            self._speech_notified = True
        self._speech_buf += text
        sentences, remainder = _split_ready_sentences(self._speech_buf)
        self._speech_buf = remainder
        for s in sentences:
            self._speech.feed(s)

    def _flush_speech_round(self) -> None:
        """Конец одного раунда ответа (answer_end/handle_chunk-путь без
        событий) — отдаёт в TTS то, что осталось в буфере без конечной
        пунктуации (иначе последнее предложение ответа никогда не
        озвучилось бы, ведь _split_ready_sentences ждёт пробел ПОСЛЕ
        точки)."""
        if self._speech is None:
            return
        if self._speech_buf.strip():
            self._speech.feed(self._speech_buf)
            self._speech_buf = ""
        self._speech.finish()

    def stop_speech(self) -> None:
        """Отмена хода (Ctrl+C) — выкидывает всё ещё не озвученное."""
        if self._speech is not None:
            self._speech.stop()
        self._speech_buf = ""

    async def _footer_loop(self) -> None:
        """Single always-on footer ticker for the WHOLE turn — one task, not
        a separate spinner-task-per-phase plus a separately-started token
        counter. Live bug: those two used to run concurrently (a spinner
        task from tool_start alongside the counter task started at the
        first answer_start) and fought over the same footer line, each
        overwriting the other's set_stats() call on its own cadence — the
        footer visibly flickered between "⠏ дёргаю рычаги... 31с | 9с" and
        "AI › 474 tok · 1м 48с". A phase change now just updates
        self._phase_label — there's nothing left to race, and the token
        count + elapsed time stay visible through every phase (thinking,
        running a tool, processing its result, generating), same unified
        format throughout instead of two different-looking displays."""
        i = 0
        try:
            while True:
                frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
                elapsed = time.monotonic() - self._t_wall
                stage_prefix = f"{self._current_stage} · " if self._current_stage else ""
                header = _ai_header(self._tok_approx, elapsed, f"{frame} {stage_prefix}{self._phase_label}")
                if self._app is not None:
                    self._app.set_stats(header)
                else:
                    try:
                        import os
                        rows = os.get_terminal_size().lines
                        safe_write(f"\033[s\033[{rows};1H\r\033[K{header}\033[u")
                    except Exception:
                        break
                await asyncio.sleep(0.1)
                i += 1
        except asyncio.CancelledError:
            if self._app is not None:
                self._app.set_stats("")
            else:
                safe_write("\r\033[K")

    async def on_event(self, event: dict) -> None:
        t = event.get("type")

        # ── THINKING ─────────────────────────────────────
        if t == "thinking_start":
            self._thinking_open = True
            safe_write("\n")
            console.print("[dim]  ╭─ размышляю ───────────────────────────────────────[/]")
            console.print("[dim]  │[/] ", end="")

        elif t == "thinking_chunk":
            text = event["text"].replace("\n", "\n\033[2m  │\033[0m ")
            safe_write(f"\033[2m{text}\033[0m")

        elif t == "thinking_end":
            self._thinking_open = False
            safe_write("\n")
            console.print("[dim]  ╰──────────────────────────────────────────────────[/]")
            console.print()

        elif t == "thinking_unavailable":
            model = event.get("model", "")
            if self._app is not None:
                self._app.set_stats("")
            else:
                safe_write("\r\033[K")
            console.print(f"[dim]  💭 {model} не поддерживает thinking, продолжаю без него[/]")

        elif t == "tools_unavailable":
            model = event.get("model", "")
            if self._app is not None:
                self._app.set_stats("")
            else:
                safe_write("\r\033[K")
            console.print(f"[dim]  🔧 {model} не поддерживает инструменты, продолжаю без них[/]")

        # ── STAGE (новый пайплайн: Router->Analyzer->Planner->Coder->
        # Verifier, mcp_agent/pipeline.py) — какая роль сейчас работает.
        # Разовая строка перехода + постоянная метка в футере (_footer_loop
        # читает self._current_stage на каждый тик), а не только один принт,
        # раз ход может провести много времени внутри одной роли.
        elif t == "stage_changed":
            stage = event.get("stage", "")
            label = _STAGE_LABELS.get(stage, stage)
            self._current_stage = label
            safe_write("\n")
            console.print(f"[bright_black]  {label}[/]")

        # ── PLAN (новый пайплайн) ─────────────────────────
        elif t == "plan":
            plan = event.get("plan", {})
            steps = plan.get("steps", [])
            tools = plan.get("tool_calls", [])
            console.print(f"[bright_black]  📝 План: {len(steps)} шагов, {len(tools)} инструментов[/]")
            for i, step in enumerate(steps):
                console.print(f"[dim]  {i+1}. {step}[/]")

        # ── PLAN CHECKLIST (mcp_agent/pipeline.py Planner->Coder) ─────────
        # В TUI (self._app) — постоянная панель над футером (ui/app.py:
        # set_plan_steps/mark_plan_step_done), сама себя перерисовывает.
        # Без TUI (run_cli.py, fallback-терминал) — некому держать
        # постоянную панель, печатаем список один раз и по одной строке на
        # каждый отмеченный шаг, тем же принципом, что "plan" выше.
        elif t == "plan_steps":
            steps = event.get("steps", [])
            if self._app is not None:
                self._app.set_plan_steps(steps)
            else:
                console.print(f"[bright_black]  📝 План ({len(steps)} шагов):[/]")
                for i, step in enumerate(steps):
                    console.print(f"[dim]  ☐ {i + 1}. {step}[/]")

        elif t == "plan_step_done":
            index = event.get("index", -1)
            if self._app is not None:
                self._app.mark_plan_step_done(index)
            else:
                console.print(f"[green]  ✅ Шаг {index + 1} выполнен[/]")

        # ── PLAN CURRENT STEP (mcp_agent/ask_user_tool.py:
        # mark_plan_step_current, Coder-only) — a pure status ping, not a
        # real tool call worth showing as its own "🔧 name {...}" box;
        # intercepted here (before the generic tool_start/tool_end below)
        # so it only ever updates the plan panel's current-step indicator.
        elif t == "tool_start" and event.get("name") == "mark_plan_step_current":
            step_number = (event.get("args") or {}).get("step_number")
            if isinstance(step_number, int):
                if self._app is not None:
                    self._app.set_plan_current(step_number - 1)
                else:
                    console.print(f"[yellow]  🟥 Шаг {step_number} — в работе[/]")

        elif t == "tool_end" and event.get("name") == "mark_plan_step_current":
            pass

        # ── TOOLING ──────────────────────────────────────
        elif t == "tool_start":
            name = event["name"]
            args = event.get("args", {})
            safe_write("\n")
            # Статус-точка: ● серая, пока тул выполняется, становится ●
            # сплошной белой на tool_end (см. там же и _render_markup выше).
            # U+25CF BLACK CIRCLE — не эмодзи-код "⏺" (U+23FA), который на
            # многих системах рисуется цветной emoji-иконкой вместо простой
            # монохромной точки (живой баг: пользователь увидел именно это).
            # В app-режиме строка адресуема (_output._lines) — точку можно
            # честно перезаписать позже. В legacy-терминале (обычный
            # scrolling stdout) исторические строки не переписать надёжно
            # (единственный слот \033[s/\033[u уже занят _footer_loop, а
            # относительный сдвиг курсора не посчитать точно, если между
            # tool_start и tool_end что-то ещё напечаталось, напр. approval-
            # промпт) — там точка остаётся серой все время выполнения.
            line_idx = None
            if self._app is not None:
                line_idx = len(self._app._output._lines) - 1
            console.print(f"[bright_black]  ● {name}[/]")
            blink_task = (
                asyncio.create_task(self._blink_tool_dot(line_idx, name))
                if line_idx is not None else None
            )
            self._pending_tool_calls.append((name, args, line_idx, blink_task))
            console.print(f"[dim]     {{")
            for key, val in args.items():
                val_str = str(val)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "…"
                console.print(f"[dim]       {key}: {val_str},")
            console.print(f"[dim]     }}")
            self._stats["tools_called"] += 1
            self._phase_label = f"{random.choice(_TOOL_RUNNING_PHRASES)}..."

        # ── ANSWER (реальный токенный стриминг) ───────────
        # answer_start/answer_chunk/answer_end покрывают И намерение-перед-
        # тулом ("одно предложение" из SYSTEM_PROMPT), И финальный ответ —
        # на момент answer_start ещё не известно, чем закончится это
        # сообщение (см. mcp_agent/agent.py:_stream_round). Один и тот же
        # "AI › " канал используется для обоих — answer_end просто решает
        # оформление ПОСЛЕ того, как текст уже показан.
        elif t == "answer_start":
            self._phase_label = f"{random.choice(_GENERATING_PHRASES)}..."
            self._first = False
            safe_write("\n\033[1;34m AI ›\033[0m ")
            if self._app is not None:
                self._app._output.mark_response_start()
            self._content_lines = 0
            self.full_response = ""
            self._last_written_was_newline = True

        elif t == "answer_chunk":
            text = event.get("text", "")
            if text:
                self.full_response += text
                self._feed_speech(text)
                self._tok_approx += 1
                self._content_lines += self._collapse_and_write(text).count("\n")
                if self._tok_approx % 5 == 0:
                    elapsed = time.monotonic() - self._t_wall
                    set_title(f"AI: ~{self._tok_approx} tok · {_format_duration(elapsed)}")

        elif t == "answer_end":
            self._flush_speech_round()
            if event.get("had_tool_calls"):
                # Это было намерение-перед-тулом, не финальный ответ — тул-блок
                # печатается сразу следом (tool_start), просто отделяем строкой.
                safe_write("\n")

        # ── VERIFYING (self_heal.py:_semantic_check, см. _VERIFYING_PHRASES
        # выше) ──────────────────────────────────────────
        elif t == "verifying_start":
            self._phase_label = f"{random.choice(_VERIFYING_PHRASES)}..."

        elif t == "verifying_end":
            pass  # следующее событие (retry-дайджест или done) сменит фразу само

        # ── MID-TURN STEER (mcp_agent/agent.py:_stream_round, mid_turn_queue)
        # — сообщение из очереди подхватилось между шагами графа, не после
        # конца всего хода (см. cli.py:_enqueue) — видимое подтверждение,
        # что оно реально дошло, а не тихо потерялось. ────────────────────
        elif t == "mid_turn_injected":
            text = event.get("text", "")
            safe_write("\n")
            console.print(f"[dim]  📥 учту по ходу: {_escape_markup(text)}[/]")
            console.print()

        elif t == "tool_end":
            name = event.get("name", "")
            result = event.get("result", "")
            # FIFO — см. _pending_tool_calls в __init__: tool_start/tool_end
            # для одного и того же вызова эмитятся строго по порядку, даже
            # если модель запросила несколько тулов одним сообщением.
            pending_args: dict = {}
            line_idx = None
            blink_task = None
            if self._pending_tool_calls:
                _, pending_args, line_idx, blink_task = self._pending_tool_calls.pop(0)
            if blink_task is not None:
                blink_task.cancel()
            # Bounds-check: /clear (or any other command) can run concurrently
            # with an in-flight turn (see cli.py's own comment on this) and
            # wipe/shrink _output._lines mid-tool-call — indexing a stale
            # line_idx from before that would otherwise crash the whole turn
            # on an IndexError over a purely cosmetic status dot.
            if line_idx is not None and self._app is not None and 0 <= line_idx < len(self._app._output._lines):
                self._app._output._lines[line_idx] = _render_markup(f"[bold white]  ●[/][bright_black] {name}[/]")
                self._app.invalidate()

            diffish = _format_file_edit_result(name, pending_args, result) if name in _FILE_EDIT_TOOL_NAMES and result else None
            if diffish:
                header, body = diffish
                console.print(f"[bright_black]     └ {_escape_markup(header)}[/]")
                show = body[:20]
                for ln in show:
                    console.print(ln)
                if len(body) > 20:
                    console.print(f"[bright_black]     … ещё {len(body) - 20} строк[/]")
            elif result:
                lines = result.splitlines()
                if len(lines) > 1:
                    # Live bug (user report): this branch is reached exactly
                    # when `diffish` above was None, i.e. the result is NOT
                    # diff-shaped (no "@@" hunk header) — a plain bash_exec
                    # output like `ls -la` still has lines starting with "-"
                    # (a regular file's "-rw-r--r--" permission string), and
                    # _diff_line_style would color those red as if removed,
                    # exactly like a git diff, even though this is plain
                    # command output with nothing to do with a diff at all.
                    show = lines[:20]
                    for ln in show:
                        console.print(f"[bright_black]     {_escape_markup(ln)}[/]")
                    if len(lines) > 20:
                        console.print(f"[bright_black]     … ещё {len(lines) - 20} строк[/]")
                else:
                    short = result if len(result) <= 200 else result[:200] + "…"
                    console.print(f"[bright_black]     ↳ {_escape_markup(short)}[/]")
            self._phase_label = f"{random.choice(_PROCESSING_PHRASES)}..."

        # ── MODEL SELECTED ───────────────────────────────
        elif t == "model_selected":
            model = event.get("model", "")
            stage = event.get("stage", "")
            stage_suffix = f" · {stage}" if stage else ""
            console.print(f"[bright_black]  ⚡ {model}{stage_suffix}[/]")

        # ── STATS ────────────────────────────────────────
        elif t == "stats":
            self.pending_stats = event

        # ── DONE ─────────────────────────────────────────
        # Nothing to do here — finish() (called from cli.py's finally block
        # right after the stream ends) cancels the footer task.

    def _collapse_and_write(self, text: str) -> str:
        """Writes `text` live, collapsing any run of 2+ consecutive
        newlines (including ones split across separate chunks/calls, via
        self._last_written_was_newline) down to exactly one — same fix as
        _rerender_markdown's blank-line stripping, just applied to the RAW
        live stream instead of only at the final re-render. Without this,
        a multi-paragraph reply looked double-spaced WHILE typing and only
        snapped tight once generation finished and the markdown re-render
        replaced it — visibly different before/after, which read as
        "why did it just shrink?" (live user report). Returns the text
        actually written (with any newlines removed accounted for), not
        the original — callers that count '\\n' for line-tracking should
        count THIS, not the input, or they overcount blank lines that
        never actually made it to the screen."""
        if not text:
            return ""
        text = re.sub(r"\n{2,}", "\n", text)
        if self._last_written_was_newline and text.startswith("\n"):
            text = text.lstrip("\n")
        if not text:
            return ""
        safe_write(text)
        self._last_written_was_newline = text.endswith("\n")
        return text

    async def handle_chunk(self, chunk: str) -> None:
        if self._first:
            # Write "AI ›" inline — mark is set on the same line so replace_from_mark
            # will re-include the marker when re-rendering with Rich markdown
            self._phase_label = f"{random.choice(_GENERATING_PHRASES)}..."
            safe_write("\n\033[1;34m AI ›\033[0m ")
            if self._app is not None:
                self._app._output.mark_response_start()
            self._content_lines = 0
            self._last_written_was_newline = True
            self._first = False

        self.full_response += chunk
        self._feed_speech(chunk)
        self._tok_approx += 1
        self._content_lines += self._collapse_and_write(chunk).count('\n')

        if self._tok_approx % 5 == 0:
            elapsed = time.monotonic() - self._t_wall
            set_title(f"AI: ~{self._tok_approx} tok · {_format_duration(elapsed)}")

    async def finalize_stream_text(self, text: str) -> None:
        """Called with stream_chat's final yielded value (see cli.py). Normal
        path: the same text was already shown live via answer_start/
        answer_chunk (see on_event above) — just trust it as ground truth for
        full_response, since it may add a warning prefix AFTER generation
        finished (see mcp_agent/agent.py:final_verdict) that was never part of
        the live token stream. Edge-case path (self._first still True — no
        messages, or the recursion-limit message): nothing was streamed live
        this turn at all, show it the old atomic way."""
        if self._first:
            await self.handle_chunk(text)
            self._flush_speech_round()
        else:
            self.full_response = text

    def _cancel_pending_tool_blinks(self) -> None:
        """Turn aborted mid-tool (Ctrl+C) — blink tasks for any tool that
        never got its tool_end would otherwise keep re-writing/invalidating
        _output._lines forever, on a line that's about to belong to a
        totally different turn. Called from finish() (this turn ending) and
        start() (defensive — in case a previous turn's cleanup was itself
        interrupted before reaching finish())."""
        for _, _, _, blink_task in self._pending_tool_calls:
            if blink_task is not None:
                blink_task.cancel()
        self._pending_tool_calls = []

    async def finish(self) -> None:
        self._cancel_pending_tool_blinks()
        if self._footer_task and not self._footer_task.done():
            self._footer_task.cancel()
            try:
                await self._footer_task
            except asyncio.CancelledError:
                pass

        if self._app is None:
            # Legacy terminal: clear counter from last line
            try:
                import os
                rows = os.get_terminal_size().lines
                safe_write(f"\033[s\033[{rows};1H\r\033[K\033[u")
            except Exception:
                pass
        else:
            # Чек-лист плана (mcp_agent/pipeline.py:"plan_steps") — снимается
            # ЛЮБЫМ завершением хода (успех/ошибка/Ctrl+C), не только заменой
            # следующим планом. finish() вызывается из cli.py's finally,
            # так что это единственное надёжное место, попадающее на все
            # три исхода сразу.
            self._app.clear_plan()
        self._current_stage = ""
        set_title("FlowAI")

    def flush_stats(self, stopped: bool = False) -> tuple[int, int, int]:
        """Update session counters and re-render response with markdown formatting.

        Returns (tokens_in, tokens_out, tokens_in_content) actually added this
        turn, so callers can also feed them into cross-session usage tracking.
        tokens_in_content — tokens_in minus the estimated repeated system-prompt
        overhead (see mcp_agent/agent.py:_SYSTEM_PROMPT_TOKENS_ESTIMATE)."""
        if not self._first:
            tok_in  = self.pending_stats.get("tokens_in", 0)
            tok_out = self.pending_stats.get("tokens_out", 0) or self._tok_approx
            tok_in_content = self.pending_stats.get("tokens_in_content", tok_in)
            dur     = self.pending_stats.get("duration_ms") or int(
                (time.monotonic() - self._t_wall) * 1000
            )
            # gen_duration_ms — чистое время генерации (mcp_agent/agent.py:
            # _stream_round), без тулов/self-heal judge-вызова. Отсутствует
            # для legacy-путей без этого поля (fallback на dur) — иначе
            # "скорость" в usage_screen делилась бы на 0/None.
            gen_dur = self.pending_stats.get("gen_duration_ms") or dur
            self._stats["tokens_in"]  += tok_in
            self._stats["tokens_in_content"] += tok_in_content
            self._stats["tokens_out"] += tok_out
            self._stats["duration_ms"] += dur
            self._stats["gen_duration_ms"] += gen_dur
            self._stats["messages"]   += 1

            elapsed = dur / 1000.0
            stats_ansi = _ai_header(tok_out, elapsed)

            if self._app is not None:
                # Update stats footer
                self._app.set_stats(stats_ansi)

            if not stopped and self.full_response.strip():
                self._rerender_markdown(tok_out, dur)
            else:
                safe_write("\n")
            return tok_in, tok_out, tok_in_content
        else:
            safe_write("\n")
            return 0, 0, 0

    def _rerender_markdown(self, tok_out: int, dur: int) -> None:
        """Re-render the response with Rich Markdown formatting."""
        if self._app is not None:
            import shutil
            try:
                term_width = shutil.get_terminal_size().columns
            except Exception:
                term_width = 120
            buf = io.StringIO()
            cap = Console(
                file=buf,
                force_terminal=True,
                highlight=False,
                color_system="truecolor",
                width=term_width,
            )
            cap.print(RichMarkdown(self.full_response, code_theme="monokai"))
            # Rich's Markdown renderer puts a blank line between every
            # block-level element (paragraph, list, heading, ...) — standard
            # for a rendered document, but a multi-paragraph reply (D&D
            # narration especially: several short paragraphs per turn) ends
            # up looking double-spaced in a terminal chat, with almost room
            # for another line between every two. Drop those blank lines
            # entirely for a denser, chat-style look — a line only counts as
            # "blank" once its own ANSI color/reset codes are stripped, so
            # this doesn't touch a genuinely blank line INSIDE a code block
            # (Rich never emits bare color-code-only lines there, only
            # around block boundaries).
            rendered = "\n".join(
                line for line in buf.getvalue().split("\n")
                if _ANSI_RE.sub("", line).strip()
            )
            # Strip leading/trailing blank lines that Rich inserts around paragraphs
            rendered = rendered.strip("\n") + "\n"
            # Re-include "AI ›" marker on the first line (mark_response_start was set
            # on that line, so replace_from_mark will overwrite it)
            lines = rendered.split("\n")
            lines[0] = "\033[1;34m AI ›\033[0m  " + lines[0]
            rendered = "\n".join(lines)
            self._app._output.replace_from_mark(rendered)
            self._app.invalidate()
        else:
            try:
                import os
                cols = os.get_terminal_size().columns
            except Exception:
                cols = 80

            lines_up = self._content_lines + 1
            for line in self.full_response.split('\n'):
                if len(line) >= cols:
                    lines_up += len(line) // cols

            safe_write(f"\033[{lines_up}A\r\033[J")
            console.print(RichMarkdown(self.full_response, code_theme="monokai"))