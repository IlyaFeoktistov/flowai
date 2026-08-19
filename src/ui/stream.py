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
# попытка". На медленном локальном железе этот вызов сам по себе может
# занимать МИНУТЫ (свой промпт из TASK+TOOL RESULTS+ANSWER, та же
# CPU-тяжёлая модель) — без отдельной фазы футер показывал бы последнюю
# фразу от answer_end/tool_end и молчал, что выглядит как зависание "уже
# всё написал, а крутится неизвестно почему".
_VERIFYING_PHRASES = (
    "проверяю ответ", "сверяю с задачей", "перечитываю сам себя",
)

# mcp_agent/pipeline.py эмитит "stage_changed" при переходе между ролями
# (mcp_agent/roles.py) — без этого пользователь видел бы только generic
# "[MCP-AGENT]" в DEBUG-выводе и не мог понять, какая именно роль сейчас
# работает (анализатор, кодер и т.д.). Ключи — то, что реально шлёт
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
_FILE_EDIT_TOOL_NAMES = ("write_file", "edit_file")

_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _shorten(val, limit: int = 80) -> str:
    s = str(val)
    return s if len(s) <= limit else s[:limit] + "…"


# Human-readable phrase per tool call, replacing the raw "tool_name { args }"
# dump that used to be the ONLY thing shown for a running tool — a
# non-technical user reads "читаю файл X (50-100)" far faster than
# "read_file { path: X, offset: 49, limit: 51 }". Falls back to a generic
# "name(args)" one-liner for any tool not explicitly covered below, so a new
# or uncommon tool still shows something reasonable instead of nothing.
def _format_tool_call(name: str, args: dict) -> str:
    if name == "read_file":
        path = args.get("path", "?")
        offset = args.get("offset") or 0
        limit = args.get("limit")
        if limit:
            rng = f" ({offset + 1}-{offset + limit})"
        elif offset:
            rng = f" (с {offset + 1})"
        else:
            rng = ""
        return f"читаю файл {path}{rng}"
    if name in ("write_file", "edit_file"):
        return f"обновляю файл {args.get('path', '?')}"
    if name in ("bash", "bash_bg"):
        suffix = " в фоне" if name == "bash_bg" else ""
        return f"выполняю команду{suffix} $ {_shorten(args.get('command', ''), 150)}"
    if name == "delete_path":
        return f"удаляю {args.get('path', '?')}"
    if name == "grep_search":
        pattern = _shorten(args.get("pattern", "?"), 60)
        path = args.get("path") or "."
        return f"ищу «{pattern}» в {path}"
    if name == "glob_search":
        return f"ищу файлы {args.get('pattern', '?')} в {args.get('path') or '.'}"
    if name == "web_search":
        return f"ищу в интернете: {_shorten(args.get('query', ''))}"
    if name == "analyze_image":
        return f"смотрю на картинку {args.get('path', '?')}"
    if name == "generate_image":
        return f"рисую: {_shorten(args.get('prompt', ''))}"
    if name == "edit_image":
        return f"редактирую картинку {args.get('path', '?')}"
    if name == "generate_music":
        return f"генерирую музыку: {_shorten(args.get('prompt', ''))}"
    if name == "generate_3d_model":
        return "генерирую 3D-модель"
    if name == "animate_3d_model":
        return f"анимирую модель {args.get('model_path', '?')}"
    if name == "generate_texture_for_model":
        return f"перегенерирую текстуру {args.get('model_path', '?')}"
    if name == "search_code_semantic":
        return f"ищу по коду: {_shorten(args.get('query', ''))}"
    if name == "search_dialog_history":
        return f"ищу в истории диалогов: {_shorten(args.get('query', ''))}"
    if name == "search_external_sources":
        return f"ищу в сохранённых страницах: {_shorten(args.get('query', ''))}"
    if name == "remember_url":
        return f"читаю страницу {args.get('url', '?')}"
    if name == "update_memory":
        return "запоминаю"
    if name == "update_knowledge":
        return f"запоминаю про проект: {args.get('key', '?')}"
    if name == "lsp":
        return f"lsp {args.get('operation', '?')} · {args.get('filePath', '?')}"
    if not args:
        return name
    preview = ", ".join(f"{k}={_shorten(v, 60)}" for k, v in args.items())
    return f"{name}({preview})"


def _format_file_edit_result(name: str, args: dict, result: str) -> tuple[str, list[str]] | None:
    """Compact "Update(path) · +N -M" summary + line-numbered diff body for
    file-edit tool results — write_file/edit_file (file_ops_server.py, via
    utils/parsing.py:unified_diff_at) return a standard unified diff with
    real file line numbers in the "@@ -a,b +c,d @@" header, which is exactly
    what this parses. Returns None if `result` isn't diff-shaped (no "@@"
    hunk header at all — e.g. write_file creating a brand-new file, with
    nothing to diff against, or a tool error) so the caller falls back to
    the generic tool_end rendering instead of showing an empty/wrong body.

    Without this, the generic rendering shows the raw diff text with only
    +/- text-color coding and no indication of which real file lines
    changed — reading it requires counting from the "@@" header by hand.
    This renders a compact header ("Update(path) · +N -M") plus a numbered
    body instead."""
    lines = result.splitlines()
    if not any(_HUNK_HEADER_RE.match(ln) for ln in lines):
        return None

    path = args.get("path") or "?"
    verb = "Write" if name == "write_file" else "Update"
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
    """Style for one line of a unified diff (write_file/edit_file already
    return this format, with real file line numbers in the "@@" header —
    see utils/parsing.py:unified_diff_at). Same red/green/cyan convention
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


# Reserve room for _ai_header's fixed-shape suffix after the label —
# " · {tok} tok · {duration}" — worst case something like
# " · 999999 tok · 23ч 59м 59с" (~28 chars) plus a margin; used by
# _footer_loop's raw-terminal fallback to truncate the variable label so
# the WHOLE rendered line can never exceed the terminal width (see its own
# comment on why an overflow there leaves stale digits on screen).
_FOOTER_FIXED_SUFFIX_WIDTH = 32


def _truncate_label_to_width(label: str, cols: int) -> str:
    """Caps `label` so `label` + _ai_header's fixed suffix fits within
    `cols` terminal columns — see _FOOTER_FIXED_SUFFIX_WIDTH and
    _footer_loop's raw-terminal branch for why an overflow there is worse
    than a truncated label."""
    budget = max(cols - _FOOTER_FIXED_SUFFIX_WIDTH, 8)
    if len(label) <= budget:
        return label
    return label[: budget - 1] + "…"


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
        # Статус-точка тула (см. on_event "tool_start"/"tool_end"): 3-й
        # элемент — зарезервированный при tool_start _ToolFold (ui/app.py:
        # _OutputControl.reserve_fold, только app-режим — в legacy-терминале
        # переписать уже проскроллившую строку небезопасно, см. комментарий
        # там же), который держит АКТУАЛЬНУЮ (может сдвинуться, если раньше
        # зарегистрированный тул успеет развернуться/свернуться) позицию
        # строки тула — читать fold.trigger_line, а не кешировать индекс.
        # FIFO, не одиночное значение — модель иногда зовёт несколько тулов
        # одним сообщением (несколько tool_start подряд до их tool_end).
        # 4-й элемент — задача мигания этой конкретной точки (см. _blink_tool_dot);
        # 5-й — форматированный заголовок (см. _format_tool_call), чтобы
        # blink/tool_end переписывали ту же фразу, а не откатывались на
        # голое имя тула.
        # 3rd element is a ui.app._ToolFold or None — not type-hinted as
        # such to avoid importing ui.app here just for a hint.
        self._pending_tool_calls: list[tuple] = []
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

    # Legacy-terminal fallback only (no mouse, no addressable lines — see
    # _print_foldable_body) — how many result lines to show before a
    # static "... N more lines" line instead of the full dump.
    _FOLD_PREVIEW = 20

    def _fill_tool_result(self, fold, trigger_text: str, markup_lines: list[str], start_expanded: bool = False) -> None:
        """Supplies a reserved fold's real content once its tool call
        finishes (ui/app.py:_OutputControl.reserve_fold/fill_fold) — the
        fold's buffer POSITION was already fixed back at tool_start, not
        here, specifically so several tools started together each land
        their result under THEIR OWN header instead of all grouping after
        whichever header happened to print last (see _ToolFold's own
        docstring). A plain result stays fully hidden until the user
        clicks the tool's "● phrase" line; click again to hide it back.
        `start_expanded` skips that — used for write_file/edit_file's diff
        (see tool_end below) so the actual code change is visible right
        away, not one extra click away, since that's the thing most worth
        seeing without being asked. Falls back to the old static,
        size-capped, always-visible print in legacy-terminal mode (no
        mouse, `fold` is None) or if the caller has no fold at all
        (defensive — a tool_end with no matching tool_start)."""
        if fold is not None:
            expanded = [_render_markup(ln) for ln in markup_lines]
            self._app._output.fill_fold(fold, trigger_text, expanded)
            if start_expanded:
                self._app._output.toggle_fold(fold)
            return
        total = len(markup_lines)
        if total <= self._FOLD_PREVIEW:
            for ln in markup_lines:
                console.print(ln)
            return
        hidden = total - self._FOLD_PREVIEW
        for ln in markup_lines[:self._FOLD_PREVIEW]:
            console.print(ln)
        console.print(f"[bright_black]     … ещё {hidden} строк[/]")

    async def _blink_tool_dot(self, fold, header: str) -> None:
        """Toggles ONE pending tool's status dot between gray and white
        every _TOOL_DOT_BLINK_S seconds. Real ANSI blink (SGR 5) is avoided
        because whether it actually blinks depends on the terminal/font and
        its rate isn't controllable — this drives the same effect explicitly
        via a redraw loop instead, same pattern as _footer_loop's own spinner
        tick, just targeting one specific historical line instead of the
        fixed footer row. App-mode only — see tool_start's own comment on
        why legacy-terminal mode can't do this safely.

        Reads `fold.trigger_line` FRESH every tick rather than a line index
        captured once at tool_start — an EARLIER tool's result being
        expanded/collapsed while this one is still running shifts every
        LATER fold's trigger_line (see _OutputControl.toggle_fold); a
        cached int would silently start rewriting the wrong line the
        moment that happens."""
        on = False
        try:
            while True:
                await asyncio.sleep(self._TOOL_DOT_BLINK_S)
                line_idx = fold.trigger_line
                if self._app is None or not (0 <= line_idx < len(self._app._output._lines)):
                    return
                on = not on
                style = "bold white" if on else "bright_black"
                self._app._output._lines[line_idx] = _render_markup(f"[{style}]  ●[/] {_escape_markup(header)}")
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
        counter. Running those as two separate tasks (a spinner task from
        tool_start alongside a counter task started at the first
        answer_start) would make them fight over the same footer line, each
        overwriting the other's set_stats() call on its own cadence — the
        footer would visibly flicker between "⠏ дёргаю рычаги... 31с | 9с"
        and "AI › 474 tok · 1м 48с". A phase change now just updates
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
                label = f"{frame} {stage_prefix}{self._phase_label}"
                if self._app is not None:
                    self._app.set_stats(_ai_header(self._tok_approx, elapsed, label))
                else:
                    try:
                        import os
                        cols, rows = os.get_terminal_size()
                        # Cap the variable part of the line so the WHOLE
                        # render can never exceed the terminal width and
                        # wrap onto the row below — `\033[K` below only
                        # clears the SINGLE row the cursor sits on, so if a
                        # longer earlier frame wrapped onto the next row,
                        # its leftover tail never gets cleared and can bleed
                        # into a later, shorter render (garbled digits in
                        # the token/duration suffix).
                        header = _ai_header(self._tok_approx, elapsed, _truncate_label_to_width(label, cols))
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
            # монохромной точки.
            # В app-режиме строка адресуема (_output._lines) — точку можно
            # честно перезаписать позже. В legacy-терминале (обычный
            # scrolling stdout) исторические строки не переписать надёжно
            # (единственный слот \033[s/\033[u уже занят _footer_loop, а
            # относительный сдвиг курсора не посчитать точно, если между
            # tool_start и tool_end что-то ещё напечаталось, напр. approval-
            # промпт) — там точка остаётся серой все время выполнения.
            # Reserved BEFORE printing the header (ui/app.py:_OutputControl.
            # reserve_fold) — fixes this tool's result position by CALL
            # order now, rather than waiting for tool_end to find out
            # wherever the buffer happens to end THEN (which, for several
            # tools started together in one turn, is after every other
            # already-printed header — see _ToolFold's own docstring).
            fold = None
            if self._app is not None:
                line_idx = len(self._app._output._lines) - 1
                fold = self._app._output.reserve_fold(line_idx)
            header = _format_tool_call(name, args)
            console.print(f"[bright_black]  ●[/] {_escape_markup(header)}")
            blink_task = (
                asyncio.create_task(self._blink_tool_dot(fold, header))
                if fold is not None else None
            )
            self._pending_tool_calls.append((name, args, fold, blink_task, header))
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

        # ── TOOL ARG CHUNK (real streaming, not shown — just counted) ─────
        # Without this, the live "~N tok" counter would only tick on
        # answer_chunk (the model's own text) — a tool call's own arguments
        # (write_file's whole new file content, edit_file's new_string,
        # ...) are real generation too, sometimes the bulk of a slow round,
        # but arrive as tool_call_chunks with .content usually empty (see
        # mcp_agent/agent.py:_stream_round's "messages" mode loop), so the
        # counter would look frozen for the ENTIRE duration and only jump
        # once at tool_start, well after the fact — e.g. "506 tok · 9m 47s"
        # reading as if the model were stuck. len//4 here (not +=1 per chunk
        # like answer_chunk) because a tool_call_chunks fragment's size
        # varies by provider/backend in a way plain content chunks don't
        # seem to on this project's own backends — a length-based estimate
        # stays roughly right regardless of how big each individual
        # fragment is.
        elif t == "tool_arg_chunk":
            text = event.get("text", "")
            if text:
                self._tok_approx += max(1, len(text) // 4)
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
            fold = None
            blink_task = None
            header = name
            if self._pending_tool_calls:
                _, pending_args, fold, blink_task, header = self._pending_tool_calls.pop(0)
            if blink_task is not None:
                blink_task.cancel()
            # Bounds-check: /clear (or any other command) can run concurrently
            # with an in-flight turn (see cli.py's own comment on this) and
            # wipe/shrink _output._lines mid-tool-call — indexing a stale
            # line index from before that would otherwise crash the whole
            # turn on an IndexError over a purely cosmetic status dot.
            trigger_text = None
            if fold is not None and self._app is not None and 0 <= fold.trigger_line < len(self._app._output._lines):
                trigger_text = _render_markup(f"[bold white]  ●[/] {_escape_markup(header)}")
                self._app._output._lines[fold.trigger_line] = trigger_text
                self._app.invalidate()
            else:
                fold = None  # can't attach a result to a fold whose header line is gone

            diffish = _format_file_edit_result(name, pending_args, result) if name in _FILE_EDIT_TOOL_NAMES and result else None
            if diffish:
                diff_header, body = diffish
                lines = [f"[bright_black]     └ {_escape_markup(diff_header)}[/]", *body]
                self._fill_tool_result(fold, trigger_text, lines, start_expanded=True)
            elif result:
                lines = result.splitlines()
                if len(lines) > 1:
                    # This branch runs exactly when `diffish` above was
                    # None, i.e. the result is NOT diff-shaped (no "@@"
                    # hunk header) — a plain bash output like `ls -la` still
                    # has lines starting with "-" (a regular file's
                    # "-rw-r--r--" permission string), which _diff_line_style
                    # would color red as if removed, exactly like a git
                    # diff, even though this is plain command output with
                    # nothing to do with a diff at all.
                    self._fill_tool_result(
                        fold, trigger_text,
                        [f"[bright_black]     {_escape_markup(ln)}[/]" for ln in lines],
                    )
                else:
                    # Legacy-terminal fallback (no click target) keeps the
                    # old 200-char safety cap — nothing there can ever
                    # reveal the rest on demand, unlike the interactive path.
                    shown = result if fold is not None or len(result) <= 200 else result[:200] + "…"
                    self._fill_tool_result(
                        fold, trigger_text,
                        [f"[bright_black]     ↳ {_escape_markup(shown)}[/]"],
                    )
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
        a multi-paragraph reply looks double-spaced WHILE typing and only
        snaps tight once generation finishes and the markdown re-render
        replaces it — a visible before/after mismatch that reads as
        "why did it just shrink?". Returns the text actually written (with
        any newlines removed accounted for), not the original — callers
        that count '\\n' for line-tracking should count THIS, not the
        input, or they overcount blank lines that never actually made it
        to the screen."""
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
        for _, _, _, blink_task, _ in self._pending_tool_calls:
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