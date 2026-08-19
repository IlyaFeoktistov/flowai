"""FlowAI full-screen TUI application built on prompt_toolkit."""
from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Callable, Awaitable

from prompt_toolkit import Application
from prompt_toolkit.auto_suggest import Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion, merge_completers
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import to_formatted_text, HTML
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.layout.containers import (
    HSplit, Window, ConditionalContainer, FloatContainer, Float,
)
from prompt_toolkit.layout.controls import (
    BufferControl, FormattedTextControl, UIControl, UIContent,
)
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style

_PERM_LABELS: dict[str, str] = {
    "bash":        "bash-команды",
    "read_file":   "чтение файлов",
    "write_file":  "запись файлов",
    "patch_file":  "редактирование файлов",
    "append_file": "добавление в файлы",
    "delete_file": "удаление файлов",
    "list_dir":    "просмотр директорий",
}

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[mKABCDEFGHJsuhr]")
_ANSI_OSC    = re.compile(r"\x1b\][^\x07]*\x07")


def _strip_ansi(text: str) -> str:
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_ESCAPE.sub("", text)
    return text


def _visible_len(text: str) -> int:
    return len(_strip_ansi(text))


def _wrap_ansi_line(line: str, width: int) -> list[str]:
    """Wrap one ANSI line to fit within *width* visible columns.

    Returns a list of sub-lines (without trailing \\n).
    Each sub-line preserves the ANSI escape codes from earlier in the line.
    """
    if _visible_len(line) <= width:
        return [line]

    result: list[str] = []
    current_visible = 0
    current_chunk: list[str] = []
    active_codes: list[str] = []  # track open SGR codes for carry-over

    # Tokenise: alternate between escape codes and plain text
    pattern = re.compile(r"(\x1b\[[0-9;]*m|\x1b\][^\x07]*\x07|\x1b\[[0-9;]*[KABCDEFGHJsuhr]|[^\x1b]+)")
    for m in pattern.finditer(line):
        token = m.group(0)
        if token.startswith("\x1b"):
            current_chunk.append(token)
            # Track SGR (colour) codes
            if token.endswith("m"):
                if token == "\x1b[0m" or token == "\x1b[m":
                    active_codes.clear()
                else:
                    active_codes.append(token)
        else:
            # plain text — may need to split
            pos = 0
            while pos < len(token):
                remaining = width - current_visible
                chunk = token[pos:pos + remaining]
                current_chunk.append(chunk)
                current_visible += len(chunk)
                pos += len(chunk)
                if current_visible >= width and pos < len(token):
                    # flush current sub-line
                    result.append("".join(current_chunk) + "\x1b[0m")
                    current_chunk = list(active_codes)  # carry open codes
                    current_visible = 0

    if current_chunk:
        result.append("".join(current_chunk))
    return result or [""]


# ── Output control ────────────────────────────────────────────────────────────

# Скролл-акселерация: одиночный тик колеса/PageUp держит базовый шаг (плавно
# листать по чуть-чуть на длинном диалоге всё ещё удобно), а серия тиков
# подряд быстрее _SCROLL_ACCEL_WINDOW секунд друг за другом разгоняет шаг
# геометрически — иначе пролистать большой диалог до нужного места означает
# сотни одинаковых мелких прыжков. Multiplier сбрасывается к 1.0, как только
# пауза между тиками превышает окно ИЛИ меняется направление — то есть один
# случайный клик колеса после разгона не улетает на всю накопленную скорость.
_SCROLL_ACCEL_WINDOW = 0.25   # секунд между тиками, чтобы считаться "быстрым"
_SCROLL_ACCEL_FACTOR = 1.6    # во сколько раз растёт множитель на каждый быстрый тик
_SCROLL_ACCEL_MAX = 8.0       # потолок множителя


class _ToolFold:
    """A togglable multi-line block of already-rendered ANSI text (a tool
    result's body, see ui/stream.py:StreamDisplay._print_foldable_body) —
    `collapsed`/`expanded` are both full renderings, `start` is the logical
    line index (into _OutputControl._lines) where the block begins. Only
    ONE of the two renderings is ever "live" in _lines at a time; toggling
    swaps the whole range in place (see _OutputControl.toggle_fold)."""

    def __init__(self, start: int, collapsed: list[str], expanded: list[str]) -> None:
        self.start = start
        self.collapsed = collapsed
        self.expanded = expanded
        self.is_expanded = False

    @property
    def lines(self) -> list[str]:
        return self.expanded if self.is_expanded else self.collapsed

    @property
    def end(self) -> int:
        """Exclusive — the range this fold currently occupies is [start, end)."""
        return self.start + len(self.lines)


class _OutputControl(UIControl):
    """Accumulates ANSI text as logical lines; wraps to width; auto-scrolls."""

    def __init__(self) -> None:
        self._lines: list[str] = []          # logical (unwrapped) lines
        self._mark_idx: int | None = None     # logical line index of response mark
        self._invalidate_cb: Callable | None = None
        self._scroll_offset: int = 0          # lines from bottom; 0 = auto-scroll to end
        self._max_scroll_offset: int = 0      # cached bound from the last create_content
        self._scroll_accel: float = 1.0       # текущий множитель скорости скролла
        self._last_scroll_ts: float | None = None
        self._last_scroll_dir: int = 0        # +1 вверх (в историю), -1 вниз (к концу)

        # Own mouse-drag selection (native terminal selection is unusable here:
        # mouse_support puts the terminal in full mouse-capture mode, so a plain
        # drag never reaches the terminal — only Shift-drag would bypass it, and
        # then the terminal's highlight desyncs from our virtual scrolling).
        self._sel_start: tuple[int, int] | None = None   # (wrapped-line idx, col)
        self._sel_end: tuple[int, int] | None = None
        self._selecting: bool = False
        self._drag_moved: bool = False        # true once the mouse actually moved mid-drag
        self._render_wrapped: list[str] = []
        self._render_continuation: list[bool] = []
        self._render_wrapped_to_logical: list[int] = []
        self._render_start: int = 0
        self._render_height: int = 0

        # Collapsible tool-output blocks (see append_fold/toggle_fold below,
        # and ui/stream.py:StreamDisplay._print_foldable_body) — a list, not
        # a dict, because hit-testing walks it to find "which fold (if any)
        # contains logical line N" and toggling one shifts every LATER
        # fold's start by the resulting line-count delta.
        self._folds: list[_ToolFold] = []
        self._hover_fold: _ToolFold | None = None

    def connect_app(self, app: "FlowAIApp") -> None:
        self._invalidate_cb = app.invalidate

    def append(self, text: str) -> None:
        """Append ANSI text (may contain newlines and partial lines)."""
        if not self._lines:
            self._lines.append("")

        # Split incoming text on newlines
        parts = text.split("\n")
        # Append first part to the last existing line
        self._lines[-1] += parts[0]
        # Each subsequent part starts a new logical line
        for part in parts[1:]:
            self._lines.append(part)

        if self._invalidate_cb:
            self._invalidate_cb()

    def _accelerated_step(self, base: int, direction: int) -> int:
        now = time.monotonic()
        if (
            self._last_scroll_ts is not None
            and direction == self._last_scroll_dir
            and now - self._last_scroll_ts < _SCROLL_ACCEL_WINDOW
        ):
            self._scroll_accel = min(_SCROLL_ACCEL_MAX, self._scroll_accel * _SCROLL_ACCEL_FACTOR)
        else:
            self._scroll_accel = 1.0
        self._last_scroll_ts = now
        self._last_scroll_dir = direction
        return max(base, round(base * self._scroll_accel))

    def scroll_up(self, step: int = 10) -> None:
        # create_content clamps only the RENDERED offset (offset =
        # min(self._scroll_offset, max_offset)) but never writes the clamp
        # back, so self._scroll_offset itself would keep growing past the
        # real top of the buffer on every extra tick at the boundary.
        # Without clamping here too, reversing direction would make
        # scroll_down burn through all that invisible excess before the
        # view actually moved, making it look "stuck". Clamping here
        # (using the bound cached from the last render) keeps the counter
        # itself honest, not just its render.
        self._scroll_offset = min(
            self._scroll_offset + self._accelerated_step(step, +1),
            self._max_scroll_offset,
        )
        if self._invalidate_cb:
            self._invalidate_cb()

    def scroll_down(self, step: int = 10) -> None:
        self._scroll_offset = max(0, self._scroll_offset - self._accelerated_step(step, -1))
        if self._invalidate_cb:
            self._invalidate_cb()

    def reset_scroll(self) -> None:
        self._scroll_offset = 0
        self._scroll_accel = 1.0
        self._last_scroll_ts = None
        if self._invalidate_cb:
            self._invalidate_cb()

    def clear(self) -> None:
        self._lines = []
        self._mark_idx = None
        self._scroll_offset = 0
        self._scroll_accel = 1.0
        self._last_scroll_ts = None
        self._sel_start = None
        self._sel_end = None
        self._selecting = False
        self._folds = []
        self._hover_fold = None
        if self._invalidate_cb:
            self._invalidate_cb()

    def mark_response_start(self) -> None:
        """Mark the current position for replace_from_mark.

        Records the current line count.  Because append() always continues
        writing into the LAST line, we store the index of the last line as
        a 'sealed' boundary: replace_from_mark truncates to mark_idx-1 lines
        (dropping the partial last line that was empty at mark time) and then
        writes new content starting fresh.
        """
        if not self._lines:
            self._lines.append("")
        # Store the index of the last empty line (the one that will be
        # modified by the next append call).  replace_from_mark will trim
        # to mark_idx-1 completed lines and restart from there.
        self._mark_idx = len(self._lines) - 1  # index of the 'open' last line

    def replace_from_mark(self, ansi_text: str) -> None:
        """Replace everything from the mark onward with *ansi_text*."""
        if self._mark_idx is None:
            self.append(ansi_text)
            return
        # Truncate to the lines BEFORE the open partial line at mark time.
        self._lines = self._lines[:self._mark_idx]
        # Write new content as fresh lines (always start on a new line)
        parts = ansi_text.split("\n")
        if not self._lines:
            self._lines.append(parts[0])
            parts = parts[1:]
        else:
            # Start fresh — push a new line for the replacement content
            self._lines.append(parts[0])
            parts = parts[1:]
        for part in parts:
            self._lines.append(part)
        if self._invalidate_cb:
            self._invalidate_cb()

    # ── Collapsible tool-output blocks ──────────────────────────────────────
    # See ui/stream.py:StreamDisplay._print_foldable_body — a tool result
    # long enough to be truncated registers one of these instead of just
    # statically printing "... N more lines", so the user can click to see
    # the rest (and click again to re-collapse). Both renderings are
    # pre-rendered ANSI text (via ui/stream.py:_render_markup), not Rich
    # markup — written straight into _lines the same way the tool-call
    # status dot already does (see on_event's tool_end in stream.py).

    def append_fold(self, collapsed: list[str], expanded: list[str]) -> None:
        """Appends a togglable block, starting collapsed. Must be called
        right after a newline (i.e. with the current last line empty) —
        every caller in this codebase already satisfies that (console.print
        always ends its own output in "\\n"), the check below is just a
        defensive fallback, not the expected path."""
        if not self._lines:
            self._lines.append("")
        if self._lines[-1] != "":
            self._lines.append("")
        start = len(self._lines) - 1
        fold = _ToolFold(start, collapsed, expanded)
        self._lines[start] = collapsed[0] if collapsed else ""
        self._lines.extend(collapsed[1:])
        self._lines.append("")  # reopen — subsequent append() calls continue here
        self._folds.append(fold)
        if self._invalidate_cb:
            self._invalidate_cb()

    def toggle_fold(self, fold: "_ToolFold") -> None:
        old_len = len(fold.lines)
        fold.is_expanded = not fold.is_expanded
        new_lines = fold.lines
        self._lines[fold.start:fold.start + old_len] = new_lines
        delta = len(new_lines) - old_len
        if delta:
            for other in self._folds:
                if other is not fold and other.start > fold.start:
                    other.start += delta
        if self._invalidate_cb:
            self._invalidate_cb()

    def _fold_at_logical(self, logical_idx: int | None) -> "_ToolFold | None":
        if logical_idx is None:
            return None
        for fold in self._folds:
            if fold.start <= logical_idx < fold.end:
                return fold
        return None

    # ── UIControl protocol ────────────────────────────────────────────────────

    def create_content(self, width: int, height: int) -> UIContent:
        from prompt_toolkit.formatted_text import ANSI, to_formatted_text

        # Build wrapped lines, tracking which ones are soft-wrap continuations
        # of the previous logical line (needed to copy text without injecting
        # spurious newlines at wrap points).
        wrapped: list[str] = []
        continuation: list[bool] = []
        # Maps a wrapped-line index back to the logical line it came from —
        # needed to hit-test a mouse position (which lands on a WRAPPED row,
        # see mouse_handler) against a fold's logical [start, end) range.
        wrapped_to_logical: list[int] = []
        for logical_idx, logical in enumerate(self._lines):
            sub = _wrap_ansi_line(logical, max(width, 1))
            for j, s in enumerate(sub):
                wrapped.append(s)
                continuation.append(j > 0)
                wrapped_to_logical.append(logical_idx)

        if not wrapped:
            wrapped = [""]
            continuation = [False]
            wrapped_to_logical = [0]

        # Scroll-aware windowing: offset=0 → auto-scroll to bottom
        total = len(wrapped)
        max_offset = max(0, total - height)
        self._max_scroll_offset = max_offset
        offset = min(self._scroll_offset, max_offset)
        end = total - offset
        start = max(0, end - height)
        visible = wrapped[start:end] if start < end else [""]

        self._render_wrapped = wrapped
        self._render_continuation = continuation
        self._render_wrapped_to_logical = wrapped_to_logical
        self._render_start = start
        self._render_height = height

        span = self._selection_span()
        hover_fold = self._hover_fold if not self._selecting else None

        def get_line(i: int):
            # A genuinely empty row (blank separator line, or padding below
            # the last real line) must still emit ONE character, not zero.
            # prompt_toolkit's Window only registers a (row, col) -> screen
            # mouse-hit mapping for characters it actually wrote (see
            # containers.py Window._copy_body: rowcol_to_yx is filled inside
            # `for c in text`) — a zero-length fragment writes nothing, so
            # NO position on that row is resolvable. Its mouse_handler wrapper
            # then falls through its "walk x backwards to find a match" loop
            # with nothing to find and reports position (0, 0) instead — i.e.
            # row 0 of the CURRENT VIEWPORT, not the row actually clicked.
            # Without this, dragging a selection across a blank line
            # between paragraphs makes _sel_end jump to the top of the
            # visible window on every such row, ballooning the selection to
            # "everything above" instead of the intended range. A single
            # unstyled space is visually identical to true emptiness but
            # gives that row exactly one resolvable column.
            if i >= len(visible):
                return [("", " ")]
            line = visible[i]
            if not line:
                fragments = [("", " ")]
            else:
                try:
                    fragments = list(to_formatted_text(ANSI(line)))
                except Exception:
                    fragments = [("", _strip_ansi(line))]
            abs_i = start + i
            if span is not None:
                (lo_line, lo_col), (hi_line, hi_col) = span
                if lo_line <= abs_i <= hi_line:
                    col_lo = lo_col if abs_i == lo_line else 0
                    col_hi = hi_col + 1 if abs_i == hi_line else None
                    fragments = _apply_selection(fragments, col_lo, col_hi)
            elif hover_fold is not None:
                logical_idx = wrapped_to_logical[abs_i] if abs_i < len(wrapped_to_logical) else None
                if logical_idx is not None and hover_fold.start <= logical_idx < hover_fold.end:
                    fragments = _apply_hover_highlight(fragments)
            return fragments

        return UIContent(
            get_line=get_line,
            line_count=len(visible),
            show_cursor=False,
        )

    def is_focusable(self) -> bool:
        return False

    def _selection_span(self):
        if self._sel_start is None or self._sel_end is None:
            return None
        return (self._sel_start, self._sel_end) if self._sel_start <= self._sel_end \
            else (self._sel_end, self._sel_start)

    def has_selection(self) -> bool:
        return self._sel_start is not None and self._sel_end is not None

    def clear_selection(self) -> None:
        self._sel_start = None
        self._sel_end = None
        if self._invalidate_cb:
            self._invalidate_cb()

    def _copy_selection(self) -> None:
        span = self._selection_span()
        wrapped = self._render_wrapped
        if span is None or not wrapped:
            return
        (lo_line, lo_col), (hi_line, hi_col) = span
        lo_line = max(0, min(lo_line, len(wrapped) - 1))
        hi_line = max(0, min(hi_line, len(wrapped) - 1))
        pieces: list[str] = []
        for i in range(lo_line, hi_line + 1):
            text = _strip_ansi(wrapped[i])
            col_from = lo_col if i == lo_line else 0
            col_to = hi_col + 1 if i == hi_line else len(text)
            if i > lo_line and not self._render_continuation[i]:
                pieces.append("\n")
            pieces.append(text[col_from:col_to])
        selected = "".join(pieces).strip("\n")
        if selected:
            from ui.images import copy_to_clipboard
            copy_to_clipboard(selected)

    def mouse_handler(self, mouse_event) -> None:
        from prompt_toolkit.mouse_events import MouseEventType

        et = mouse_event.event_type
        if et == MouseEventType.SCROLL_UP:
            self.scroll_up(3)
        elif et == MouseEventType.SCROLL_DOWN:
            self.scroll_down(3)
        elif et == MouseEventType.MOUSE_DOWN:
            pos = (self._render_start + mouse_event.position.y, mouse_event.position.x)
            self._sel_start = pos
            self._sel_end = pos
            self._selecting = True
            self._drag_moved = False
            if self._invalidate_cb:
                self._invalidate_cb()
        elif et == MouseEventType.MOUSE_MOVE:
            if self._selecting:
                y = mouse_event.position.y
                if y <= 0:
                    self.scroll_up(1)
                elif y >= self._render_height - 1:
                    self.scroll_down(1)
                y = max(0, min(y, self._render_height - 1))
                new_end = (self._render_start + y, mouse_event.position.x)
                if new_end != self._sel_start:
                    self._drag_moved = True
                self._sel_end = new_end
                if self._invalidate_cb:
                    self._invalidate_cb()
                return
            # Not dragging a selection — hover-highlight a collapsible
            # tool-output block under the cursor (see append_fold), so it
            # reads as clickable before the user actually clicks it.
            fold = self._fold_at_logical(self._logical_at(mouse_event.position.y))
            if fold is not self._hover_fold:
                self._hover_fold = fold
                if self._invalidate_cb:
                    self._invalidate_cb()
        elif et == MouseEventType.MOUSE_UP:
            if self._selecting:
                self._selecting = False
                if self._drag_moved:
                    self._copy_selection()
                else:
                    # Plain click, no drag — don't leave a 1-char selection
                    # behind; toggle a collapsible tool-output block instead,
                    # if the click landed on one (see append_fold/toggle_fold).
                    clicked_wrapped_idx = self._sel_start[0] if self._sel_start else None
                    self._sel_start = None
                    self._sel_end = None
                    logical_idx = (
                        self._render_wrapped_to_logical[clicked_wrapped_idx]
                        if clicked_wrapped_idx is not None
                        and 0 <= clicked_wrapped_idx < len(self._render_wrapped_to_logical)
                        else None
                    )
                    fold = self._fold_at_logical(logical_idx)
                    if fold is not None:
                        self.toggle_fold(fold)
                    if self._invalidate_cb:
                        self._invalidate_cb()

    def _logical_at(self, screen_y: int) -> int | None:
        """Logical line index under a screen row *within the current
        viewport* — same (render_start + y) convention MOUSE_DOWN/MOUSE_MOVE
        already use for selection, just resolved one step further via
        _render_wrapped_to_logical."""
        abs_i = self._render_start + screen_y
        if 0 <= abs_i < len(self._render_wrapped_to_logical):
            return self._render_wrapped_to_logical[abs_i]
        return None


def _apply_selection(fragments, col_lo: int, col_hi: int | None):
    """Return *fragments* with a 'reverse' attribute added over [col_lo, col_hi)."""
    out = []
    pos = 0
    for style, text in fragments:
        for ch in text:
            in_sel = pos >= col_lo and (col_hi is None or pos < col_hi)
            out.append((f"{style} reverse" if in_sel else style, ch))
            pos += 1
    return out


def _apply_hover_highlight(fragments):
    """Lightens fragment TEXT color for hover (collapsible tool-output
    blocks, see _fold_at_logical) — NOT reverse-video like _apply_selection:
    swapping fg/bg read as a confusing background box/outline rather than
    "this text is clickable", per direct user feedback. Appending a `fg:`
    override wins over whatever color the original ANSI carried
    (bright_black, diff green/red/cyan, ...) because prompt_toolkit resolves
    same-attribute style tokens left-to-right, later wins — verified via
    Style.get_attrs_for_style_str."""
    return [(f"{style} bold fg:ansiwhite", text) for style, text in fragments]


# ── Command completer ───────────────────────────────────────────────────────

COMMANDS: list[tuple[str, str]] = [
    ("/gen",      "generate image directly"),
    ("/img",      "load image from disk"),
    ("/paste",    "paste image from clipboard"),
    ("/music",    "stream generated music, /music again or Ctrl+C to stop"),
    ("/music_gen", "generate a single music track directly"),
    ("/gen_model", "generate a 3D model (--rig, --raw, @ref from img-refs/)"),
    ("/anim",      "animate the last generated (rigged) 3D model"),
    ("/gen_texture", "repaint an existing model's texture from a reference image (@model @ref, any order)"),
    ("/talk",      "speak text directly, no model"),
    ("/usage",    "session statistics"),
    ("/doctor",   "health check: Ollama/model/MCP servers/storage"),
    ("/update",   "check and pull flowAI updates from git"),
    ("/clean",    "clean up accumulated junk: logs/trash/snapshots/project indexes"),
    ("/settings", "model and GPU settings"),
    ("/memory",   "view and delete remembered facts/knowledge"),
    ("/plugin",   "list installed plugins/skills/hooks and what each provides"),
    ("/dnd",      "D&D mode: list saves / new game / continue / exit"),
    ("/inventory", "current D&D character's inventory (D&D mode only)"),
    ("/status",    "current D&D character/world status: who/where/when/health/party (D&D mode only)"),
    ("/facts",     "remembered D&D facts/lore for this game (D&D mode only)"),
    ("/clear",    "clear history"),
    ("/help",     "show help"),
]

_KNOWN_CMDS = {cmd for cmd, _ in COMMANDS}

# cli.py only handles these three while _dnd_active is True (see its own
# "/inventory" and "/status"/"/facts" if-blocks) -- outside a D&D game
# they fall through as a plain chat message, and the model has no matching
# tool to call for "текущий инвентарь"/etc., just an error. Hidden from the
# popup outside D&D mode so they're not offered when they'd only error.
_DND_ONLY_CMDS = {"/inventory", "/status", "/facts"}


class _CmdCompleter(Completer):
    def __init__(self, app: "FlowAIApp | None" = None):
        self._app = app

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        dnd_active = bool(self._app and self._app._dnd_active)
        for cmd, desc in COMMANDS:
            if cmd in _DND_ONLY_CMDS and not dnd_active:
                continue
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text),
                                 display=cmd, display_meta=desc)


class _ImgRefCompleter(Completer):
    """@name completion for /gen_model's img-refs/ syntax, /anim's and
    /gen_texture's generated/models/ syntax (see gen3d/img_refs.py,
    gen3d/model_refs.py) -- only fires after "/gen_model ", "/anim " or
    "/gen_texture ", on the @-token the cursor currently sits in, so a stray
    '@' in normal chat text (an email, a handle) never triggers a popup."""

    _TRIGGER = re.compile(r"^/(gen_model|anim|gen_texture)\s")
    _AT_TOKEN = re.compile(r"@([\w.\-]*)$")

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        trigger = self._TRIGGER.match(text)
        if not trigger:
            return
        m = self._AT_TOKEN.search(text)
        if not m:
            return
        partial = m.group(1)
        try:
            if trigger.group(1) == "anim":
                from gen3d.model_refs import list_models
                groups = [(list_models(), "generated/models/")]
            elif trigger.group(1) == "gen_texture":
                # order-independent -- either @-token can end up being the
                # model or the reference image, so suggest both.
                from gen3d.model_refs import list_models
                from gen3d.img_refs import list_refs
                groups = [(list_models(), "generated/models/"), (list_refs(), "img-refs/")]
            else:
                from gen3d.img_refs import list_refs
                groups = [(list_refs(), "img-refs/")]
        except Exception:
            return
        for paths, meta in groups:
            for path in paths:
                if path.name.lower().startswith(partial.lower()):
                    yield Completion(f"@{path.name}", start_position=-len(m.group(0)),
                                     display=f"@{path.name}", display_meta=meta)


class _FileSearchCompleter(Completer):
    """@partial completion in plain chat text -- recursively searches the
    current working directory (the one flowai was launched from) for files
    whose relative path contains the typed text, so the user can reference a
    project file by name instead of typing the full path. Skipped under
    /gen_model, /anim, /gen_texture -- those already have their own
    @-completer scoped to img-refs/generated dirs, see _ImgRefCompleter
    above; running both there would just show irrelevant project files
    alongside the actual model/image refs those commands take."""

    _SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules",
                  ".mypy_cache", ".pytest_cache", "build", "dist"}
    _MAX_RESULTS = 30
    _MAX_SCANNED = 5000

    _AT_TOKEN = re.compile(r"@([^\s@]*)$")

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if _ImgRefCompleter._TRIGGER.match(text):
            return
        m = self._AT_TOKEN.search(text)
        if not m:
            return
        partial = m.group(1).lower()
        root = Path.cwd()
        scanned = 0
        matches = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                scanned += 1
                rel = Path(dirpath, name).relative_to(root).as_posix()
                if partial in rel.lower():
                    matches.append(rel)
                if scanned >= self._MAX_SCANNED:
                    break
            if scanned >= self._MAX_SCANNED:
                break
        matches.sort(key=lambda rel: (not Path(rel).name.lower().startswith(partial), len(rel)))
        for rel in matches[:self._MAX_RESULTS]:
            yield Completion(f"@{rel}", start_position=-len(m.group(0)), display=f"@{rel}")


class _CmdLexer(Lexer):
    def lex_document(self, document):
        def tokenize(line_no: int):
            line = document.lines[line_no]
            if not line.startswith("/"):
                return [("", line)]
            parts = line.split(maxsplit=1)
            cmd  = parts[0]
            rest = (" " + parts[1]) if len(parts) > 1 else ""
            style = "class:cmd.ok" if cmd in _KNOWN_CMDS else "class:cmd.bad"
            tokens = [(style, cmd)]
            if rest:
                tokens.append(("class:cmd.arg", rest))
            return tokens
        return tokenize


# ── FlowAIApp ─────────────────────────────────────────────────────────────────

class FlowAIApp:
    """Full-screen prompt_toolkit application for FlowAI."""

    def __init__(self) -> None:
        self._submit_cb: Callable[[str], Awaitable[None]] | None = None
        self._output = _OutputControl()
        self._header_line_count: int = 0
        self._stats_text: str = ""
        self._recap_text: str = ""
        self._queue_size: int = 0
        self._history: list[str] = []
        self._history_pos: int = -1
        self._saved_input: str = ""

        self._active_task: "asyncio.Task | None" = None

        # Ctrl+C priority targets that must be stopped BEFORE it falls back
        # to cancelling the chat request (see _build_keybindings/_interrupt)
        # — an in-progress mic recording or a running /music stream, either
        # set via set_recording_active/set_music_active right when they start.
        self._recording_active: bool = False
        self._stop_recording_cb: "Callable[[], None] | None" = None
        self._music_active: bool = False
        self._stop_music_cb: "Callable[[], None] | None" = None

        # /dnd (cli.py) — DIFFERENT priority than recording/music above: it's
        # not a parallel side-activity to stop, it's a standing MODE that
        # should only be exited when nothing is actively generating (see
        # _interrupt) — first Ctrl+C during a dnd turn must still just cancel
        # that turn like any normal chat response, same as _active_task
        # below; only a Ctrl+C with NOTHING running exits the mode.
        self._dnd_active: bool = False
        self._dnd_exit_cb: "Callable[[], None] | None" = None

        # /gen_model's subprocess chain (gen3d/pipeline.py) is different from
        # recording/music above: it runs AS the active task itself (a tool
        # call inside a normal chat turn), not a parallel side-activity — so
        # it doesn't need its own priority branch in _interrupt, just piggy-
        # backs on the existing _active_task.cancel() fallback (see there).
        # Cancelling the task alone only unblocks the UI -- it does NOT stop
        # the blocking subprocess running in a run_in_executor thread (that's
        # a fundamental run_in_executor limitation, not fixable from here);
        # this callback (gen3d/pipeline.py's cancel_event.set()) is what
        # actually kills the child process.
        self._stop_gen3d_cb: "Callable[[], None] | None" = None

        # Permission dialog state
        self._perm_action: str = ""
        self._perm_detail: str = ""
        self._perm_future: "asyncio.Future | None" = None
        self._perm_sel: int = 0  # 0=Y, 1=A, 2=N

        # ask_user dialog state (question with options + free-text custom answer)
        self._ask_question: str = ""
        self._ask_options: list[dict] = []  # [{"label": ..., "description": ...}, ...]
        self._ask_recommended: str | None = None  # label of the recommended option, if any
        self._ask_sel: int = 0  # index into options; len(options) == "custom answer" entry
        self._ask_custom_mode: bool = False
        self._ask_saved_buffer_text: str = ""
        self._ask_future: "asyncio.Future | None" = None

        # Plan checklist (mcp_agent/pipeline.py Planner->Coder stage) — set
        # once when Planner's plan is confirmed, checked off step by step as
        # Coder executes (mcp_agent/pipeline.py emits plan_steps/
        # plan_step_done, see ui/stream.py:StreamDisplay.on_event). Cleared
        # by StreamDisplay.finish() on EVERY turn end (success, error, or
        # Ctrl+C): leaving it up after the turn (even fully checked) would
        # read as "still doing something" once a NEW unrelated turn starts,
        # since nothing else in the footer says otherwise.
        # _plan_current — separate from _plan_done: which step Coder is
        # WORKING ON right now (mark_plan_step_current tool call, see
        # ui/stream.py), so the user isn't only told what's finished at the
        # very end of the whole round with no live indication of progress.
        self._plan_steps: list[str] = []
        self._plan_done: set[int] = set()
        self._plan_current: int | None = None

        self._app = self._build_app()
        self._output.connect_app(self)

    # ── Stats / Recap controls ──────────────────────────────────────────────

    def _stats_control(self) -> FormattedTextControl:
        def get_text():
            try:
                from prompt_toolkit.formatted_text import ANSI
                return ANSI(self._stats_text)
            except Exception:
                return [("", _strip_ansi(self._stats_text))]
        return FormattedTextControl(get_text)

    def _recap_control(self) -> FormattedTextControl:
        def get_text():
            import settings as _s
            if not _s.get("recap_enabled") or not self._recap_text:
                return [("", "")]
            return [("class:footer-recap", f" ※ {self._recap_text}")]
        return FormattedTextControl(get_text)

    def _recap_visible(self) -> bool:
        import settings as _s
        return bool(self._recap_text) and bool(_s.get("recap_enabled"))

    def _perm_formatted_text(self):
        if not self._perm_future:
            return [("", "")]
        action = self._perm_action
        label = _PERM_LABELS.get(action, action)
        detail_lines = self._perm_detail.splitlines()[:5]

        parts: list = []
        parts.append(("class:perm-title", "  ⚠  Запрос разрешения\n"))
        parts.append(("class:perm-dim", "  Действие: "))
        parts.append(("class:perm-action", f"{action}\n"))
        for i, line in enumerate(detail_lines):
            prefix = "  Команда: " if i == 0 else "           "
            parts.append(("class:perm-dim", prefix))
            parts.append(("class:perm-cmd", f"{line}\n"))
        parts.append(("", "\n"))

        if action == "bash":
            cmd_name = self._perm_detail.strip().split()[0] if self._perm_detail.strip() else "bash"
            all_label = f"да, всегда ({cmd_name})"
        else:
            all_label = f"да, всегда ({_PERM_LABELS.get(action, action)})"
        opts = [
            ("class:perm-yes", "[Y] да"),
            ("class:perm-all", f"[A] {all_label}"),
            ("class:perm-no",  "[N] нет"),
        ]
        for i, (style, text) in enumerate(opts):
            sel = self._perm_sel == i
            parts.append((f"{style} reverse" if sel else style, f"  {text}  "))
        parts.append(("", "\n"))
        return parts

    def _ask_formatted_text(self):
        if not self._ask_future:
            return [("", "")]

        parts: list = []
        parts.append(("class:ask-title", f"  ❓ {self._ask_question}\n\n"))

        if self._ask_custom_mode:
            for i, opt in enumerate(self._ask_options, 1):
                parts.append(("class:ask-dim", f"  [{i}] {opt.get('label', '')}\n"))
            parts.append(("class:ask-custom-active",
                          "  ✎ Свой ответ — печатайте внизу, Enter отправит, Esc отменит\n"))
        else:
            for i, opt in enumerate(self._ask_options):
                sel = self._ask_sel == i
                label = opt.get("label", "")
                desc = opt.get("description", "")
                style = "class:ask-opt reverse" if sel else "class:ask-opt"
                parts.append((style, f"  [{i + 1}] {label}"))
                if label == self._ask_recommended:
                    tag_style = style if sel else "class:ask-recommended"
                    parts.append((tag_style, "  ★ рекомендуется"))
                parts.append((style, "\n"))
                if desc:
                    desc_style = "class:ask-desc reverse" if sel else "class:ask-desc"
                    parts.append((desc_style, f"      {desc}\n"))
            custom_idx = len(self._ask_options)
            sel = self._ask_sel == custom_idx
            style = "class:ask-opt reverse" if sel else "class:ask-opt"
            parts.append((style, "  ✎ Свой вариант...\n"))
        parts.append(("", "\n"))
        return parts

    def _plan_formatted_text(self):
        if not self._plan_steps:
            return [("", "")]
        done = len(self._plan_done)
        total = len(self._plan_steps)
        parts: list = [("class:plan-title", f"  📝 План ({done}/{total})\n")]
        for i, step in enumerate(self._plan_steps):
            if i in self._plan_done:
                parts.append(("class:plan-done", f"  ✅ {i + 1}. {step}\n"))
            elif i == self._plan_current:
                parts.append(("class:plan-current", f"  🟥 {i + 1}. {step}\n"))
            else:
                parts.append(("class:plan-pending", f"  ☐ {i + 1}. {step}\n"))
        return parts

    def set_plan_steps(self, steps: list[str]) -> None:
        """Заменяет текущий план целиком (новый Planner-раунд) — не
        накапливает шаги поверх предыдущего плана."""
        self._plan_steps = list(steps)
        self._plan_done = set()
        self._plan_current = None
        self.invalidate()

    def mark_plan_step_done(self, index: int) -> None:
        if 0 <= index < len(self._plan_steps):
            self._plan_done.add(index)
            if self._plan_current == index:
                self._plan_current = None
            self.invalidate()

    def set_plan_current(self, index: int) -> None:
        """Coder started working on step `index` (0-based) — see
        mark_plan_step_current in mcp_agent/ask_user_tool.py and its
        interception in ui/stream.py. Ignored for an already-done step
        (Coder re-narrating/self-correcting shouldn't un-cross a
        finished item) or an out-of-range index (a hallucinated step
        number shouldn't crash the UI)."""
        if 0 <= index < len(self._plan_steps) and index not in self._plan_done:
            self._plan_current = index
            self.invalidate()

    def clear_plan(self) -> None:
        self._plan_steps = []
        self._plan_done = set()
        self._plan_current = None
        self.invalidate()

    # ── Build application ───────────────────────────────────────────────────

    def _build_app(self) -> Application:
        # Output window — fills remaining space
        output_win = Window(
            content=self._output,
            style="class:output",
            wrap_lines=False,
        )

        # Divider between output and footer
        divider = Window(height=1, char="─", style="class:footer-divider")

        # Stats window (always reserved to keep output area stable).
        stats_ctrl = self._stats_control()
        stats_win = Window(
            content=stats_ctrl,
            height=1,
            style="class:footer-stats",
        )

        # Recap window (always reserved to avoid footer shift when recap toggles).
        recap_ctrl = self._recap_control()
        recap_win = Window(
            content=recap_ctrl,
            height=1,
            style="class:footer-recap",
        )

        # Input buffer
        self._buffer = Buffer(
            name="input",
            completer=merge_completers([_CmdCompleter(self), _ImgRefCompleter(), _FileSearchCompleter()]),
            complete_while_typing=True,
            multiline=True,
        )

        input_ctrl = BufferControl(
            buffer=self._buffer,
            lexer=_CmdLexer(),
            include_default_input_processors=True,
            focusable=True,
        )

        # A hardcoded height of 1 would leave a long line no room to wrap —
        # get_line_prefix below already distinguishes wrap_count>0
        # (continuation lines get "  " instead of "› "), i.e. wrapping is
        # intended here, but with only 1 row available prompt_toolkit falls
        # back to scrolling the single visible row horizontally to keep the
        # cursor in view instead of showing the wrapped continuation.
        # Dimension(min=1, max=10) lets the window grow with the buffer's
        # actual wrapped line count (up to 10 rows) instead of a fixed
        # height.
        #
        # With just min/max, HSplit's _divide_heights does a SECOND pass
        # after giving every child its content-driven preferred size — it
        # then keeps growing whichever children haven't hit their own `max`
        # yet to soak up any leftover terminal height, regardless of what
        # their content needs. Since output_win has no explicit height
        # (effectively max=huge) and this window's max was a static 10,
        # both are eligible for that leftover-space giveaway, so a terminal
        # taller than a few dozen rows would inflate a ONE-LINE input to
        # the full 10 rows (confirmed against HSplit._divide_heights: short
        # text renders at height=10, not 1). dont_extend_height=True caps
        # this window's reported max at whatever its CONTENT actually
        # prefers (still ≤10 from the Dimension above), so it can no longer
        # grow past what the current text needs, while still being free to
        # grow up to 10 as the text actually wraps into more lines.
        input_win = Window(
            content=input_ctrl,
            height=Dimension(min=1, max=10),
            dont_extend_height=True,
            wrap_lines=True,
            style="class:footer-input",
            get_line_prefix=lambda line_no, wrap_count: (
                [("class:footer-prompt", "› ")]
                if line_no == 0 and wrap_count == 0 else
                [("class:footer-prompt", "  ")]
            ),
        )

        # Permission dialog (in-TUI, shown above footer when awaiting confirmation)
        is_confirming = Condition(lambda: self._perm_future is not None)
        perm_ctrl = FormattedTextControl(self._perm_formatted_text)
        perm_win = ConditionalContainer(
            content=Window(content=perm_ctrl, dont_extend_height=True),
            filter=is_confirming,
        )
        perm_divider = ConditionalContainer(
            content=Window(height=1, char="─", style="class:footer-divider"),
            filter=is_confirming,
        )

        # ask_user dialog (in-TUI question with options + free-text custom answer)
        is_asking = Condition(lambda: self._ask_future is not None)
        ask_ctrl = FormattedTextControl(self._ask_formatted_text)
        ask_win = ConditionalContainer(
            content=Window(content=ask_ctrl, dont_extend_height=True),
            filter=is_asking,
        )
        ask_divider = ConditionalContainer(
            content=Window(height=1, char="─", style="class:footer-divider"),
            filter=is_asking,
        )

        # Without hiding it, the real input bar (input_win below) stays
        # visible and cursor-active-LOOKING the whole time a perm/ask dialog
        # is open, even though every key while just navigating a list/Y-N
        # (is_confirming, or is_asking WITHOUT custom mode) is swallowed by
        # the "<any>" catch-all bindings below and never reaches this buffer
        # at all — an empty "› " prompt line sitting right under the dialog's
        # own "✎ Свой вариант..."/options looks like a SECOND place to type
        # (the input bar appearing duplicated). Only actually show it in
        # normal mode, or once the user has explicitly entered ask_user's
        # custom-answer mode (where typing into it is the real, intended
        # thing — see _ask_enter_custom_mode/is_asking_custom below).
        _hide_input_bar = is_confirming | (is_asking & Condition(lambda: not self._ask_custom_mode))

        # Plan checklist (mcp_agent/pipeline.py) — non-blocking, unlike
        # perm_win/ask_win above (no future to wait on), just a persistent
        # status board while/after Planner->Coder run. No divider of its own
        # here — stats_win's own comment below ("above the divider line")
        # already assumes exactly ONE divider for this whole block (the
        # `divider` window further down, right before input_display). A
        # second divider here would double that line the moment a plan was
        # visible, with only stats_win/recap_win (1-2 often-blank lines)
        # between the two, reading as one bar duplicated rather than two
        # intentional separators.
        has_plan = Condition(lambda: bool(self._plan_steps))
        plan_ctrl = FormattedTextControl(self._plan_formatted_text)
        plan_win = ConditionalContainer(
            content=Window(content=plan_ctrl, dont_extend_height=True),
            filter=has_plan,
        )

        # Hints toolbar (shows queue badge / perm hints when needed)
        def _hints_text():
            if self._perm_future is not None:
                return [("class:footer-hints",
                         "  ←→·выбор   Y·да   A·все   N·нет   Enter·подтвердить")]
            if self._ask_future is not None:
                if self._ask_custom_mode:
                    return [("class:footer-hints", "  Enter·отправить   Esc·назад к вариантам")]
                return [("class:footer-hints",
                         "  ↑↓·выбор   1-9·быстрый выбор   Enter·подтвердить   Esc·пропустить")]
            parts = [("class:footer-hints",
                      " Tab·команды  ↑↓·история  Alt+V·вставить картинку  Alt+R·голосовой ввод  Shift+мышь·выделить  Ctrl+C·стоп  Ctrl+D·выход")]
            if self._queue_size > 0:
                parts.append(("class:footer-queue", f"  ·  +{self._queue_size} в очереди"))
            return parts
        hints_ctrl = FormattedTextControl(_hints_text)
        hints_win = Window(
            content=hints_ctrl,
            height=1,
            style="class:footer",
        )

        # Float completion menu
        divider2 = Window(height=1, char="─", style="class:footer-divider")

        # См. _hide_input_bar выше — не рисуем реальный input_win, когда
        # печатать в него бессмысленно (perm-диалог, ask_user-список без
        # custom-режима): eager keybindings всё равно перехватывают клавиши
        # раньше, чем они дошли бы до буфера, скрытие только визуальное.
        input_display = ConditionalContainer(content=input_win, filter=~_hide_input_bar)

        float_container = FloatContainer(
            content=HSplit([
                output_win,
                plan_win,    # plan checklist (conditional, above the footer)
                stats_win,   # spinner/counter above the divider line
                perm_win,    # permission dialog (conditional)
                perm_divider,
                ask_win,     # ask_user dialog (conditional)
                ask_divider,
                recap_win,
                divider,
                input_display,
                divider2,
                hints_win,
            ]),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=2),
                )
            ],
        )

        layout = Layout(float_container, focused_element=input_win)

        kb = self._build_keybindings()

        app_style = Style.from_dict({
            "cmd.ok":  "bold cyan",
            "cmd.bad": "bold red",
            "cmd.arg": "",
            "footer-divider":                   "#cdd6f4",
            "footer-stats":                     "#89b4fa",
            "footer-recap":                     "#a6adc8",
            "footer-input":                     "#cdd6f4",
            "footer-prompt":                    "ansigreen bold",
            "footer-hints":                     "#6c7086",
            "footer-queue":                     "#f38ba8",
            "auto-suggestion":                  "#6c7086 italic",
            "completion-menu.completion":              "bg:#1e1e2e #cdd6f4",
            "completion-menu.completion.current":      "bg:#89b4fa #1e1e2e bold",
            "completion-menu.meta.completion":         "bg:#313244 #a6adc8",
            "completion-menu.meta.completion.current": "bg:#89b4fa #1e1e2e",
            "scrollbar.background": "bg:#313244",
            "scrollbar.button":     "bg:#89b4fa",
            "perm-title":  "bold ansiyellow",
            "perm-dim":    "#6c7086",
            "perm-action": "ansicyan bold",
            "perm-cmd":    "#cdd6f4",
            "perm-yes":    "bold ansigreen",
            "perm-all":    "bold ansiblue",
            "perm-no":     "bold ansired",
            "ask-title":         "bold ansiyellow",
            "ask-dim":           "#6c7086",
            "ask-opt":           "#cdd6f4",
            "ask-desc":          "#6c7086 italic",
            "ask-recommended":   "bold ansigreen",
            "ask-custom-active": "bold ansicyan",
            "plan-title":   "bold ansiblue",
            "plan-done":    "#a6adc8 strike",
            "plan-pending": "#cdd6f4",
            "plan-current": "bold ansired",
        })

        return Application(
            layout=layout,
            key_bindings=kb,
            style=app_style,
            full_screen=True,
            mouse_support=True,
            color_depth=None,
        )

    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()
        is_confirming = Condition(lambda: self._perm_future is not None)
        is_asking = Condition(lambda: self._ask_future is not None)
        is_asking_list = Condition(lambda: self._ask_future is not None and not self._ask_custom_mode)
        is_asking_custom = Condition(lambda: self._ask_future is not None and self._ask_custom_mode)
        is_blocking = is_confirming | is_asking

        # ── Permission dialog keys (active only while dialog is visible) ────

        def _resolve(result: str) -> None:
            if self._perm_future and not self._perm_future.done():
                self._perm_future.set_result(result)

        @kb.add("left",   filter=is_confirming, eager=True)
        @kb.add("s-tab",  filter=is_confirming, eager=True)
        def _perm_prev(event):
            self._perm_sel = (self._perm_sel - 1) % 3
            self.invalidate()

        @kb.add("right",  filter=is_confirming, eager=True)
        @kb.add("tab",    filter=is_confirming, eager=True)
        def _perm_next(event):
            self._perm_sel = (self._perm_sel + 1) % 3
            self.invalidate()

        @kb.add("enter",  filter=is_confirming, eager=True)
        def _perm_enter(event):
            _resolve(["y", "a", "n"][self._perm_sel])

        @kb.add("y", filter=is_confirming, eager=True)
        @kb.add("Y", filter=is_confirming, eager=True)
        def _perm_yes(event): _resolve("y")

        @kb.add("a", filter=is_confirming, eager=True)
        @kb.add("A", filter=is_confirming, eager=True)
        def _perm_all(event): _resolve("a")

        @kb.add("n",      filter=is_confirming, eager=True)
        @kb.add("N",      filter=is_confirming, eager=True)
        @kb.add("escape", filter=is_confirming, eager=True)
        @kb.add("c-c",    filter=is_confirming, eager=True)
        def _perm_no(event): _resolve("n")

        @kb.add("<any>",  filter=is_confirming, eager=True)
        def _perm_swallow(event):
            pass  # block all other keys from reaching the input buffer

        # ── ask_user dialog keys (question + options + free-text answer) ────

        def _ask_resolve(value: str | None) -> None:
            if self._ask_future and not self._ask_future.done():
                self._ask_future.set_result(value)

        def _ask_enter_custom_mode() -> None:
            self._ask_custom_mode = True
            self._ask_saved_buffer_text = self._buffer.text
            self._buffer.reset()
            self.invalidate()

        def _ask_leave_custom_mode() -> None:
            self._buffer.reset()
            if self._ask_saved_buffer_text:
                self._buffer.insert_text(self._ask_saved_buffer_text)
            self._ask_saved_buffer_text = ""
            self._ask_custom_mode = False
            self.invalidate()

        @kb.add("up",   filter=is_asking_list, eager=True)
        @kb.add("left", filter=is_asking_list, eager=True)
        def _ask_prev(event):
            total = len(self._ask_options) + 1
            self._ask_sel = (self._ask_sel - 1) % total
            self.invalidate()

        @kb.add("down",  filter=is_asking_list, eager=True)
        @kb.add("right", filter=is_asking_list, eager=True)
        @kb.add("tab",   filter=is_asking_list, eager=True)
        def _ask_next(event):
            total = len(self._ask_options) + 1
            self._ask_sel = (self._ask_sel + 1) % total
            self.invalidate()

        @kb.add("enter", filter=is_asking_list, eager=True)
        def _ask_enter(event):
            if self._ask_sel < len(self._ask_options):
                _ask_resolve(self._ask_options[self._ask_sel]["label"])
            else:
                _ask_enter_custom_mode()

        def _make_ask_number(n: int):
            def _ask_number(event):
                idx = n - 1
                if idx < len(self._ask_options):
                    _ask_resolve(self._ask_options[idx]["label"])
            return _ask_number

        for _n in range(1, 10):
            kb.add(str(_n), filter=is_asking_list, eager=True)(_make_ask_number(_n))

        @kb.add("escape", filter=is_asking_list, eager=True)
        @kb.add("c-c",    filter=is_asking_list, eager=True)
        def _ask_dismiss(event):
            _ask_resolve(None)

        @kb.add("<any>",  filter=is_asking_list, eager=True)
        def _ask_swallow(event):
            pass  # block all other keys from reaching the input buffer

        @kb.add("enter", filter=is_asking_custom, eager=True)
        def _ask_submit_custom(event):
            text = self._buffer.text.strip()
            if not text:
                return
            self._ask_saved_buffer_text = ""
            self._buffer.reset()
            self._ask_custom_mode = False
            _ask_resolve(text)

        @kb.add("escape", filter=is_asking_custom, eager=True)
        @kb.add("c-c",    filter=is_asking_custom, eager=True)
        def _ask_cancel_custom(event):
            _ask_leave_custom_mode()

        # ── App-level keys ─────────────────────────────────────────────────

        @kb.add("c-d")
        def _exit(event):
            self._app.exit()

        @kb.add("c-c", filter=~is_blocking)
        def _interrupt(event):
            # Mouse-drag selection already copies on release (see
            # _OutputControl.mouse_handler) — the highlight just stays visible
            # afterwards. Without this check, a leftover highlight turned a
            # habitual "copy" Ctrl+C into an accidental generation cancel.
            if self._output.has_selection():
                self._output.clear_selection()
                return
            # Приоритет ПЕРЕД отменой чат-запроса: запись/музыка — фоновые
            # процессы, о которых ниже (_active_task) ничего не знает, и без
            # этой проверки Ctrl+C во время записи/потока молча падал бы на
            # bare-ветку "очистить буфер ввода", никак их не останавливая.
            if self._recording_active and self._stop_recording_cb:
                self._stop_recording_cb()
                return
            if self._music_active and self._stop_music_cb:
                from ui.console import console
                self._stop_music_cb()
                console.print("[dim]  🎵 останавливаю (доиграет текущий кусок)...[/]\n")
                return
            # D&D exit — ONLY when nothing is actively generating (checked
            # first, before the _active_task branch below would otherwise
            # cancel it): a Ctrl+C DURING the DM's reply must just stop that
            # reply, same as any other chat turn — jumping straight out of
            # the mode on the very first Ctrl+C would surprise-exit a game
            # mid-response instead of just interrupting it. This branch only
            # ever fires on a SECOND Ctrl+C, once generation has already
            # ended (or never started this turn).
            if self._dnd_active and self._dnd_exit_cb and (self._active_task is None or self._active_task.done()):
                self._dnd_exit_cb()
                return
            if self._active_task and not self._active_task.done():
                # /gen_model's subprocess (if any is running right now) needs
                # to actually be killed, not just have its awaiting coroutine
                # cancelled — see _stop_gen3d_cb's own comment in __init__.
                if self._stop_gen3d_cb:
                    self._stop_gen3d_cb()
                self._active_task.cancel()
            elif self._buffer.text:
                self._buffer.reset()

        @kb.add("tab", filter=~is_blocking)
        def _tab(event):
            buf = event.app.current_buffer
            if not buf.text:
                if buf.suggestion and buf.suggestion.text:
                    buf.insert_text(buf.suggestion.text)
                    return
                buf.insert_text("/")
            buf.start_completion(select_first=False)

        @kb.add("c-v")
        @kb.add("escape", "v")   # Alt+V Latin layout
        @kb.add("escape", "м")   # Alt+V Russian layout (V key → м)
        def _paste_image(event):
            from ui.images import paste_image_from_clipboard
            from ui.console import console
            placeholder = paste_image_from_clipboard()
            if placeholder:
                event.app.current_buffer.insert_text(placeholder)
            else:
                console.print("[red] ✗ Буфер не содержит изображение[/]\n")

        @kb.add("escape", "r")   # Alt+R Latin layout
        @kb.add("escape", "к")   # Alt+R Russian layout (R key → к)
        def _record_voice(event):
            from ui.console import console
            from ui.audio import start_recording, stop_recording, transcribe

            # Alt+R снова во время уже идущей записи — не открывать вторую
            # запись поверх первой (Ctrl+C — единственный способ остановить,
            # см. _interrupt/set_recording_active выше).
            if self._recording_active:
                return

            buf = event.app.current_buffer
            proc = start_recording()
            if proc is None:
                console.print("[red] ✗ Не удалось начать запись с микрофона[/]\n")
                return

            async def _finish():
                loop = asyncio.get_event_loop()
                console.print("[dim]  🎤 распознаю...[/]")
                wav_path = await loop.run_in_executor(None, stop_recording, proc)
                if not wav_path:
                    console.print("[red] ✗ Не удалось записать с микрофона[/]\n")
                    return
                try:
                    text = await loop.run_in_executor(None, transcribe, wav_path)
                except Exception as e:
                    console.print(f"[red] ✗ Распознавание не удалось: {e}[/]\n")
                    return
                if text:
                    buf.insert_text(text)
                else:
                    console.print("[red] ✗ Ничего не распознано[/]\n")

            def _stop_now():
                # Синхронно, ДО фонового _finish() — иначе повторный Ctrl+C
                # (до того, как stop_recording успел отработать) увидел бы
                # _recording_active ещё True и попытался бы остановить
                # запись во второй раз.
                self.set_recording_active(False)
                event.app.create_background_task(_finish())

            console.print("[dim]  🎤 запись... Ctrl+C — стоп[/]")
            self.set_recording_active(True, _stop_now)

        from prompt_toolkit.keys import Keys
        from ui.paste_store import (is_large, store_paste,
                                     placeholder_before_cursor, placeholder_after_cursor)

        @kb.add(Keys.BracketedPaste)
        def _large_paste(event):
            text = event.data
            if is_large(text):
                event.app.current_buffer.insert_text(store_paste(text))
            else:
                event.app.current_buffer.insert_text(text)

        @kb.add("backspace")
        def _bs(event):
            buf = event.app.current_buffer
            ph = placeholder_before_cursor(buf.document.text_before_cursor)
            if ph:
                buf.delete_before_cursor(len(ph))
            else:
                buf.delete_before_cursor(1)

        @kb.add("left")
        def _left(event):
            buf = event.app.current_buffer
            ph = placeholder_before_cursor(buf.document.text_before_cursor)
            if ph:
                buf.cursor_position -= len(ph)
            else:
                buf.cursor_position -= 1

        @kb.add("right")
        def _right(event):
            buf = event.app.current_buffer
            ph = placeholder_after_cursor(buf.document.text_after_cursor)
            if ph:
                buf.cursor_position += len(ph)
            else:
                buf.cursor_position += 1

        @kb.add("pageup")
        def _scroll_up(event):
            self._output.scroll_up(10)

        @kb.add("pagedown")
        def _scroll_down(event):
            self._output.scroll_down(10)

        @kb.add("up", filter=~is_blocking)
        def _hist_up(event):
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_previous()
                return
            if not self._history:
                return
            if self._history_pos == -1:
                self._saved_input = buf.text
                self._history_pos = len(self._history) - 1
            elif self._history_pos > 0:
                self._history_pos -= 1
            text = self._history[self._history_pos]
            buf.set_document(Document(text, len(text)))

        @kb.add("down", filter=~is_blocking)
        def _hist_down(event):
            buf = event.app.current_buffer
            if buf.complete_state:
                buf.complete_next()
                return
            if self._history_pos == -1:
                return
            if self._history_pos < len(self._history) - 1:
                self._history_pos += 1
                text = self._history[self._history_pos]
            else:
                self._history_pos = -1
                text = self._saved_input
            buf.set_document(Document(text, len(text)))

        @kb.add("enter", filter=~is_blocking)
        def _submit(event):
            buf = event.app.current_buffer
            if buf.complete_state:
                # Live crash: the completion menu can be open with NOTHING
                # highlighted (e.g. Tab opened it with select_first=False
                # and the user never arrowed to an entry) — current_completion
                # is then None, and apply_completion(None) blows up inside
                # prompt_toolkit trying to read completion.start_position,
                # taking down the whole event loop. Enter with nothing
                # selected should just close the menu and submit the text
                # as typed, same as most editors/shells.
                completion = buf.complete_state.current_completion
                if completion is not None:
                    buf.apply_completion(completion)
                else:
                    buf.cancel_completion()
            text = buf.text
            if not text.strip():
                return
            self._history.append(text)
            self._history_pos = -1
            self._saved_input = ""
            buf.reset()
            self._output.reset_scroll()
            if self._submit_cb:
                event.app.create_background_task(self._submit_cb(text))
        return kb

    # ── Public API ──────────────────────────────────────────────────────────

    def write(self, text: str) -> None:
        """Write ANSI text to the output pane."""
        self._output.append(text)

    def replace_header(self, text: str) -> None:
        """Replace the header at the top of output without duplicating it."""
        new_lines = text.split("\n")
        self._output._lines = new_lines + self._output._lines[self._header_line_count:]
        self._header_line_count = len(new_lines)
        self.invalidate()

    def set_stats(self, text: str) -> None:
        """Update the stats footer line (shown after/during AI response)."""
        self._stats_text = text
        self.invalidate()

    async def show_permission_dialog(self, action: str, detail: str) -> str:
        """Show in-TUI confirmation. Returns 'y', 'a', or 'n'."""
        loop = asyncio.get_event_loop()
        self._perm_action = action
        self._perm_detail = detail
        self._perm_sel = 0
        self._perm_future = loop.create_future()
        self.invalidate()
        try:
            return await self._perm_future
        finally:
            self._perm_future = None
            self.invalidate()

    async def show_ask_user_dialog(
        self, question: str, options: list[dict], recommended: str | None = None
    ) -> str | None:
        """Show an in-TUI question with selectable options (each a dict with
        "label" and "description") plus a free-text custom answer. Returns
        the chosen option's label / typed text, or None if the user
        dismissed the dialog (Esc/Ctrl+C) without answering."""
        loop = asyncio.get_event_loop()
        self._ask_question = question
        self._ask_options = options
        self._ask_recommended = recommended
        self._ask_sel = 0
        self._ask_custom_mode = False
        self._ask_future = loop.create_future()
        self.invalidate()
        try:
            return await self._ask_future
        finally:
            self._ask_future = None
            self._ask_custom_mode = False
            self.invalidate()

    def set_active_task(self, task: "asyncio.Task | None") -> None:
        """Track the currently running request task for Ctrl+C cancellation."""
        self._active_task = task

    def clear_active_task(self, task: "asyncio.Task") -> None:
        """Clear _active_task only if it's still THIS task — use from a
        task's own completion handler instead of set_active_task(None).

        A "/"-command (e.g. /settings) runs as its OWN tracked task even
        while a chat turn is already streaming (see cli.py:_enqueue —
        commands must run immediately, not wait behind the chat queue).
        Both call set_active_task(task) on start and used to call
        set_active_task(None) unconditionally on completion. Opening
        /settings mid-response and closing it again finishes the SETTINGS
        task first — its unconditional set_active_task(None) would clobber
        the still-running chat task's registration, leaving Ctrl+C with
        nothing to cancel for the rest of that turn (self._active_task
        stays None even though the chat task is still generating). Guarding
        on identity here fixes both directions of the race
        (command-finishes-first, and the reverse chat-finishes-first case
        the original set_active_task(None) call site never actually
        protected against either — see its comment history in cli.py)."""
        if self._active_task is task:
            self._active_task = None

    def set_recording_active(self, active: bool, stop_cb: "Callable[[], None] | None" = None) -> None:
        """Mark a mic recording as running (or finished) and register the
        callback Ctrl+C should invoke to stop it — see _interrupt above."""
        self._recording_active = active
        self._stop_recording_cb = stop_cb if active else None

    def set_music_active(self, active: bool, stop_cb: "Callable[[], None] | None" = None) -> None:
        """Mark a /music stream as running (or finished) and register the
        callback Ctrl+C should invoke to stop it — see _interrupt above."""
        self._music_active = active
        self._stop_music_cb = stop_cb if active else None

    def set_dnd_active(self, active: bool, exit_cb: "Callable[[], None] | None" = None) -> None:
        """Mark /dnd mode as entered (or exited) and register the callback
        a Ctrl+C with nothing running should invoke to leave it — see
        _interrupt above. Called from cli.py on /dnd new//dnd <id> (active),
        /dnd exit (inactive), and from exit_cb itself once it fires."""
        self._dnd_active = active
        self._dnd_exit_cb = exit_cb if active else None

    def set_gen3d_active(self, active: bool, stop_cb: "Callable[[], None] | None" = None) -> None:
        """Register the callback Ctrl+C should invoke to actually kill
        /gen_model's current subprocess (gen3d/pipeline.py's cancel_event.set())
        — see _interrupt above and this field's own comment in __init__ for
        why it piggybacks on _active_task.cancel() instead of getting its own
        priority branch like recording/music."""
        self._stop_gen3d_cb = stop_cb if active else None

    def set_queue_size(self, n: int) -> None:
        """Update pending-queue badge in the hints bar."""
        self._queue_size = n
        self.invalidate()

    def set_recap(self, text: str) -> None:
        """Update the recap footer line."""
        self._recap_text = text
        self.invalidate()

    def clear_output(self) -> None:
        """Clear the output pane."""
        self._output.clear()

    def set_submit_callback(self, cb: Callable[[str], Awaitable[None]]) -> None:
        """Set the coroutine to call when the user submits input."""
        self._submit_cb = cb

    def invalidate(self) -> None:
        """Request a redraw of the TUI."""
        try:
            self._app.invalidate()
        except Exception:
            pass

    async def run_async(self) -> None:
        """Run the application event loop."""
        await self._app.run_async()

    def exit(self) -> None:
        """Exit the application."""
        self._app.exit()

    def set_input_suggestion(self, text: str) -> None:
        """Show *text* as dismissable ghost text in the input box — rendered
        by prompt_toolkit's own AppendAutoSuggestion processor (already part
        of include_default_input_processors) via Buffer.suggestion, styled
        "class:auto-suggestion" below. Accept with Tab (see the "tab"
        keybinding); typing anything clears it for free, since
        Buffer._text_changed() unconditionally resets .suggestion on any
        edit — no separate "dismiss on keystroke" logic needed.
        Only offered on an EMPTY buffer — if the user is already mid-typing
        their own message, a stale suggestion for a different reply has
        nothing to do with what they're writing now."""
        if self._buffer.text:
            return
        self._buffer.suggestion = Suggestion(text)
        self.invalidate()

    def clear_input_suggestion(self) -> None:
        """Remove a pending ghost-text suggestion without waiting for the
        user to type (e.g. before showing a fresh one for the next turn)."""
        self._buffer.suggestion = None
        self.invalidate()

    @property
    def pt_app(self) -> Application:
        return self._app
