"""Collapsible tool-output blocks (ui/app.py:_OutputControl.reserve_fold/
fill_fold/toggle_fold/mouse_handler) — no real terminal needed,
create_content() and mouse_handler() are plain methods that can be driven
directly with synthetic mouse events."""
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.data_structures import Point

from ui.app import _OutputControl


def _click(ctrl, x, y):
    down = MouseEvent(Point(x=x, y=y), MouseEventType.MOUSE_DOWN, MouseButton.LEFT, frozenset())
    up = MouseEvent(Point(x=x, y=y), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset())
    ctrl.mouse_handler(down)
    ctrl.mouse_handler(up)


def _hover(ctrl, x, y):
    move = MouseEvent(Point(x=x, y=y), MouseEventType.MOUSE_MOVE, MouseButton.NONE, frozenset())
    ctrl.mouse_handler(move)


def test_reserve_fold_starts_empty_and_not_ready():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    assert ctrl._lines == ["header line", ""]  # no placeholder written yet
    assert fold.ready is False
    assert fold.start == fold.end == 1


def test_fill_fold_writes_arrow_and_makes_it_ready():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.fill_fold(fold, "header line", ["e0", "e1", "e2"])
    assert fold.ready is True
    assert ctrl._lines == ["header line ▸", ""]


def test_click_before_ready_is_a_noop():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.create_content(width=80, height=10)
    _click(ctrl, x=0, y=0)
    assert fold.is_expanded is False
    assert ctrl._lines == ["header line", ""]  # untouched — nothing to show yet


def test_click_on_trigger_line_expands_and_collapses_body():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.fill_fold(fold, "header line", ["e0", "e1", "e2"])
    ctrl.create_content(width=80, height=10)

    _click(ctrl, x=0, y=0)  # click the trigger line itself
    assert fold.is_expanded is True
    assert ctrl._lines == ["header line ▾", "e0", "e1", "e2", ""]

    ctrl.create_content(width=80, height=10)
    _click(ctrl, x=0, y=0)
    assert fold.is_expanded is False
    assert ctrl._lines == ["header line ▸", ""]


def test_click_outside_fold_does_not_toggle():
    ctrl = _OutputControl()
    ctrl.append("plain line\n")
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=1)
    ctrl.fill_fold(fold, "header line", ["e0"])
    ctrl.create_content(width=80, height=10)

    _click(ctrl, x=0, y=0)  # "plain line" — not part of any fold
    assert fold.is_expanded is False


def test_hover_ignores_not_ready_fold():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.create_content(width=80, height=10)

    _hover(ctrl, x=0, y=0)
    assert ctrl._hover_fold is None  # not ready — never offered as hoverable


def test_hover_on_ready_trigger_line_highlights_fold():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.fill_fold(fold, "header line", ["e0", "e1"])
    ctrl.create_content(width=80, height=10)

    _hover(ctrl, x=0, y=0)
    assert ctrl._hover_fold is fold

    _hover(ctrl, x=0, y=1)  # the still-empty reopened line, not the fold
    assert ctrl._hover_fold is None


def test_clear_resets_folds_and_hover():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    fold = ctrl.reserve_fold(trigger_line=0)
    ctrl.fill_fold(fold, "header line", ["e0"])
    ctrl.create_content(width=80, height=10)
    _hover(ctrl, x=0, y=0)
    assert ctrl._folds and ctrl._hover_fold is not None

    ctrl.clear()
    assert ctrl._folds == []
    assert ctrl._hover_fold is None


def test_parallel_tools_result_lands_under_its_own_header_regardless_of_completion_order():
    """The bug this whole reserve/fill split exists to fix: a message with
    several INDEPENDENT tool calls (see mcp_agent/prompts.py's analyzer
    prompt encouraging exactly this) prints every header before any of
    them necessarily finishes — completion order can differ from call
    order. A result must still land right after ITS OWN header, not
    wherever the buffer happens to end when its tool_end happens to fire."""
    ctrl = _OutputControl()

    # tool_start x3, back-to-back — mirrors on_event's real sequence: each
    # call reserves its fold BEFORE the next header even prints.
    ctrl.append("\n")
    h1 = len(ctrl._lines) - 1
    ctrl.append("● tool one")
    fold1 = ctrl.reserve_fold(trigger_line=h1)
    ctrl.append("\n\n")
    h2 = len(ctrl._lines) - 1
    ctrl.append("● tool two")
    fold2 = ctrl.reserve_fold(trigger_line=h2)
    ctrl.append("\n\n")
    h3 = len(ctrl._lines) - 1
    ctrl.append("● tool three")
    fold3 = ctrl.reserve_fold(trigger_line=h3)
    ctrl.append("\n")

    # Completion order is 3, 1, 2 — deliberately NOT the call order.
    ctrl.fill_fold(fold3, "● tool three", ["result three"])
    ctrl.fill_fold(fold1, "● tool one", ["result one"])
    ctrl.fill_fold(fold2, "● tool two", ["result two"])

    ctrl.create_content(width=80, height=20)
    _click(ctrl, x=0, y=fold1.trigger_line)
    ctrl.create_content(width=80, height=20)
    _click(ctrl, x=0, y=fold2.trigger_line)
    ctrl.create_content(width=80, height=20)
    _click(ctrl, x=0, y=fold3.trigger_line)

    text = "\n".join(ctrl._lines)
    assert text.index("tool one") < text.index("result one") < text.index("tool two")
    assert text.index("tool two") < text.index("result two") < text.index("tool three")
    assert text.index("tool three") < text.index("result three")
