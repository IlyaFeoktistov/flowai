"""Collapsible tool-output blocks (ui/app.py:_OutputControl.append_fold/
toggle_fold/mouse_handler) — no real terminal needed, create_content() and
mouse_handler() are plain methods that can be driven directly with
synthetic mouse events."""
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


def test_append_fold_inserts_collapsed_and_reopens():
    ctrl = _OutputControl()
    ctrl.append("before\n")
    ctrl.append_fold(["c0", "c1", "marker"], ["e0", "e1", "e2", "e3", "marker2"])
    assert ctrl._lines == ["before", "c0", "c1", "marker", ""]
    assert len(ctrl._folds) == 1
    assert ctrl._folds[0].start == 1
    assert ctrl._folds[0].end == 4


def test_toggle_fold_expands_and_shifts_later_folds():
    ctrl = _OutputControl()
    ctrl.append_fold(["a0", "a-marker"], ["a0", "a1", "a2", "a-marker2"])
    ctrl.append("between\n")
    ctrl.append_fold(["b0", "b-marker"], ["b0", "b1", "b-marker2"])

    fold_a, fold_b = ctrl._folds
    assert fold_a.start == 0 and fold_a.end == 2
    assert fold_b.start == 3 and fold_b.end == 5

    ctrl.toggle_fold(fold_a)
    assert fold_a.is_expanded is True
    assert ctrl._lines[0:4] == ["a0", "a1", "a2", "a-marker2"]
    # fold_b shifted by +2 (4 expanded lines - 2 collapsed lines)
    assert fold_b.start == 5 and fold_b.end == 7
    assert ctrl._lines[5:7] == ["b0", "b-marker"]

    ctrl.toggle_fold(fold_a)
    assert fold_a.is_expanded is False
    assert fold_b.start == 3


def test_fold_at_logical():
    ctrl = _OutputControl()
    ctrl.append_fold(["c0", "c1", "marker"], ["e0", "e1", "e2", "marker2"])
    fold = ctrl._folds[0]
    assert ctrl._fold_at_logical(0) is fold
    assert ctrl._fold_at_logical(2) is fold
    assert ctrl._fold_at_logical(3) is None
    assert ctrl._fold_at_logical(None) is None


def test_click_toggles_fold_under_cursor():
    ctrl = _OutputControl()
    ctrl.append_fold(["c0", "c1", "marker"], ["e0", "e1", "e2", "marker2"])
    ctrl.create_content(width=80, height=10)  # populates _render_* state
    fold = ctrl._folds[0]
    assert fold.is_expanded is False

    _click(ctrl, x=5, y=1)  # row 1 within the viewport == logical line 1 (c1)
    assert fold.is_expanded is True

    ctrl.create_content(width=80, height=10)  # re-render after expand
    _click(ctrl, x=0, y=0)
    assert fold.is_expanded is False


def test_click_outside_fold_does_not_toggle():
    ctrl = _OutputControl()
    ctrl.append("plain line\n")
    ctrl.append_fold(["c0", "marker"], ["e0", "e1", "marker2"])
    ctrl.create_content(width=80, height=10)
    fold = ctrl._folds[0]

    _click(ctrl, x=0, y=0)  # "plain line" — not part of any fold
    assert fold.is_expanded is False


def test_hover_sets_and_clears_hover_fold():
    ctrl = _OutputControl()
    ctrl.append("plain\n")
    ctrl.append_fold(["c0", "marker"], ["e0", "e1", "marker2"])
    ctrl.create_content(width=80, height=10)
    fold = ctrl._folds[0]

    _hover(ctrl, x=0, y=1)  # row 1 == "c0", inside the fold
    assert ctrl._hover_fold is fold

    _hover(ctrl, x=0, y=0)  # row 0 == "plain", outside the fold
    assert ctrl._hover_fold is None


def test_clear_resets_folds_and_hover():
    ctrl = _OutputControl()
    ctrl.append_fold(["c0", "marker"], ["e0", "e1", "marker2"])
    ctrl.create_content(width=80, height=10)
    _hover(ctrl, x=0, y=0)
    assert ctrl._folds and ctrl._hover_fold is not None

    ctrl.clear()
    assert ctrl._folds == []
    assert ctrl._hover_fold is None


def test_triggered_fold_starts_fully_hidden():
    """collapsed=[] (the default shape for a tool result, see
    ui/stream.py:_print_foldable_body) — nothing is shown for the body at
    all, and no placeholder line is inserted; the trigger line gets the
    collapsed arrow appended."""
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    ctrl.append_fold([], ["e0", "e1", "e2"], trigger_line=0, trigger_text="header line")
    assert ctrl._lines == ["header line ▸", ""]
    fold = ctrl._folds[0]
    assert fold.start == fold.end == 1  # zero-length body range while collapsed
    assert fold.trigger_line == 0


def test_click_on_trigger_line_expands_and_collapses_body():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    ctrl.append_fold([], ["e0", "e1", "e2"], trigger_line=0, trigger_text="header line")
    ctrl.create_content(width=80, height=10)

    _click(ctrl, x=0, y=0)  # click the trigger line itself
    fold = ctrl._folds[0]
    assert fold.is_expanded is True
    assert ctrl._lines == ["header line ▾", "e0", "e1", "e2", ""]

    ctrl.create_content(width=80, height=10)
    _click(ctrl, x=0, y=0)
    assert fold.is_expanded is False
    assert ctrl._lines == ["header line ▸", ""]


def test_triggered_fold_shifts_later_content_on_expand():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    ctrl.append_fold([], ["e0", "e1"], trigger_line=0, trigger_text="header line")
    ctrl.append("after\n")
    ctrl.create_content(width=80, height=10)

    _click(ctrl, x=0, y=0)
    assert ctrl._lines == ["header line ▾", "e0", "e1", "after", ""]


def test_hover_on_trigger_line_highlights_fold():
    ctrl = _OutputControl()
    ctrl.append("header line\n")
    ctrl.append_fold([], ["e0", "e1"], trigger_line=0, trigger_text="header line")
    ctrl.create_content(width=80, height=10)
    fold = ctrl._folds[0]

    _hover(ctrl, x=0, y=0)  # row 0 == the trigger line
    assert ctrl._hover_fold is fold

    _hover(ctrl, x=0, y=1)  # row 1 == the still-empty reopened line, not the fold
    assert ctrl._hover_fold is None
