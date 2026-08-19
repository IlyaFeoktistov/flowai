"""ui/stream.py:_truncate_label_to_width — caps the footer's variable
label so label + _ai_header's fixed "N tok · Xm Ys" suffix can never
exceed the terminal width in the raw-terminal (non-Textual) fallback
branch of _footer_loop. Without this, a long label wraps onto the next
row, which the fallback's single-line `\033[K` clear never touches — a
shorter later render then leaves stale characters from the wrapped tail
bleeding into the redraw."""
from ui.stream import _FOOTER_FIXED_SUFFIX_WIDTH, _truncate_label_to_width


def test_short_label_is_left_untouched():
    assert _truncate_label_to_width("⠏ Coder · пишу код", 80) == "⠏ Coder · пишу код"


def test_long_label_is_truncated_with_an_ellipsis():
    label = "⠏ Coder · " + "x" * 100
    out = _truncate_label_to_width(label, 80)
    budget = 80 - _FOOTER_FIXED_SUFFIX_WIDTH
    assert len(out) == budget
    assert out.endswith("…")


def test_narrow_terminal_never_shrinks_the_budget_below_the_floor():
    out = _truncate_label_to_width("x" * 100, 10)
    assert len(out) == 8
