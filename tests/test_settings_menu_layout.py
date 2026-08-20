"""ui/tui/settings.py's /settings screen — live-caught bug: a toggle's
hint was drawn with curses.addstr(y, xv + 7, hint, ...) with NO bound on
its length. On anything narrower than a very wide terminal, a long hint
either wrapped onto the settings row BELOW it (visually merging two
different settings' text together) or raised curses.error partway
through the write (caught elsewhere, leaving an arbitrary mid-word cut).
Fixed two ways: _fit() bounds every drawn value/hint to the room actually
available before addstr ever sees it, and the worst-offending
_TOGGLE_HINTS entries were rewritten short (a rambling paragraph doesn't
fit ANY single terminal row well, regardless of bounding)."""
from ui.tui.settings import _ITEMS, _LABEL_COL_WIDTH, _TOGGLE_HINTS, _fit


def test_fit_returns_text_unchanged_when_it_already_fits():
    assert _fit("short", 10) == "short"


def test_fit_truncates_with_ellipsis_when_too_long():
    result = _fit("a much longer string than allowed", 10)
    assert len(result) == 10
    assert result.endswith("…")


def test_fit_handles_zero_or_negative_room():
    assert _fit("anything", 0) == ""
    assert _fit("anything", -5) == ""


def test_fit_handles_room_of_exactly_one():
    assert _fit("anything", 1) == "…"


def test_label_column_width_is_reasonable():
    # Regression guard for the concrete bug: a label like "оптимизированные
    # тулы (урезанный список для всех агентов)" (57 chars) drove
    # _LABEL_COL_WIDTH so wide that almost nothing was left for the hint on
    # a normal terminal. No single label should need to be a full sentence.
    assert _LABEL_COL_WIDTH <= 35
    for label, _key, _kind in _ITEMS:
        assert len(label) <= 33, f"label too long, will bloat _LABEL_COL_WIDTH: {label!r}"


def test_every_toggle_key_referenced_in_items_has_a_hint_or_is_self_explanatory():
    # Not every toggle NEEDS a hint (e.g. a self-explanatory label), but
    # every hint that exists must correspond to a real toggle in _ITEMS —
    # otherwise it's dead text nobody will ever see.
    toggle_keys = {key for _label, key, kind in _ITEMS if kind == "toggle"}
    for key in _TOGGLE_HINTS:
        assert key in toggle_keys, f"_TOGGLE_HINTS has an entry for {key!r}, which isn't a toggle in _ITEMS"


def test_requires_annotation_only_on_hints_that_actually_gate_on_a_prerequisite():
    # "(Требуется X)" should be reserved for a genuine prerequisite — not
    # sprinkled everywhere. Sanity bound so it doesn't creep back into
    # every hint like the old rambling paragraphs did.
    with_requires = [k for k, h in _TOGGLE_HINTS.items() if "Требуется" in h]
    assert 1 <= len(with_requires) <= 5
