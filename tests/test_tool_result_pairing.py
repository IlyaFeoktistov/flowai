"""StreamDisplay.on_event's tool_start/tool_end pairing (ui/stream.py) —
live-caught bug: _pending_tool_calls used to be a plain FIFO queue
(list.pop(0)), which silently mispairs a tool_start with the WRONG
tool_end whenever they don't arrive in strict alternating order — exactly
what delegate's nested sub-agent stream does in practice (its own
tool_start/tool_end interleave with the outer conversation's, see
delegate_tool.py's module docstring on this cross-talk). Fixed by keying
_pending_tool_calls by the event's own "id" (tool_call_id) instead of
queue position."""
import pytest

from ui.app import _OutputControl
from ui.stream import StreamDisplay


class _FakeApp:
    """Just enough of FlowAIApp for on_event's tool_start/tool_end branch —
    a real _OutputControl (already covered by test_output_control_fold.py)
    plus a no-op invalidate(). One line pre-seeded so reserve_fold's
    line_idx (len(_lines) - 1) is a valid, non-negative index — on_event
    normally relies on the app's own console output already having grown
    _lines by the time a tool_start fires; nothing in this fake harness
    does that on its own."""

    def __init__(self):
        self._output = _OutputControl()
        self._output.append("\n")

    def invalidate(self):
        pass


def _display():
    app = _FakeApp()
    return StreamDisplay(session_stats={"tools_called": 0}, app=app), app


@pytest.mark.asyncio
async def test_tool_end_pairs_by_id_not_arrival_order():
    """Two tool calls started back to back, but their tool_end events
    arrive in the OPPOSITE order (call B finishes before call A) — each
    result must still land under its OWN header, matched by id."""
    display, app = _display()

    await display.on_event({"type": "tool_start", "name": "grep_search", "args": {"pattern": "A"}, "id": "call-a"})
    await display.on_event({"type": "tool_start", "name": "list_memory", "args": {}, "id": "call-b"})

    # Out of order on purpose: B's tool_end arrives first.
    await display.on_event({"type": "tool_end", "name": "list_memory", "result": "facts:\n- fact one", "id": "call-b"})
    await display.on_event({"type": "tool_end", "name": "grep_search", "result": "a.py\nb.py", "id": "call-a"})

    folds = app._output._folds
    assert len(folds) == 2
    fold_a, fold_b = folds
    assert any("a.py" in ln for ln in fold_a.expanded)
    assert any("fact one" in ln for ln in fold_b.expanded)
    assert not any("fact one" in ln for ln in fold_a.expanded)
    assert not any("a.py" in ln for ln in fold_b.expanded)


@pytest.mark.asyncio
async def test_tool_end_without_id_falls_back_to_oldest_pending():
    """Defensive fallback for a caller that genuinely has no id (shouldn't
    happen after every real emission site was given one) — degrades to
    the old FIFO behavior instead of losing the result outright."""
    display, app = _display()

    await display.on_event({"type": "tool_start", "name": "read_file", "args": {}, "id": "call-a"})
    await display.on_event({"type": "tool_end", "name": "read_file", "result": "content"})  # no id

    fold_a = app._output._folds[0]
    assert any("content" in ln for ln in fold_a.expanded)
