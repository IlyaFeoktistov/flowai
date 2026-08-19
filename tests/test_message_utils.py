"""mcp_agent/message_utils.py — _calls_by_id (single home for the
tool_call_id -> tc join, previously copy-pasted in 4 places) and the two
repeated-tool-call nudges in _dedupe_identical_tool_results: the
exact-content one, which stops a role from looping on an identical failing
call (e.g. `go get`) with no progress, and the args-only "thrashing" one,
which catches the same command failing DIFFERENTLY each time (e.g. `go
build` after a different bad cast each round) — see message_utils.py's
comments on both branches."""
from conftest import ai_message, tool_message

from mcp_agent.message_utils import _calls_by_id, _dedupe_identical_tool_results, _find_call_by_id


def test_calls_by_id_joins_tool_call_id_to_the_call_dict():
    msgs = [
        ai_message([{"id": "1", "name": "read_file", "args": {"path": "a.py"}}]),
        tool_message("read_file", tool_call_id="1"),
    ]
    calls = _calls_by_id(msgs)
    assert calls["1"]["name"] == "read_file"
    assert calls["1"]["args"] == {"path": "a.py"}


def test_calls_by_id_ignores_tool_calls_with_a_falsy_id():
    msgs = [ai_message([{"id": "", "name": "read_file", "args": {}}])]
    assert _calls_by_id(msgs) == {}


def test_calls_by_id_empty_for_no_ai_messages():
    assert _calls_by_id([tool_message("bash", tool_call_id="1")]) == {}


def test_find_call_by_id_returns_the_message_with_all_its_sibling_calls():
    msgs = [
        ai_message([
            {"id": "1", "name": "ask_user", "args": {}},
            {"id": "2", "name": "read_file", "args": {"path": "a.py"}},
        ], content="checking in"),
        tool_message("ask_user", tool_call_id="1"),
    ]
    found = _find_call_by_id(msgs, "2")
    assert found is not None
    assert found.content == "checking in"
    assert {tc["name"] for tc in found.tool_calls} == {"ask_user", "read_file"}


def test_find_call_by_id_returns_none_when_not_found():
    assert _find_call_by_id([tool_message("bash", tool_call_id="1")], "1") is None


def _bash_round(n, result_text):
    return [
        ai_message([{"id": f"c{n}", "name": "bash", "args": {"command": "go get pkg@bad"}}]),
        tool_message("bash", content=result_text, tool_call_id=f"c{n}"),
    ]


def test_older_identical_repeats_are_collapsed_to_a_placeholder():
    msgs = _bash_round(0, "same error") + _bash_round(1, "same error") + _bash_round(2, "same error")
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert "collapsed here to save context" in tool_msgs[0].content
    assert "collapsed here to save context" in tool_msgs[1].content


def test_newest_identical_repeat_carries_the_stop_repeating_hint():
    msgs = _bash_round(0, "same error") + _bash_round(1, "same error") + _bash_round(2, "same error")
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    last = tool_msgs[-1].content
    assert "same error" in last  # real content preserved, not replaced
    assert "3 times this turn" in last
    assert "will not produce a different outcome" in last


def test_single_occurrence_is_left_untouched():
    msgs = _bash_round(0, "unique result")
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert tool_msgs[0].content == "unique result"


def test_list_shaped_mcp_content_is_joined_cleanly_not_escaped():
    content = [{"type": "text", "text": "line1\nline2"}]
    msgs = [
        ai_message([{"id": "c0", "name": "bash", "args": {"command": "x"}}]),
        tool_message("bash", content=content, tool_call_id="c0"),
        ai_message([{"id": "c1", "name": "bash", "args": {"command": "x"}}]),
        tool_message("bash", content=content, tool_call_id="c1"),
    ]
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    last = tool_msgs[-1].content
    assert "line1\nline2" in last  # real newline, not escaped '\\n'
    assert "\\n" not in last.split("line1\nline2")[0]


def test_different_results_for_same_call_are_not_flagged_as_repeats():
    # e.g. re-running a test after a fix — legitimately different content,
    # both copies must stay intact (see this function's own docstring on
    # why bash is never blocked from re-running).
    msgs = _bash_round(0, "fail") + _bash_round(1, "pass")
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert tool_msgs[0].content == "fail"
    assert tool_msgs[1].content == "pass"


def test_same_command_failing_differently_each_time_gets_a_thrashing_hint():
    # Coder tries a different cast each round, `go build` fails with a
    # DIFFERENT error every time — the exact-content dedup above never
    # matches (content differs), so this needs the separate name+args-only
    # counter to catch it.
    msgs = (
        _bash_round(0, "Error (exit 2): cannot use x (int)")
        + _bash_round(1, "Error (exit 2): cannot use x (float64)")
        + _bash_round(2, "Error (exit 2): cannot use x (string)")
    )
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert "cannot use x (string)" in tool_msgs[-1].content  # real content kept
    assert "3 times this turn" in tool_msgs[-1].content
    assert "getting a DIFFERENT result" in tool_msgs[-1].content
    # earlier, differently-failing attempts are untouched by this nudge —
    # only exact-content dedup (a different mechanism) ever rewrites those
    assert "cannot use x (int)" in tool_msgs[0].content
    assert "cannot use x (float64)" in tool_msgs[1].content


def test_thrashing_hint_does_not_fire_below_the_threshold():
    msgs = (
        _bash_round(0, "Error (exit 2): cannot use x (int)")
        + _bash_round(1, "Error (exit 2): cannot use x (float64)")
    )
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert tool_msgs[1].content == "Error (exit 2): cannot use x (float64)"


def test_thrashing_hint_does_not_fire_once_the_command_finally_passes():
    # Real convergence, not thrashing — the third attempt succeeds, so no
    # "you're not converging" hint should be tacked onto a clean result.
    msgs = (
        _bash_round(0, "Error (exit 2): cannot use x (int)")
        + _bash_round(1, "Error (exit 2): cannot use x (float64)")
        + _bash_round(2, "build succeeded")
    )
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert tool_msgs[-1].content == "build succeeded"


def test_thrashing_hint_is_scoped_to_loop_prone_tools_only():
    # write_file/edit_file already have their own failure-classification
    # path (self_heal.py) — this nudge must not double up on those.
    msgs = [
        ai_message([{"id": "w0", "name": "write_file", "args": {"path": "a.py"}}]),
        tool_message("write_file", content="Error: old_string not found", tool_call_id="w0"),
        ai_message([{"id": "w1", "name": "write_file", "args": {"path": "a.py"}}]),
        tool_message("write_file", content="Error: old_string not unique", tool_call_id="w1"),
        ai_message([{"id": "w2", "name": "write_file", "args": {"path": "a.py"}}]),
        tool_message("write_file", content="Error: still not found", tool_call_id="w2"),
    ]
    out = _dedupe_identical_tool_results(msgs)
    tool_msgs = [m for m in out if hasattr(m, "tool_call_id")]
    assert tool_msgs[-1].content == "Error: still not found"
