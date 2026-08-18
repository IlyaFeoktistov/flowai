"""mcp_agent/message_utils.py — _calls_by_id (single home for the
tool_call_id -> tc join, previously copy-pasted in 4 places) and the
repeated-identical-tool-call nudge in _dedupe_identical_tool_results (live
incident: Verifier looped on an identical failing `go get` call 5x with no
progress, see message_utils.py's comment on the nudge branch)."""
from conftest import ai_message, tool_message

from mcp_agent.message_utils import _calls_by_id, _dedupe_identical_tool_results


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
