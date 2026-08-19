"""self_heal.py:_write_stage_outcome (shared classification behind
stages/coder.py and stages/quick_fix.py) plus the two verdict functions
themselves — confirms both roles still produce byte-identical output to
before the extraction (see stages/coder.py's git history), differing only
in wording, never in the decision itself."""
from langchain_core.messages import AIMessage

from conftest import write_round

from mcp_agent.self_heal import _write_stage_outcome
from mcp_agent.stages.coder import coder_verdict
from mcp_agent.stages.quick_fix import quick_fix_verdict


def _new_tool_msgs(round_msgs):
    from langchain_core.messages import ToolMessage
    return [m for m in round_msgs if isinstance(m, ToolMessage)]


def test_outcome_no_write_when_nothing_written():
    assert _write_stage_outcome([], "some text") == "no_write"


def test_outcome_failed_write_when_only_attempt_errored():
    round_msgs = write_round(write_ok=False)
    assert _write_stage_outcome(_new_tool_msgs(round_msgs), "report") == "failed_write"


def test_outcome_ok_when_a_later_retry_succeeded_despite_earlier_failure():
    # _has_successful_write guard: one failed attempt followed by a real
    # success in the SAME round must not be flagged as failed_write.
    from conftest import ai_message, tool_message
    round_msgs = [
        ai_message([{"id": "w1", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", content="Error: bad", status="error", tool_call_id="w1"),
        ai_message([{"id": "w2", "name": "write_file", "args": {"path": "x.py"}}]),
        tool_message("write_file", content="ok", tool_call_id="w2"),
    ]
    assert _write_stage_outcome(_new_tool_msgs(round_msgs), "fixed it") == "ok"


def test_outcome_no_report_when_wrote_but_no_final_text():
    round_msgs = write_round(final_text="")
    round_msgs = [m for m in round_msgs if not (isinstance(m, AIMessage) and m.content == "" and not m.tool_calls)]
    assert _write_stage_outcome(_new_tool_msgs(round_msgs), "") == "no_report"


def test_outcome_ok_when_wrote_and_reported_and_round_msgs_omitted():
    # round_msgs=None (default) skips the verification checks entirely —
    # used by callers/tests that don't care about them.
    round_msgs = write_round(final_text="done, wrote x.py")
    assert _write_stage_outcome(_new_tool_msgs(round_msgs), "done, wrote x.py") == "ok"


def test_outcome_not_verified_when_wrote_but_never_ran_bash():
    round_msgs = write_round(final_text="done, wrote x.py")  # bash_ok=None — no bash call at all
    outcome = _write_stage_outcome(_new_tool_msgs(round_msgs), "done, wrote x.py", round_msgs)
    assert outcome == "not_verified"


def test_outcome_execution_failure_when_the_real_check_failed():
    round_msgs = write_round(bash_ok=False, final_text="done, wrote x.py")
    outcome = _write_stage_outcome(_new_tool_msgs(round_msgs), "done, wrote x.py", round_msgs)
    assert outcome == "execution_failure"


def test_outcome_ok_when_wrote_reported_and_verified():
    round_msgs = write_round(bash_ok=True, final_text="done, wrote x.py")
    outcome = _write_stage_outcome(_new_tool_msgs(round_msgs), "done, wrote x.py", round_msgs)
    assert outcome == "ok"


def test_coder_and_quick_fix_verdicts_agree_on_outcome_differ_only_in_wording():
    round_msgs = write_round(write_ok=False)
    new_tool_msgs = _new_tool_msgs(round_msgs)
    coder_v = coder_verdict(round_msgs, new_tool_msgs, "report")
    qf_v = quick_fix_verdict(round_msgs, new_tool_msgs, "report")
    assert coder_v["relevant"] is False and qf_v["relevant"] is False
    assert "nothing was actually written" in coder_v["reason"]
    assert "nothing was actually written" in qf_v["reason"]
    assert "the plan isn't done" in coder_v["reason"]  # coder-specific wording
    assert "the plan isn't done" not in qf_v["reason"]  # quick_fix has no plan


def test_coder_and_quick_fix_verdicts_reject_an_unverified_round():
    round_msgs = write_round(final_text="applied the fix")  # no bash call
    new_tool_msgs = _new_tool_msgs(round_msgs)
    coder_v = coder_verdict(round_msgs, new_tool_msgs, "applied the fix")
    qf_v = quick_fix_verdict(round_msgs, new_tool_msgs, "applied the fix")
    assert coder_v["relevant"] is False and qf_v["relevant"] is False
    assert "no real check" in coder_v["reason"]
    assert "no real check" in qf_v["reason"]


def test_coder_and_quick_fix_verdicts_reject_a_failed_check_with_execution_failure_kind():
    round_msgs = write_round(bash_ok=False, final_text="applied the fix")
    new_tool_msgs = _new_tool_msgs(round_msgs)
    coder_v = coder_verdict(round_msgs, new_tool_msgs, "applied the fix")
    qf_v = quick_fix_verdict(round_msgs, new_tool_msgs, "applied the fix")
    assert coder_v["relevant"] is False and coder_v["kind"] == "execution_failure"
    assert qf_v["relevant"] is False and qf_v["kind"] == "execution_failure"


def test_coder_and_quick_fix_verdicts_both_succeed_on_a_verified_clean_round():
    round_msgs = write_round(bash_ok=True, final_text="applied the fix")
    new_tool_msgs = _new_tool_msgs(round_msgs)
    assert coder_verdict(round_msgs, new_tool_msgs, "applied the fix")["relevant"] is True
    assert quick_fix_verdict(round_msgs, new_tool_msgs, "applied the fix")["relevant"] is True
