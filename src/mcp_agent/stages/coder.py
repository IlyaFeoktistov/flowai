"""
Verdict/guidance для стадии Coder (mcp_agent/roles.py, mcp_agent/pipeline.py).

Coder получает утверждённый нумерованный план и должен его исполнить —
верификации здесь нет вообще (это отдельная стадия, Verifier), так что
verdict проверяет только "были ли реально сделаны правки" — те же
детерминированные предикаты self_heal.py, что раньше жили в общем дереве
mcp_agent/agent.py, просто без всей верификационной части (execution_
failure/syntax_only и т.д. — то, что здесь БЫЛО про bash, целиком
переехало в mcp_agent/stages/verifier.py)."""
from mcp_agent.self_heal import _failed_write_messages, _wrote_code, _write_stage_outcome


def coder_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    outcome = _write_stage_outcome(new_tool_msgs, round_final_text)
    if outcome == "failed_write":
        failed = _failed_write_messages(new_tool_msgs)
        return {
            "relevant": False,
            "reason": f"the `{failed[0].name}` call failed with a tool error — nothing was actually written, the plan isn't done",
        }
    if outcome == "no_write":
        return {
            "relevant": False,
            "reason": "no write/edit tool was called this round — the plan requires actual code changes, not just reading",
        }
    if outcome == "no_report":
        return {
            "relevant": False,
            "reason": "edits were made but no final report was written — report what changed for each numbered plan step",
        }
    return {"relevant": True, "reason": "applied edits and reported what changed"}


def coder_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    failed = _failed_write_messages(new_tool_msgs)
    if failed:
        errors = "\n".join(f"- {m.name}: {str(m.content)[:500]}" for m in failed)
        # A vaguer guidance like "read the error and fix your arguments" is
        # technically correct but vague enough that the next attempt can
        # spend its ENTIRE round re-reading files and never retry the
        # write at all (coder_verdict then fails it a SECOND time for "no
        # write/edit tool was called this round"). edit_file's own error
        # already says exactly what's wrong (old_string not found, or
        # found more than once and needs more context/replace_all) —
        # that's normally enough to fix the call directly without a fresh
        # read.
        return (
            "Your write/edit call failed — nothing was written, don't treat it "
            "as done. The error below already tells you exactly what's wrong "
            "with old_string (not found, or not unique) — fix it directly and "
            "call the SAME edit again THIS round. Only re-read the file first "
            "if the error doesn't already give you what you need — don't spend "
            f"the whole round just reading without retrying the write:\n{errors}"
        )
    if not _wrote_code(new_tool_msgs):
        return (
            "You didn't make any edits this round. Follow the numbered plan "
            "literally — use edit_file/write_file to apply each step, in "
            "order, don't just re-read files you already have from the "
            "plan/Analyzer's findings above."
        )
    return "Write a final report describing what you changed, numbered 1:1 with the plan's steps."
