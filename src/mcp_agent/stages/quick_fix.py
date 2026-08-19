"""
Verdict/guidance для стадии QuickFix (mcp_agent/roles.py, mcp_agent/pipeline.py).

QuickFix — облегчённая ветка пайплайна для маленьких, однозначных правок
(router.py: needs_change=true и change_is_ambiguous=false): вместо
Analyzer->Planner->Coder эта стадия сама читает, что нужно, и сразу правит
— без отдельного плана и без ask_user (см. roles.py:executor_tools), а
теперь и сама проверяет результат реальной командой (bash — только для
запуска проверок) прежде чем отдавать Verifier'у. Verdict здесь — те же
детерминированные предикаты, что у coder_verdict (mcp_agent/stages/
coder.py): были ли реально внесены правки, проверены ли они реальной
командой, есть ли финальный отчёт — просто без формулировок про "план",
которого в этой ветке нет.
"""
from mcp_agent.self_heal import _failed_write_messages, _last_check_error, _wrote_code, _write_stage_outcome


def quick_fix_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    outcome = _write_stage_outcome(new_tool_msgs, round_final_text, round_msgs)
    if outcome == "failed_write":
        failed = _failed_write_messages(new_tool_msgs)
        return {
            "relevant": False,
            "reason": f"the `{failed[0].name}` call failed with a tool error — nothing was actually written",
        }
    if outcome == "no_write":
        return {
            "relevant": False,
            "reason": "no write/edit tool was called this round — a quick fix requires an actual code change, not just reading",
        }
    if outcome == "no_report":
        return {
            "relevant": False,
            "reason": "edits were made but no final report was written — report what changed and why it fixes the issue",
        }
    if outcome == "not_verified":
        return {
            "relevant": False,
            "reason": "edits were made but no real check (build/test/run) was run via bash — a write only means the file was saved, not that it works",
        }
    if outcome == "execution_failure":
        return {
            "relevant": False,
            "reason": "ran a real check and it failed",
            "kind": "execution_failure",
        }
    return {"relevant": True, "reason": "investigated, applied the fix, verified it, and reported what changed"}


def quick_fix_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    failed = _failed_write_messages(new_tool_msgs)
    if failed:
        errors = "\n".join(f"- {m.name}: {str(m.content)[:500]}" for m in failed)
        return (
            "Your write/edit call failed — nothing was written, don't treat it "
            "as done. The error below already shows the file's ACTUAL current "
            "content — use it directly to fix your arguments and call the SAME "
            f"write again THIS round:\n{errors}"
        )
    if not _wrote_code(new_tool_msgs):
        return (
            "You haven't made any edits yet. This was routed here specifically "
            "because it's a narrow, unambiguous fix — you don't need a "
            "separate plan or confirmation, just read what you genuinely still "
            "need and apply the change directly with edit_file/write_file "
            "this round. If it turns out the request is actually broader/more "
            "ambiguous than it looked, say so plainly in your report instead "
            "of forcing an edit — don't keep reading without ever writing or "
            "explaining why."
        )
    if verdict.get("kind") == "execution_failure":
        error_text = _last_check_error(round_msgs)
        return (
            "Your own check failed — read the actual error below and fix "
            "the code, then run the SAME check again to confirm before "
            f"reporting done:\n{error_text}"
        )
    if not round_final_text.strip():
        return "Write a final report describing what you changed and why it fixes the issue."
    return (
        "You wrote code but never ran a real check (build/test/run) via "
        "bash to confirm it actually works — a successful write only means "
        "the file was saved. Run the appropriate check now; if it fails, "
        "fix the code and check again before reporting done."
    )
