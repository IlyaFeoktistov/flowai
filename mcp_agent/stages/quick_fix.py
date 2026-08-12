"""
Verdict/guidance для стадии QuickFix (mcp_agent/roles.py, mcp_agent/pipeline.py).

QuickFix — облегчённая ветка пайплайна для маленьких, однозначных правок
(router.py: needs_change=true и change_is_ambiguous=false): вместо
Analyzer->Planner->Coder эта стадия сама читает, что нужно, и сразу правит
— без отдельного плана и без ask_user (см. roles.py:executor_tools).
Verdict здесь — те же детерминированные
предикаты, что у coder_verdict (mcp_agent/stages/coder.py): были ли реально
внесены правки, не провалился ли сам вызов записи, есть ли финальный
отчёт — просто без формулировок про "план", которого в этой ветке нет.
"""
from mcp_agent.self_heal import _failed_write_messages, _has_successful_write, _wrote_code


def quick_fix_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    failed = _failed_write_messages(new_tool_msgs)
    # _has_successful_write guard: a failed attempt earlier in the round
    # that the model then retried and got right must not still fail the
    # whole round — only reject here when EVERY write attempt failed.
    if failed and not _has_successful_write(new_tool_msgs):
        return {
            "relevant": False,
            "reason": f"the `{failed[0].name}` call failed with a tool error — nothing was actually written",
        }
    if not _wrote_code(new_tool_msgs):
        return {
            "relevant": False,
            "reason": "no write/edit tool was called this round — a quick fix requires an actual code change, not just reading",
        }
    if not round_final_text.strip():
        return {
            "relevant": False,
            "reason": "edits were made but no final report was written — report what changed and why it fixes the issue",
        }
    return {"relevant": True, "reason": "investigated and applied the fix directly, reported what changed"}


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
            "need and apply the change directly with replace_lines/"
            "insert_lines/edit_file/write_file this round. If it turns out "
            "the request is actually broader/more ambiguous than it looked, "
            "say so plainly in your report instead of forcing an edit — don't "
            "keep reading without ever writing or explaining why."
        )
    return "Write a final report describing what you changed and why it fixes the issue."
