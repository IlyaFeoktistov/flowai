"""
Verdict/guidance для стадии Coder (mcp_agent/roles.py, mcp_agent/pipeline.py).

Coder получает утверждённый нумерованный план и должен его исполнить —
верификации здесь нет вообще (это отдельная стадия, Verifier), так что
verdict проверяет только "были ли реально сделаны правки" — те же
детерминированные предикаты self_heal.py, что раньше жили в общем дереве
mcp_agent/agent.py, просто без всей верификационной части (execution_
failure/syntax_only и т.д. — то, что здесь БЫЛО про bash_exec, целиком
переехало в mcp_agent/stages/verifier.py)."""
from mcp_agent.self_heal import _failed_write_messages, _has_successful_write, _wrote_code


def coder_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    failed = _failed_write_messages(new_tool_msgs)
    # _has_successful_write guard: a failed attempt earlier in the round
    # that the model then retried and got right must not still fail the
    # whole round — only reject here when EVERY write attempt failed.
    if failed and not _has_successful_write(new_tool_msgs):
        return {
            "relevant": False,
            "reason": f"the `{failed[0].name}` call failed with a tool error — nothing was actually written, the plan isn't done",
        }
    if not _wrote_code(new_tool_msgs):
        return {
            "relevant": False,
            "reason": "no write/edit tool was called this round — the plan requires actual code changes, not just reading",
        }
    if not round_final_text.strip():
        return {
            "relevant": False,
            "reason": "edits were made but no final report was written — report what changed for each numbered plan step",
        }
    return {"relevant": True, "reason": "applied edits and reported what changed"}


def coder_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    failed = _failed_write_messages(new_tool_msgs)
    if failed:
        errors = "\n".join(f"- {m.name}: {str(m.content)[:500]}" for m in failed)
        # Live run (mail-server, ee810cad03a24f33beb9b21b9d2b25c0): this
        # guidance used to just say "read the error and fix your
        # arguments" — technically correct, but vague enough that the next
        # attempt spent its ENTIRE round re-reading files and never
        # retried the write at all (coder_verdict then failed it a SECOND
        # time for "no write/edit tool was called this round"). The error
        # below already contains the file's exact current content at that
        # range (fs_extra_server.py's replace_lines echoes it back on a
        # mismatch) — that's normally enough to fix expected_first_line/
        # expected_last_line without a fresh read at all.
        return (
            "Your write/edit call failed — nothing was written, don't treat it "
            "as done. The error below already shows the file's ACTUAL current "
            "content at that range — use it directly to fix your arguments "
            "(expected_first_line/expected_last_line must be copied byte-for-"
            "byte from a SINGLE line, not multiple lines joined together) and "
            "call the SAME write again THIS round. Only re-read the file first "
            "if the error doesn't already give you what you need — don't spend "
            f"the whole round just reading without retrying the write:\n{errors}"
        )
    if not _wrote_code(new_tool_msgs):
        return (
            "You didn't make any edits this round. Follow the numbered plan "
            "literally — use replace_lines/insert_lines/copy_lines/write_file "
            "to apply each step, in order, don't just re-read files you "
            "already have from the plan/Analyzer's findings above."
        )
    return "Write a final report describing what you changed, numbered 1:1 with the plan's steps."
