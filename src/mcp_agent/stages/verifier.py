"""
Verdict/guidance для стадии Verifier (mcp_agent/roles.py, mcp_agent/pipeline.py).

Ключевое отличие от легаси mcp_agent/agent.py: там "execution_failure" был
ПРОВАЛОМ self-heal раунда (весь ход должен был сам и написать, и
проверить). Здесь Verifier — ОТДЕЛЬНАЯ роль, чья единственная работа —
честно запустить проверку и доложить результат: реальный fail — это
ВАЛИДНЫЙ, relevant=True исход её раунда (она сделала свою работу), просто
с kind="execution_failure", которую mcp_agent/pipeline.py читает отдельно,
чтобы решить, возвращать ли правки Coder'у на новый круг."""
from mcp_agent.self_heal import (
    _execution_evidence_shows_failure,
    _has_execution_evidence,
    _verified_with_syntax_check_only_despite_discoverable_tests,
)


def verifier_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
    if not _has_execution_evidence(new_tool_msgs):
        return {
            "relevant": False,
            "reason": "no bash call was made this round — Verifier must actually run a real check, reading files/diffs alone is not verification",
        }
    if not round_final_text.strip():
        return {"relevant": False, "reason": "ran a check but wrote no final report"}
    if _verified_with_syntax_check_only_despite_discoverable_tests(round_msgs):
        return {
            "relevant": False,
            "kind": "syntax_only_verification",
            "reason": "only a bare syntax/lint check was run despite a real test file being discoverable in this round's own tool results",
        }
    if _execution_evidence_shows_failure(round_msgs):
        # relevant=True — Verifier сделала СВОЮ работу честно (запустила
        # реальную проверку и получила реальный результат), сам факт что
        # результат FAIL не делает её раунд self-heal-провалом. pipeline.py
        # решает по kind, что делать с этим fail дальше.
        return {
            "relevant": True,
            "kind": "execution_failure",
            "reason": "ran a real check and it failed",
        }
    return {"relevant": True, "reason": "ran real checks and they passed"}


def verifier_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    if not _has_execution_evidence(new_tool_msgs):
        return (
            "You must actually run a real check via bash (the test "
            "suite, a linter/type-checker, or executing the changed script) "
            "— reading files or the diff alone is not verification."
        )
    if verdict.get("kind") == "syntax_only_verification":
        return (
            "A syntax/lint check alone (py_compile, tsc --noEmit, php -l) "
            "only proves the file parses, not that the behavior is correct "
            "— a real test file for this area was visible in your own tool "
            "results, run the project's actual test runner instead."
        )
    return (
        "Write a final report: for each plan step, done/missing/wrong, then "
        "pass or fail for the real checks, the exact command you ran, and "
        "its actual output."
    )
