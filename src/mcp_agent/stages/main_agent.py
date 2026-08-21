"""
Verdict/guidance для основного монолитного агента (mcp_agent/agent.py) —
извлечено из его stream_chat почти построчно (та же дисциплина, что
описана в agent.py's own module docstring: копия по точным диапазонам, не
пересочинено по памяти), чтобы agent.py:stream_chat стал таким же тонким
вызывателем mcp_agent/stage_runner.py:run_stage, как и mcp_agent/pipeline.py
— вместо второй, отдельно поддерживаемой копии всего self-heal цикла
(recursion-limit/context-overflow/ResponseError/leaked-tool-call/punt-to-
user — всё это уже общее в run_stage, см. его докстринг).

В отличие от остальных ролей пайплайна (mcp_agent/stages/analyzer.py и
все остальные — чисто детерминированные, никто не завязан на
self_heal.py:_semantic_check, см. docstring analyzer.py) основной агент
делает всё разом (исследование+запись+проверка в ОДНОМ раунде без
разделения на роли), так что, в отличие от них, у него есть финальный
LLM-судья fallback, когда ни один детерминированный чек не сработал —
ровно то же дерево, что раньше жило прямо в agent.py:stream_chat.

verdict_fn получает judge_model/task_text/on_event через замыкание
(make_main_verdict) — единственная роль, которой они вообще нужны;
менять общий verdict_fn(round_msgs, new_tool_msgs, round_final_text)
контракт (mcp_agent/stage_runner.py:run_stage) ради этого одного случая не
стоило."""
from mcp_agent.message_utils import _tool_text
from mcp_agent.self_heal import (
    _called_ask_user,
    _described_commits_without_diff,
    _execution_evidence_shows_failure,
    _extract_diffed_paths,
    _failed_write_messages,
    _final_answer_ignores_diff,
    _git_status_reports_changes,
    _has_diff_evidence,
    _has_execution_evidence,
    _has_successful_write,
    _has_truncated_output,
    _retried_after_rejection,
    _semantic_check,
    _truncated_git_diff,
    _verified_with_syntax_check_only_despite_discoverable_tests,
    _wrote_code,
)


def make_main_verdict(judge_model, task_text: str, on_event):
    """Возвращает verdict_fn для этого ОДНОГО хода — judge_model/task_text/
    on_event зафиксированы через замыкание, run_stage продолжает звать
    результат с обычной (round_msgs, new_tool_msgs, round_final_text)
    сигнатурой, как и любой другой verdict_fn."""

    async def _judge(tool_messages: list, ask_user_called: bool, round_final_text: str) -> dict:
        # verifying_start/end — тот же UI-сигнал, что раньше стоял вокруг
        # ОБОИХ вызовов _semantic_check в agent.py (без-тульный "?"-ответ и
        # финальный fallback ниже) — сам judge-вызов может занять минуты на
        # медленной локальной модели, без явного события это выглядит как
        # "ответ уже написан, а крутится неизвестно почему".
        if on_event:
            await on_event({"type": "verifying_start"})
        try:
            return await _semantic_check(judge_model, task_text, tool_messages, round_final_text, ask_user_called)
        finally:
            if on_event:
                await on_event({"type": "verifying_end"})

    async def main_verdict(round_msgs: list, new_tool_msgs: list, round_final_text: str) -> dict:
        if not new_tool_msgs:
            # Дешёвый ФИЛЬТР, не вердикт: без "?" в тексте раунд без единого
            # тула считается штатным прямым ответом — судью гонять незачем
            # (см. agent.py's original comment on this exact filter).
            if "?" not in round_final_text and "？" not in round_final_text:
                return {"relevant": True, "reason": "plain text answer, nothing to verify"}
            return await _judge([], False, round_final_text)

        failed_writes = _failed_write_messages(new_tool_msgs)
        # _has_successful_write guard — same fix as coder_verdict/
        # quick_fix_verdict's shared _write_stage_outcome (self_heal.py):
        # an earlier edit_file/write_file attempt that failed (e.g.
        # old_string not found) and was then retried and got RIGHT later
        # in this same round must not fail the whole round — only reject
        # here when EVERY write attempt failed. Without this, a round that
        # actually applied the fix (and even verified it with a real bash
        # run) still got rejected with "nothing was actually written",
        # because an earlier, since-corrected mismatch was the only thing
        # this check looked at.
        if failed_writes and not _has_successful_write(new_tool_msgs):
            return {
                "relevant": False,
                "reason": (
                    f"the `{failed_writes[0].name}` call failed with a tool "
                    "error — nothing was actually written/edited, the task "
                    "isn't done"
                ),
            }
        rejected_retry_tool = _retried_after_rejection(new_tool_msgs)
        if rejected_retry_tool:
            return {
                "relevant": False,
                "reason": (
                    f"the model called `{rejected_retry_tool}` again "
                    "immediately after the user rejected it — the tool "
                    "result explicitly said not to retry unless the user "
                    "asks, and that was ignored"
                ),
            }
        if _wrote_code(new_tool_msgs) and _execution_evidence_shows_failure(round_msgs):
            return {
                "relevant": False,
                "kind": "execution_failure",
                "reason": (
                    "the command run to verify the change (bash) failed "
                    "with an error — the code doesn't actually work yet, "
                    "writing/editing the file is not enough"
                ),
            }
        if _wrote_code(new_tool_msgs) and not _has_execution_evidence(new_tool_msgs):
            return {
                "relevant": False,
                "reason": (
                    "a file was written/edited but no command was actually "
                    "executed (no bash call) — writing/editing a file "
                    "is not verification that it works"
                ),
            }
        if _wrote_code(new_tool_msgs) and _verified_with_syntax_check_only_despite_discoverable_tests(round_msgs):
            return {
                "relevant": False,
                "kind": "syntax_only_verification",
                "reason": (
                    "the only bash calls this round were bare syntax/"
                    "lint checks (e.g. `php -l`, `py_compile`, `tsc "
                    "--noEmit`) — that confirms the file parses, not that "
                    "the change behaves correctly, and a real test file "
                    "surfaced in this same round's tool results was never "
                    "actually run"
                ),
            }
        if _git_status_reports_changes(new_tool_msgs) and not _has_diff_evidence(new_tool_msgs):
            return {
                "relevant": False,
                "reason": (
                    "git_status reported non-empty staged/unstaged changes, "
                    "but the diff tools called don't cover the full picture "
                    "— either call git_diff('HEAD'), or call BOTH "
                    "git_diff_staged AND git_diff_unstaged (one of them "
                    "alone can miss half the changes if there's content in "
                    "both the staged and unstaged sections)"
                ),
            }
        if _truncated_git_diff(new_tool_msgs):
            return {
                "relevant": False,
                "reason": (
                    "a git_diff/git_diff_staged/git_diff_unstaged result was "
                    "truncated — these tools can't be scoped to a single "
                    "file (only context_lines, which makes output bigger, "
                    "not smaller); use bash with `git diff -- <path>` "
                    "per file instead of retrying the same call"
                ),
            }
        if _has_truncated_output(new_tool_msgs):
            return {
                "relevant": False,
                "reason": (
                    "a tool result was truncated (marked with '...[TRUNCATED') "
                    "before showing the whole output — answering from the "
                    "truncated part alone risks missing the actual content "
                    "of the change; narrow the query (specific file/path, "
                    "smaller context_lines/max_results) or call the tool "
                    "again to see the rest before answering"
                ),
            }
        diffed_paths = _extract_diffed_paths(new_tool_msgs)
        if _final_answer_ignores_diff(round_final_text, diffed_paths):
            return {
                "relevant": False,
                "reason": (
                    "the diff tools returned real changes to specific files "
                    f"({', '.join(sorted(diffed_paths))}), but the answer "
                    "doesn't mention any of them — it looks like it answered "
                    "from something else (a tangential lookup) instead of "
                    "that diff; rewrite the answer based on the diff content "
                    "already retrieved, don't call more tools"
                ),
            }
        if _described_commits_without_diff(new_tool_msgs, round_final_text):
            return {
                "relevant": False,
                "kind": "commits_described_without_diff",
                "reason": (
                    "the answer describes specific commits by hash, but "
                    "git_log only returned metadata (hash/author/date/"
                    "message) — no git_show or git_diff call ever read what "
                    "those commits actually changed, so the per-commit "
                    "descriptions were written from commit messages alone, "
                    "not real diff content"
                ),
            }
        verdict = await _judge(new_tool_msgs, _called_ask_user(new_tool_msgs), round_final_text)
        # "kind" маркер, не входящий в _semantic_check's собственный вывод —
        # main_guidance ниже читает его, чтобы отличить "вердикт пришёл от
        # LLM-судьи" (semantic_verdict_used в оригинале) от детерминированных
        # веток выше, не пересчитывая заново, какая именно ветка сработала.
        verdict["kind"] = "semantic"
        return verdict

    return main_verdict


def main_guidance(verdict: dict, round_msgs: list, new_tool_msgs: list, round_final_text: str) -> str:
    if not new_tool_msgs:
        # Достижимо только когда verdict пришёл от _judge на без-тульном
        # "?"-ответе (см. main_verdict) — run_stage's собственный
        # _seed_retry уже добавляет "The previous tool results don't answer
        # the task (reason: ...)." ПЕРЕД этим текстом (то же самое
        # оригинальное вступление, см. mcp_agent/stage_runner.py), так что
        # здесь нужно только "что делать дальше", без повторения reason.
        return (
            "If it just handed a decision or open question back to the "
            "user in plain text, call the ask_user tool now with that "
            "question and the concrete options — a question typed into "
            "your response doesn't wait for or receive an answer. If the "
            "task actually told you to decide yourself, answer with your "
            "own reasoned pick instead of asking."
        )

    failed_writes = _failed_write_messages(new_tool_msgs)
    rejected_retry_tool = _retried_after_rejection(new_tool_msgs)
    diffed_paths = _extract_diffed_paths(new_tool_msgs)
    semantic_verdict_used = verdict.get("kind") == "semantic"

    # Собираем корректирующую подсказку из ПРИМЕНИМЫХ частей вместо одного
    # жёстко зашитого текста — иначе сценарий "записан код" получал бы совет
    # про bash даже когда провал был из-за непрочитанного диффа, и наоборот
    # (см. оригинальный комментарий в agent.py).
    guidance_parts = []
    if failed_writes:
        errors = "\n".join(f"- {m.name}: {_tool_text(m.content)[:500]}" for m in failed_writes)
        guidance_parts.append(
            "Your write_file/edit_file call failed with a tool error — "
            "nothing was written, don't treat it as done or move on. "
            "Read the error below and fix your arguments to match its "
            "exact schema before retrying (edit_file in particular: "
            "`edits` must be an ARRAY of {oldText, newText} objects, "
            f"even for a single change, never a bare object):\n{errors}"
        )
    if rejected_retry_tool:
        guidance_parts.append(
            f"The user already rejected `{rejected_retry_tool}` and the "
            "tool result told you not to retry it unless asked — do not "
            "call it again. Answer with whatever information you "
            "already have, or explain in your answer what you wanted to "
            "do and why, instead of retrying a denied action."
        )
    if semantic_verdict_used and not verdict["relevant"] and not _called_ask_user(new_tool_msgs):
        guidance_parts.append(
            "If your last answer just handed a decision or open question "
            "back to the user in plain text instead of calling ask_user "
            "— don't call more research tools first, you likely already "
            "have enough context. Call the ask_user tool now with that "
            "same question and the concrete options you were weighing "
            "(or, if the task actually asked you to decide yourself, "
            "answer with your own reasoned choice instead of asking)."
        )
    if _wrote_code(new_tool_msgs) and not _has_execution_evidence(new_tool_msgs) and not failed_writes:
        guidance_parts.append(
            "A file was written/edited — call bash and actually run "
            "it (or its tests), don't claim it works without running it."
        )
    elif verdict.get("kind") == "execution_failure":
        bash_errors = "\n".join(
            f"- {_tool_text(m.content)[:500]}" for m in new_tool_msgs if m.name == "bash"
        )
        guidance_parts.append(
            "The command you ran to verify the change FAILED — but the "
            "edit itself from the previous attempt is still there, you "
            "don't need to redo it or re-explore the files you already "
            "found. Do NOT restate the task or start over from scratch; "
            "your only remaining job this round is to get a real "
            "verification result. First check whether the error below is "
            "actually about YOUR code, or about how you ran the check "
            "(wrong interpreter/command, missing tool that's normally "
            "available in this project) — if it's the latter, find the "
            "right way to run it (project launcher, Makefile, README/CI, "
            "existing venv/toolchain) and rerun, don't conclude the code "
            "is broken. If you can't pin down the fix within your "
            "remaining attempts, say so plainly in your final answer "
            "instead of reporting success:\n" + bash_errors
        )
    elif verdict.get("kind") == "syntax_only_verification":
        guidance_parts.append(
            "A syntax/lint check (php -l, py_compile, tsc --noEmit, ...) "
            "only proves the file parses — it does not verify the "
            "changed behavior is correct. This round's own tool results "
            "already named a real test file for this area — run the "
            "project's actual test runner (phpunit, pytest, npm test, go "
            "test, ...) covering the file(s) you changed, and use ITS "
            "pass/fail result as your verification instead of the "
            "syntax check alone."
        )
    if _git_status_reports_changes(new_tool_msgs) and not _has_diff_evidence(new_tool_msgs):
        guidance_parts.append(
            "git_status reported non-empty changes — call git_diff('HEAD'), "
            "or call BOTH git_diff_staged and git_diff_unstaged (one of "
            "the two alone is not enough if changes exist in both "
            "sections) — git_status only names files, it doesn't show "
            "what changed inside them. That's enough to answer — do NOT "
            "also read whole files (read_file/read_text_file) or call "
            "list_directory/directory_tree, the diff already has "
            "everything you need."
        )
    if _truncated_git_diff(new_tool_msgs):
        guidance_parts.append(
            "Part of the previous git_diff/git_diff_staged/git_diff_unstaged "
            "result was truncated — these tools have NO way to scope to a "
            "single file (their only parameter besides the repo is "
            "context_lines, which makes output BIGGER, not smaller). "
            "Retrying with a different context_lines won't help. Instead, "
            "use bash with a plain `git diff -- <path>` (or `git diff "
            "--cached -- <path>`) for ONE file at a time, and repeat per "
            "file if there are several."
        )
    elif _has_truncated_output(new_tool_msgs):
        guidance_parts.append(
            "Part of the previous tool result was truncated (marked "
            "'[TRUNCATED...]') — don't answer as if you saw all of it; "
            "retry the SAME tool with a narrower query (specific file/"
            "path, smaller context_lines/max_results). Don't switch to a "
            "different tool (read_file on the whole file, list_directory, "
            "etc.) — that won't show the truncated part and will waste "
            "your last attempt."
        )
    if _final_answer_ignores_diff(round_final_text, diffed_paths):
        guidance_parts.append(
            "You already have the real diff content from earlier in this "
            f"investigation (changes to {', '.join(sorted(diffed_paths))}) "
            "— your last answer didn't use it and talked about something "
            "else instead. Don't call more tools or go research another "
            "tangent — write the answer directly from that diff content."
        )
    if verdict.get("kind") == "commits_described_without_diff":
        guidance_parts.append(
            "You described specific commits by hash without ever reading "
            "what they actually changed — git_log only gives you the "
            "message, not the code. Call git_show(<hash>) (or "
            "git_diff(target=<hash>)) for each commit you're about to "
            "describe, then rewrite the answer quoting the real diff "
            "content for each one instead of paraphrasing its commit "
            "message."
        )
    if not guidance_parts:
        guidance_parts.append(
            "The judge's reason above is the real signal — don't default "
            "to 'gather more information' without reading it. You may "
            "already have everything you need from what's in the digest "
            "above: if so, don't call ANY more research tools — answer "
            "now with a concrete decision, and if the task implies "
            "changing code, propose AND WRITE the actual fix (a "
            "description of how the current code behaves, with no "
            "proposed change, is likely what got this rejected). Only "
            "call more tools if you genuinely lack a specific piece of "
            "information, and even then don't repeat a structural "
            "listing (list_directory/directory_tree) with a different "
            "path — it already showed what exists."
        )
    return " ".join(guidance_parts)
