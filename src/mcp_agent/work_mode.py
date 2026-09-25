"""
plan / build — two ways to run a turn of the main agent (mcp_agent/agent.py,
"основной" mode; the Analyzer->Planner->Coder->Verifier pipeline keeps its
own staging and ignores this), same idea as opencode's plan/build agents:

- build — the normal mode: the agent reads, edits, runs commands.
- plan  — read-only research: investigate and answer, change nothing; an
  implementation plan only when the user asks for one. Enforced
  mechanically by PlanModeMiddleware (agent_builder._base_agent_middleware):
  only read-only tools pass, bash only for read-only commands
  (agent_builder._is_read_only_bash_command) — not just a prompt request.

A plan-mode reply that IS a plan (has a Steps section, see is_plan) is saved to
<project>/.flowai/plans/<timestamp>-<slug>.md, so it never has to be copied
out of the chat window: `/build` (CLI and web) switches to build mode and
hands the latest saved plan to the agent as the task; the file itself can
also be edited by hand before running it.

current_work_mode is set by the caller (cli.py/main.py) right before the
turn, same pattern as plugins.current_skill_restriction.
"""
import re
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

PLAN = "plan"
BUILD = "build"

current_work_mode: ContextVar[str] = ContextVar("work_mode", default=BUILD)

# Allowlist, not a denylist: a new or plugin-provided tool must not become
# able to mutate things in plan mode just because nobody listed it.
PLAN_ALLOWED_TOOLS = frozenset({
    "read_file", "grep_search", "glob_search", "lsp", "search_code_semantic",
    "search_dialog_history", "list_episodic_sessions", "read_episodic_session",
    "web_search", "fetch", "web_read", "search_external_sources",
    "get_knowledge", "list_memory", "flowai_guide", "skill", "ask_user",
    # agent: writing sub-agent types are refused in plan mode (delegate_tool._resolve)
    "agent", "agent_result", "analyze_image", "list_file_snapshots", "list_deleted_paths",
    "bash", "bash_bg", "bash_bg_check", "bash_bg_list",  # bash: read-only commands only, see PlanModeMiddleware
})

# Model-facing (English, see CLAUDE.md). Appended to the END of the system
# prompt in plan mode (agent_builder._build_agent) — last position, since it
# must override the base prompt's "write the change, then verify" workflow.
PLAN_SYSTEM_PROMPT_SECTION = (
    "\n\n## PLAN MODE — read-only research (overrides the code-change workflow above)\n"
    "You are in PLAN mode for this whole turn: investigate and answer, change "
    "nothing. Only read-only tools are available; you cannot write, edit or "
    "delete files, and bash accepts read-only commands only (git log/diff/"
    "status, ls, cat, grep...). Skip the 'write the change' and 'verify' "
    "steps above entirely. If a skill or the request calls for edits, don't "
    "attempt them — say what would change instead.\n"
    "Answer what the user actually asked: an explanation, findings, a review, "
    "a comparison — in the user's language, backed by what you read (file:line). "
    "Don't turn every answer into a plan.\n"
    "ONLY when the user asks for a plan (or asks how to implement/change "
    "something), reply with an implementation plan in Markdown:\n"
    "## Goal — one or two sentences\n"
    "## Context — what you found that matters (files, functions, constraints)\n"
    "## Steps — numbered, concrete: which file, which function, what change\n"
    "## Verification — the exact commands/checks that prove it works\n"
    "## Risks / open questions — if any\n"
    "(headings may be in the user's language, e.g. ## Шаги for Steps). Such a "
    "plan is saved to a file and later executed by an agent in build mode "
    "that has NOT seen this conversation, so every step must be "
    "self-contained: full file paths, names, and the reasoning it needs. "
    "Read the code until you know exactly what changes — a plan built on "
    "guesses is useless; a real decision that belongs to the user goes to "
    "ask_user."
)

# Short per-turn reminder on the user's message — the full rules live in
# the system prompt; this keeps them from being lost behind a long history.
PLAN_MODE_INSTRUCTION = "\n\n(PLAN MODE: read-only — investigate and answer; write a plan only if asked; do not modify anything.)"

# A reply counts as a plan (saved for /build) only if it has the Steps
# section the plan format above requires — plain research answers in plan
# mode are not plans.
_PLAN_STEPS_HEADING = re.compile(r"^#{1,4}\s*(steps|шаги|план действий)\b", re.IGNORECASE | re.MULTILINE)


def is_plan(text: str) -> bool:
    return bool(_PLAN_STEPS_HEADING.search(text or ""))


def plans_dir(repo_path: str) -> Path:
    return Path(repo_path) / ".flowai" / "plans"


def _slug(text: str) -> str:
    words = re.findall(r"[\w-]+", text.lower())[:6]
    return "-".join(words)[:50].strip("-") or "plan"


def save_plan(repo_path: str, task: str, plan_text: str) -> Path:
    directory = plans_dir(repo_path)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug(task)}.md"
    header = f"<!-- flowai plan · {datetime.now().strftime('%Y-%m-%d %H:%M')} · task: {' '.join(task.split())[:200]} -->\n\n"
    path.write_text(header + plan_text.strip() + "\n", encoding="utf-8")
    return path


def latest_plan(repo_path: str) -> Path | None:
    directory = plans_dir(repo_path)
    if not directory.is_dir():
        return None
    plans = sorted(directory.glob("*.md"))
    return plans[-1] if plans else None


def build_task_from_plan(plan_path: Path, extra: str = "") -> str:
    """Model-facing task for `/build`: the plan file's CURRENT content (it
    may have been edited by hand since it was saved)."""
    plan = plan_path.read_text(encoding="utf-8")
    task = (
        f"Implement the following plan (saved at {plan_path}). Follow its steps "
        "in order, verify the result as it describes, and report what you "
        "changed. If a step turns out to be wrong once you see the real code, "
        "adapt it and say so.\n\n" + plan
    )
    if extra.strip():
        task += f"\n\nAdditional instructions from the user: {extra.strip()}"
    return task
