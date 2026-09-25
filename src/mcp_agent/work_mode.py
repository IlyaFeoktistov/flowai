"""
plan / build — two ways to run a turn of the main agent (mcp_agent/agent.py,
"основной" mode; the Analyzer->Planner->Coder->Verifier pipeline keeps its
own staging and ignores this), same idea as opencode's plan/build agents:

- build — the normal mode: the agent reads, edits, runs commands.
- plan  — investigate and produce a plan, change nothing. Enforced
  mechanically by PlanModeMiddleware (agent_builder._base_agent_middleware):
  only read-only tools pass, bash only for read-only commands
  (agent_builder._is_read_only_bash_command) — not just a prompt request.

The plan a plan-mode turn produces is saved to
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
    # delegate's sub-agent only has read-only tools (delegate_tool._ALLOWED_TOOLS)
    "delegate", "analyze_image", "list_file_snapshots", "list_deleted_paths",
    "bash", "bash_bg", "bash_bg_check", "bash_bg_list",  # bash: read-only commands only, see PlanModeMiddleware
})

# Model-facing (English, see CLAUDE.md). Appended to the END of the system
# prompt in plan mode (agent_builder._build_agent) — last position, since it
# must override the base prompt's "write the change, then verify" workflow.
PLAN_SYSTEM_PROMPT_SECTION = (
    "\n\n## PLAN MODE (overrides the code-change workflow above)\n"
    "You are in PLAN mode for this whole turn. Your job is to investigate and "
    "produce an implementation plan — NOT to implement it. Only read-only "
    "tools are available; you cannot write, edit or delete files, and bash "
    "accepts read-only commands only (git log/diff/status, ls, cat, grep...). "
    "Skip the 'write the change' and 'verify' steps above entirely.\n"
    "1. Read the code the request touches until you know exactly which "
    "files/functions change and how — a plan built on guesses is useless.\n"
    "2. If a real decision belongs to the user (two valid designs, unclear "
    "scope), ask it with ask_user; don't ask what the code can answer.\n"
    "3. Reply with the final plan in Markdown, in the user's language:\n"
    "## Goal — one or two sentences\n"
    "## Context — what you found that matters (files, functions, constraints)\n"
    "## Steps — numbered, concrete: which file, which function, what change\n"
    "## Verification — the exact commands/checks that prove it works\n"
    "## Risks / open questions — if any\n"
    "The plan is saved to a file and later executed by an agent in build "
    "mode that has NOT seen this conversation, so every step must be "
    "self-contained: full file paths, names, and the reasoning it needs."
)

# Short per-turn reminder on the user's message — the full rules live in
# the system prompt; this keeps them from being lost behind a long history.
PLAN_MODE_INSTRUCTION = "\n\n(PLAN MODE: investigate and reply with a plan; do not modify anything.)"


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
