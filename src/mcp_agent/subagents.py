"""
Sub-agent types for `delegate` (mcp_agent/delegate_tool.py), in the Claude
Code style: a few built-in types plus user-defined ones from markdown files.

Built-in:
- explore — read-only investigation (the original delegate): reads,
  searches, web; reports findings with file:line citations.
- plan    — read-only; returns an implementation plan for a change.
- general — every tool the main agent has, including file edits and bash
  (with the same permission dialogs): does a whole piece of work and
  reports what it changed.

Custom agents — <project>/.flowai/agents/<name>.md or ~/.flowai/agents/<name>.md
(project first; a custom agent may override a built-in name):

    ---
    name: test-writer
    description: Writes pytest tests for a given module. Use when ...
    tools: read_file, grep_search, write_file, bash   (optional; omitted = all tools)
    ---
    <system prompt of the sub-agent>

All sub-agents run on the SAME resident model as the main agent (never a
second instance) — how many can generate at once is settings.parallel_slots
(expert_streaming.py's -np), see delegate_tool.py.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# Tools a sub-agent never gets, whatever its spec says: delegate (no nested
# sub-agents — one resident model, and recursion would multiply the load),
# ask_user (a sub-agent can't talk to the user, its task must be
# self-contained), and the main agent's plan-checklist tools (they drive
# the main conversation's UI).
NEVER_FOR_SUBAGENTS = frozenset({"agent", "agent_result", "delegate", "ask_user", "submit_plan", "mark_plan_step_current"})

# Tools that change something — a spec that can call any of these is a
# "writing" agent: needs permission prompts; in plan mode it runs downgraded to read-only tools.
MUTATING_TOOLS = frozenset({
    "write_file", "edit_file", "delete_path", "restore_deleted_path", "restore_file_snapshot",
    "bash", "bash_bg", "update_memory", "update_knowledge", "remember_url",
    "generate_image", "edit_image", "generate_music", "generate_3d_model",
    "animate_3d_model", "generate_texture_for_model",
})

# Model-facing prompts (English, see CLAUDE.md).
_COMMON_TAIL = (
    "You are a sub-agent: another agent delegated ONE task to you. You can't "
    "ask the user anything and you don't see the rest of the conversation — "
    "the task text is all you have. You have your own step budget; don't "
    "re-read files you already read, prefer lsp (goToDefinition/"
    "findReferences) over guessing where a symbol lives. If you run out of "
    "steps or can't finish, say plainly what you did, what you found and "
    "what's still unknown — never guess to fill a gap."
)

EXPLORE_PROMPT = (
    "You are a focused research sub-agent. You have READ-ONLY tools — no "
    "writes, no shell, no git mutations.\n\n" + _COMMON_TAIL + "\n\n"
    "When you have enough to answer, STOP and write a complete, concrete "
    "final answer with exact file paths and line numbers, so the caller can "
    "verify without re-reading everything you read."
)

PLAN_PROMPT = (
    "You are a software-architect sub-agent. You have READ-ONLY tools. Read "
    "the code the change touches until you know exactly which files and "
    "functions change and how, then reply with an implementation plan: "
    "Goal; Context (what you found: files, functions, constraints); Steps "
    "(numbered, concrete: file, function, change); Verification (exact "
    "commands); Risks. Aim for the smallest justified change and reuse "
    "existing patterns of the repository.\n\n" + _COMMON_TAIL
)

GENERAL_PROMPT = (
    "You are a general-purpose coding sub-agent: carry the delegated task "
    "all the way through. You can read, edit and create files and run "
    "commands (the user may be asked to approve some of them; a rejected "
    "call is final — don't retry it). Investigate before editing, make the "
    "smallest justified change, reuse existing patterns, review your diff "
    "and run the relevant check (tests, linter, or the code itself).\n\n"
    + _COMMON_TAIL + "\n\n"
    "Final answer: what you changed (file:line), why, what you verified "
    "(command + result), and what remains unverified."
)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    system_prompt: str
    tools: frozenset[str] | None  # None = every tool the main agent has
    source: str  # "builtin" | "project" | "user"

    def can_mutate(self, available: set[str]) -> bool:
        names = available if self.tools is None else self.tools & available
        return bool(names & MUTATING_TOOLS)


def _builtin(read_only_tools: frozenset[str]) -> dict[str, AgentSpec]:
    return {
        "explore": AgentSpec(
            "explore",
            "Read-only investigation across many files: how something works end to end, "
            "where it's defined, what calls what. Reports findings with file:line citations.",
            EXPLORE_PROMPT, read_only_tools, "builtin",
        ),
        "plan": AgentSpec(
            "plan",
            "Read-only: studies the relevant code and returns a concrete implementation plan "
            "for a change (files, functions, steps, verification).",
            PLAN_PROMPT, read_only_tools, "builtin",
        ),
        "general": AgentSpec(
            "general",
            "General-purpose: does a whole self-contained piece of work — can edit files and "
            "run commands — and reports what it changed and verified.",
            GENERAL_PROMPT, None, "builtin",
        ),
    }


def user_agents_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".flowai" / "agents"


def _parse(path: Path, source: str) -> AgentSpec | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    meta: dict = {}
    body = text
    if text.startswith("---"):
        _, _, rest = text.partition("\n")
        front, sep, after = rest.partition("\n---")
        if sep:
            try:
                meta = yaml.safe_load(front) or {}
            except yaml.YAMLError:
                meta = {}
            body = after.partition("\n")[2]
    if not isinstance(meta, dict):
        meta = {}
    name = str(meta.get("name") or path.stem).strip()
    tools = meta.get("tools")
    if isinstance(tools, str):
        tools = [t.strip() for t in tools.split(",")]
    tool_set = frozenset(t for t in tools if t) if isinstance(tools, list) else None
    prompt = body.strip()
    if not name or not prompt:
        return None
    return AgentSpec(
        name=name,
        description=" ".join(str(meta.get("description") or "").split()),
        system_prompt=prompt + "\n\n" + _COMMON_TAIL,
        tools=tool_set,
        source=source,
    )


def discover_agents(repo_path: str | None, read_only_tools: frozenset[str]) -> dict[str, AgentSpec]:
    agents = _builtin(read_only_tools)
    roots = [(user_agents_dir(), "user")]
    if repo_path:
        roots.append((Path(repo_path) / ".flowai" / "agents", "project"))  # last = wins
    for root, source in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.md")):
            spec = _parse(path, source)
            if spec is not None:
                agents[spec.name] = spec
    return agents


def agents_prompt_block(agents: dict[str, AgentSpec]) -> str:
    lines = [
        "\n\n## Sub-agent types (agent tool's subagent_type)",
        "agent(description, prompt, subagent_type) runs a sub-agent on the same "
        "model with its own context. Pick the type that fits (default explore); "
        "write `prompt` as a complete task — it doesn't see this conversation.",
    ]
    for spec in agents.values():
        lines.append(f"- {spec.name}: {spec.description or '(no description)'}")
    return "\n".join(lines)
