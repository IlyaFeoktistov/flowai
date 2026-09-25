"""
Custom MCP server: flowai_guide — the model's own self-description tool,
for when a user asks what flowAI is / what it can do / how it works,
rather than having the model guess or hallucinate about its own host app.

Deliberately NOT a duplicate of cli.py's /help text (the exhaustive,
exact-flag command reference) — this is orientation, not a reference
manual; /help is the source of truth for exact command syntax and this
tool says so. Kept out of cli.py on purpose too: cli.py rewires sys.stdout/
sys.stderr at import time (see its own top-of-file comment) as a real,
load-bearing side effect for the TUI — importing it from an MCP subprocess
would run that again for no reason and is exactly the kind of cross-import
this project avoids elsewhere (see debug_log.py's docstring on the same
lesson for a different case).

Static text (only the user's home path is filled in), not templated from settings.py at call time: the one setting
genuinely worth surfacing live (the current chat model) already has its
own tool-free path (the model already knows what it's running as, same as
any LLM knows its own identity) — everything else here is architecture,
not configuration, and doesn't change between calls.

Запуск: python3 -m mcp_agent.servers.guide_server
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("guide")

_HOME = os.path.expanduser("~")

_GUIDE = f"""flowAI is a terminal CLI (plus a web UI) AI chat application. Everything \
runs against local models on the user's own machine — Ollama, or flowAI's \
own llama.cpp fork for MoE expert-streaming — no cloud model backend, \
nothing leaves the machine except through tools the model explicitly calls \
(fetch/web_search).

Two ways a turn can run (the user picks in /settings, "агентный режим"):
- основной (the default "main" agent): one agent with all tools. It has two \
work modes the user switches with Shift+Tab or /plan, /build: build (edits \
code) and plan (read-only: investigate and reply with a plan, which is \
saved to <project>/.flowai/plans/ and executed later with /build).
- агентный (pipeline): Analyzer (read-only investigation) -> Planner (plan, \
confirmed via ask_user) -> Coder (edits) -> Verifier (checks) — each stage \
its own LLM call with its own tools.
Casual chat and standalone snippets get a direct answer without tools.

Tools across modes: read/write/edit files, grep/glob search, lsp, shell \
(approval-gated), fetch, web search, semantic search over code/past \
dialog, a per-project knowledge base, per-user long-term memory; optional \
generative tools (images, music, 3D; off by default); voice in/out.

Sub-agents (main agent only) — the `agent` tool, same as Claude Code's: \
agent(description, prompt, subagent_type, run_in_background). Built-in \
types: explore (read-only investigation), plan (read-only implementation \
plan), general (all tools incl. edits and bash; in plan mode every type \
runs read-only). All sub-agents run on the \
SAME loaded model (never a second instance); how many generate at once is \
the "параллельные потоки" setting (default 1 — they then run one after \
another; extra calls wait their turn, background ones included). \
run_in_background=true returns an agent_id immediately; collect the report \
with agent_result(agent_id).

HOW TO ADD EXTENSIONS FOR flowAI ITSELF (not the open project's own code). \
Create directories that don't exist yet; nothing needs registering — \
everything is rescanned every turn:
- Skill (Claude Code format) — instructions YOU load on demand later. A \
folder <dir>/<name>/SKILL.md (room for the skill's own scripts/templates \
next to it) or a single file <dir>/<name>.md. <dir>: <open project>/.flowai/skills \
by default and whenever the user says "for this project"; \
{_HOME}/.flowai/skills only when they ask for it to work in every project. \
The YAML frontmatter is REQUIRED and must survive every later edit — \
without `description` the skill is never matched to a task. Template:
---
name: <name>
description: <WHEN to use it, e.g. "Use when the user asks for release notes">
allowed-tools: bash, read_file   # optional
---
<Numbered, imperative steps for yourself — exact commands to run, what to \
do with their output, how to format the answer. Not documentation for a \
human reader.> $ARGUMENTS is replaced with the invocation's arguments.
Used by you via the `skill` tool, or by the user as /<name>. To check it, \
call skill(name) and actually carry out its steps once.
- Custom sub-agent — a file <dir>/<name>.md with YAML frontmatter `name`, \
`description` (when to use it), optional `tools` (comma list; omitted = \
all tools); the body is the sub-agent's system prompt. <dir> is \
{_HOME}/.flowai/agents (user-wide) or <open project>/.flowai/agents. Then \
call it as agent(..., subagent_type="<name>").
- Python slash command — <open project>/.flowai/skills/<name>.py with a \
module-level run(args: str, console) (may be async; return a str to hand \
a task to the agent). Hook — <open project>/.flowai/hooks/<name>.py with \
post_file_edit(path, repo_path) and/or pre_commit(command, repo_path) -> \
str | None (non-empty return blocks the commit).
- Global plugin — <flowAI repo root>/plugins/<name>/plugin.json (commands, \
mcp_servers, hooks); heavier, meant to be shared.

Persistence: memory, knowledge, usage stats, settings (including per-model \
generation parameters) live in one SQLite database under \
~/.local/share/flowai/ (or $XDG_DATA_HOME / $FLOWAI_DATA_DIR).

For the exact slash commands and flags, point the user at /help; loaded \
models and their memory use — /instances."""


@mcp.tool()
async def flowai_guide() -> str:
    """Explain what flowAI itself is and how it works — its architecture
    (the Analyzer/Planner/Coder/Verifier pipeline for project work vs. the
    direct-answer fast path for casual chat/snippets), what categories of
    tools exist and when they're available, the optional generative
    features, where persistent data lives, and how to add a flowAI
    skill (SKILL.md), custom sub-agent, slash command, hook or plugin —
    exact folders and file formats. Call this when the user asks what you
    are, what you can do, how you work, or asks you to write/add a skill,
    agent or other extension for yourself — don't guess folders or formats."""
    return _GUIDE


if __name__ == "__main__":
    mcp.run()
