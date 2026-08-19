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

Static text, not templated from settings.py at call time: the one setting
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

_GUIDE = """flowAI is a terminal CLI AI chat application. Everything runs against \
local models served by Ollama on the user's own machine (plus an \
experimental custom llama.cpp fork for MoE expert-streaming) — no cloud \
model backend, nothing leaves the machine except through tools the model \
explicitly calls (fetch/web_search).

How a turn is handled:
- Casual conversation and standalone code snippets (no reference to the \
user's actual project) get a direct, fast answer with no tools at all.
- A request that touches the user's actual project goes through a \
multi-stage pipeline: Analyzer (read-only investigation) -> Planner \
(drafts a numbered plan, confirms it with the user via ask_user) -> Coder \
(makes the edits) -> Verifier (checks the result, can send it back for a \
fix). Each stage is its own LLM call with its own tool access and attempt \
budget — a small, unambiguous fix skips straight to a combined \
investigate+edit stage instead of the full four-stage path.
- Tools available across these stages: reading/writing/editing files, \
grep/glob search, running shell commands (approval-gated), fetching URLs, \
web search, semantic search over the project's code/past dialog/saved \
pages, a structured per-project knowledge base, and per-user long-term \
memory (facts the user asked to be remembered, persisted across sessions).
- Optional generative tools (off by default, toggled in /settings): local \
image generation/editing (Stable Diffusion/FLUX), local music generation \
(MusicGen), local 3D model generation/rigging/animation. Voice input/output \
(speech-to-text/text-to-speech) is available independently of those.
- /dnd starts a separate solo tabletop-RPG chat mode, unrelated to the \
coding pipeline.

flowAI's own extension mechanism (unrelated to whatever the currently open \
project's code does) — use this when the user asks to add a plugin/skill/ \
hook FOR flowAI itself, whether that's global or scoped to the project \
they have open right now:
- Global plugin — a folder at `<flowAI repo root>/plugins/<name>/` with a \
`plugin.json` manifest (commands/mcp_servers/hooks, all optional), shared \
across every project. Heavier format, meant to be reused/distributed.
- Per-project skill — `.flowai/skills/<name>.py` inside the project \
CURRENTLY OPEN (not flowAI's own repo), a module-level `run(args: str, \
console) -> None` (may be async); the filename minus `.py` becomes the \
slash-command name. No manifest.
- Per-project hook — `.flowai/hooks/<name>.py` in the same place, a \
module-level `post_file_edit(path, repo_path)` (runs after a successful \
write_file/edit_file, return value ignored) and/or `pre_commit(command, \
repo_path) -> str | None` (runs before a `git commit` bash call; a \
non-empty return BLOCKS the commit with that string as the reason). \
Either or both in one file; both may be async. No manifest.
- Create the `.flowai/skills`/`.flowai/hooks` directory in the project if \
it doesn't exist yet — nothing needs registering elsewhere, the files are \
discovered by scanning that directory on the next turn/restart.

Persistence: everything (long-term memory, per-project knowledge, usage \
stats, settings) lives in one SQLite database under the user's own data \
directory (~/.local/share/flowai/ or $XDG_DATA_HOME, overridable via \
$FLOWAI_DATA_DIR) — independent of whatever directory the user happens to \
be running flowai from.

For the exact list of slash commands and their flags, point the user at \
/help rather than repeating a possibly-stale copy of it here."""


@mcp.tool()
async def flowai_guide() -> str:
    """Explain what flowAI itself is and how it works — its architecture
    (the Analyzer/Planner/Coder/Verifier pipeline for project work vs. the
    direct-answer fast path for casual chat/snippets), what categories of
    tools exist and when they're available, the optional generative
    features, where persistent data lives, and how to add a flowAI
    plugin/skill/hook (global plugins/ vs. per-project .flowai/skills or
    .flowai/hooks, exact file/function shapes). Call this when the user
    asks what you are, what you can do, how you work, or asks you to add
    one of these extensions — don't guess or make up capabilities/formats."""
    return _GUIDE


if __name__ == "__main__":
    mcp.run()
