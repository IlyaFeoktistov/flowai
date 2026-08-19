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
    features, and where persistent data lives. Call this when the user
    asks what you are, what you can do, or how you work — don't guess or
    make up capabilities."""
    return _GUIDE


if __name__ == "__main__":
    mcp.run()
