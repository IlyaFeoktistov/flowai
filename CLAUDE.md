# flowAI

Python CLI AI chat application powered by Ollama.

## Language rule

**Everything the model reads must be in English:**
- Tool descriptions and parameter descriptions in `tools/__init__.py` — English only
- Return values from tool handlers (`bash_exec.py`, `file_ops.py`, etc.) — English only
- System prompt context injected in `agent.py` (today's date label, home dir label, memory block) — English only

**Everything the user reads stays in Russian:**
- UI strings: spinner labels, confirmation dialogs, header, toolbar, error messages shown to the user
- `ui/` directory: all user-facing text in Russian
- `cli.py`: command help, status messages in Russian

## Stack

All source lives under `src/` (flat layout physically under `src/`, not
`src/flowai/` — `pyproject.toml`'s `sources = ["src"]` strips the prefix at
install time, so imports everywhere else still read as top-level `import
cli`/`import mcp_agent`/...).

- `mcp_agent/agent.py` — legacy agentic loop (voice_mode/`pipeline_mode=off`); `mcp_agent/pipeline.py` — the default Analyzer→Planner→Coder→Verifier pipeline
- `mcp_agent/plugins.py` — plugin loader: global plugins (slash commands, MCP servers, hooks) under `<repo root>/plugins/`, plus manifest-free per-project skills/hooks under `<open project>/.flowai/{skills,hooks}/`; `mcp_agent/plugin_hooks.py` — the post_file_edit/pre_commit hook middleware; `examples/plugins/hello-world/` and `examples/project-skills-hooks/` — reference examples; see `docs/plugins.md` for the full mechanism
- `tools/` — tool handlers: bash_exec, file_ops, web_search, read_page, image_gen, memory
- `ui/` — terminal UI: stream display, prompt_toolkit input, Rich console
- `settings.py` — model selection, GPU routing (persisted in SQLite, `~/.local/share/flowai/`)
- `memory/` — persistent user memory (SQLite, `~/.local/share/flowai/`)
- `FLOWAI.md` (optional, in the target working directory) — project-specific instructions the agent reads and follows, appended to the system prompt if present

## Models

RTX 4050 Laptop (5.9 GB VRAM). Current default (`settings.py`'s `chat_model`
fallback) is `glm-4.7-flash:q4_K_M` — MoE, ~3B active params, requires
`expert_streaming_enabled=ВКЛ` (also defaulted on, `settings.py`) since the
plain Ollama path doesn't support its `glm4moelite` architecture without the
patched `vendor/llama-expert-streaming` fork — see `expert_streaming.py`'s
"GLM-4.7-Flash" section. `qwen3-coder:30b` — MoE, ~3.3B active params,
mostly runs on CPU — is still fully supported on the plain Ollama path (no
extra build) if `expert_streaming_enabled` is off (see
`mcp_agent/model_config.py:OLLAMA_NUM_CTX` comments for the live-measured
CPU/GPU split on that model).

Note `.env`/`.env.example` no longer carry `OLLAMA_MODEL` — it only ever
seeded `settings.py`'s `_state` dict on a fresh install with no
`~/.local/share/flowai/flowai.db` row yet; once persisted there (which
happens as soon as `/settings` is touched, or immediately for anyone with an
existing DB), the env var is silently ignored. Change the model via
`/settings` at runtime, or edit the fallback in `settings.py` directly for a
new install's default.

Whether a model "fits fully on GPU" is NOT just weights-size vs. VRAM — the
agent always loads models with `OLLAMA_NUM_CTX=65536`, and the KV-cache at
that context size is often several GB on top of the weights, easily pushing
a model that looks like it should fit onto partial CPU offload instead. The
only reliable answer is a live measurement (`ollama ps` right after a real
request) — see `_MEASURED_GPU_SHARE` in `ui/tui/settings.py` for the current
measured values, and its comment for how to re-measure after changing
`OLLAMA_NUM_CTX` or the model set. Don't state a model "fits"/"doesn't fit"
from weight size alone.

## Comment style

Comments explain WHY — the goal a piece of code serves, the constraint it
works around, the invariant it protects — not the specific debugging
session that led to it. Don't narrate a "live run"/incident as a story
("on 2026-08-11 the model did X, then Y happened, log showed Z..."); state
the resulting rule/reasoning directly ("X must happen before Y, otherwise
Z" — no need to say how that was discovered). If a concrete number or
threshold genuinely needs a data point to justify it, give the number
itself, not the narrative around measuring it.

## Live test runs

Qwen3 models think by default — on this machine that turned a single short
reply into 1722 tokens / 368s of generation instead of 21 tokens / 2.6s (66x
slower), with zero benefit since nothing consumes that reasoning during a
manual test run. When running live tests (`src/mcp_agent/run_cli.py` or
similar one-off invocations), always disable thinking — `ChatOllama(..., reasoning=False)`
or `"think": false` in raw Ollama API calls — otherwise every iteration pays
for a huge, invisible reasoning trace.
