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

- `agent.py` — main agentic loop, Ollama streaming, tool orchestration
- `tools/` — tool handlers: bash_exec, file_ops, web_search, read_page, image_gen, memory
- `ui/` — terminal UI: stream display, prompt_toolkit input, Rich console
- `settings.py` — model selection, GPU routing (persisted in SQLite, `~/.local/share/flowai/`)
- `memory.py` — persistent user memory (SQLite, `~/.local/share/flowai/`)
- `FLOWAI.md` (optional, in the target working directory) — project-specific instructions the agent reads and follows, appended to the system prompt if present

## Models

RTX 4050 Laptop (5.9 GB VRAM). Current default (`.env.example`/`settings.py`)
is `qwen3-coder:30b` — MoE, ~3.3B active params, mostly runs on CPU (see
`mcp_agent/model_config.py:OLLAMA_NUM_CTX` comments for the live-measured
CPU/GPU split).

Whether a model "fits fully on GPU" is NOT just weights-size vs. VRAM — the
agent always loads models with `OLLAMA_NUM_CTX=65536`, and the KV-cache at
that context size is often several GB on top of the weights, easily pushing
a model that looks like it should fit onto partial CPU offload instead. The
only reliable answer is a live measurement (`ollama ps` right after a real
request) — see `_MEASURED_GPU_SHARE` in `ui/tui/settings.py` for the current
measured values, and its comment for how to re-measure after changing
`OLLAMA_NUM_CTX` or the model set. Don't state a model "fits"/"doesn't fit"
from weight size alone.

## Live test runs

Qwen3 models think by default — on this machine that turned a single short
reply into 1722 tokens / 368s of generation instead of 21 tokens / 2.6s (66x
slower), with zero benefit since nothing consumes that reasoning during a
manual test run. When running live tests (`mcp_agent/run_cli.py` or similar
one-off invocations), always disable thinking — `ChatOllama(..., reasoning=False)`
or `"think": false` in raw Ollama API calls — otherwise every iteration pays
for a huge, invisible reasoning trace.
