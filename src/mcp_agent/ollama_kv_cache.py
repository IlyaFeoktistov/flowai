"""
Keeps the system-wide `ollama` daemon's OLLAMA_KV_CACHE_TYPE matched to
whichever text model is about to run -- a single env var for the whole
daemon, not a per-request Ollama API option, so switching models means
switching this and restarting the daemon.

Live-confirmed root cause (2026-08-11 journalctl trail): gpt-oss:20b ran
fine under OLLAMA_KV_CACHE_TYPE=f16, then someone flipped the systemd unit
to q8_0 for unrelated reasons (smaller KV cache for qwen3-coder:30b at this
project's OLLAMA_NUM_CTX=65536) -- the very next gpt-oss load crashed with
GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor)), a known Ollama bug
(github.com/ollama/ollama/issues/16946) triggered by gpt-oss's tensor shapes
under q8_0 specifically. qwen3-coder:30b has no such issue.

_switch_ollama_kv_cache_type() shells out via sudo to
scripts/ollama_kv_cache_switch.sh, which rewrites a systemd drop-in
(/etc/systemd/system/ollama.service.d/flowai-kv-cache.conf) and restarts
`ollama` -- requires a one-time NOPASSWD sudoers rule scoped to exactly that
script path (see README's "Системные пререквизиты"), never a broad
`ALL=(ALL) NOPASSWD: ALL`-style grant. Without that rule, sudo prompts for a
password non-interactively and fails instantly -- ensure_kv_cache_type
returns (False, reason) in that case rather than hanging the turn, same
fail-open contract as expert_streaming.ensure_running.
"""
import subprocess
from pathlib import Path

# One extra .parent — this file lives one level deeper under src/
# (src/mcp_agent/ollama_kv_cache.py) now, but scripts/ is a real repo-root
# directory that never moved there.
FLOWAI_ROOT = Path(__file__).resolve().parent.parent.parent
SWITCH_SCRIPT = FLOWAI_ROOT / "scripts" / "ollama_kv_cache_switch.sh"

# Models known to crash under OLLAMA_KV_CACHE_TYPE=q8_0 (GGML_ASSERT, see
# module docstring) -- everything else keeps the smaller q8_0 footprint.
_REQUIRES_F16 = {"gpt-oss:20b"}

# What we last successfully switched the daemon to, THIS PROCESS's lifetime
# -- avoids a sudo round-trip (and, if the value actually needs to change,
# an ollama restart that unloads every resident model) on every single
# turn when nothing changed. Not a substitute for the switch script's own
# idempotency check (some other process/manual edit could've changed the
# real state) -- just skips redundant work in the common case.
_last_applied: str | None = None
_last_failure: str | None = None


def _kv_cache_type_for(model_tag: str) -> str:
    return "f16" if model_tag in _REQUIRES_F16 else "q8_0"


def ensure_kv_cache_type(model_tag: str) -> tuple[bool, str]:
    """Makes sure the ollama daemon's KV cache type matches `model_tag`'s
    requirement, switching (and restarting the daemon) if needed. Returns
    (ok, message) -- callers (agent_builder._build_chat_model) should treat
    a False here the same way they treat expert_streaming's own failures:
    log a warning and fall back to Ollama's current (possibly wrong-for-
    this-model) config rather than raise, since a wrong KV cache type is a
    load-time crash for gpt-oss specifically, not a startup blocker for
    every model."""
    global _last_applied, _last_failure
    target = _kv_cache_type_for(model_tag)
    if _last_applied == target:
        return True, "already applied"

    if not SWITCH_SCRIPT.is_file():
        _last_failure = f"switch script not found: {SWITCH_SCRIPT}"
        return False, _last_failure

    try:
        result = subprocess.run(
            ["sudo", "-n", str(SWITCH_SCRIPT), target],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _last_failure = f"не удалось запустить {SWITCH_SCRIPT}: {e}"
        return False, _last_failure

    if result.returncode != 0:
        _last_failure = (
            f"sudo {SWITCH_SCRIPT.name} {target} упал (код {result.returncode}) — "
            f"{(result.stderr or result.stdout or '').strip()[:300]} — "
            "проверь sudoers-правило из README"
        )
        return False, _last_failure

    _last_applied = target
    _last_failure = None
    return True, "switched"
