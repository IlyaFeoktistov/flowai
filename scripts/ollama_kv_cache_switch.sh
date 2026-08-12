#!/bin/bash
# Switches the running `ollama` systemd service's OLLAMA_KV_CACHE_TYPE and
# restarts it -- called via sudo by agent_builder.py (see
# mcp_agent/model_config.py:_kv_cache_type_for) right before building the
# chat model, so the daemon-wide KV cache type always matches whichever
# model is about to run:
#   - gpt-oss:20b crashes on load under q8_0 (GGML_ASSERT, a known Ollama
#     bug, github.com/ollama/ollama/issues/16946) -- needs f16.
#   - qwen3-coder:30b (and everything else) uses q8_0 -- half the KV-cache
#     RAM/VRAM footprint at this project's OLLAMA_NUM_CTX=65536.
#
# Installed via a narrow NOPASSWD sudoers rule (see README's "Системные
# пререквизиты") that allows running ONLY this exact script path as root --
# not systemctl/tee/etc. in general -- so a compromised flowAI process can't
# leverage the rule for anything beyond flipping this one setting.
set -euo pipefail

VALUE="${1:-}"
case "$VALUE" in
    f16|q8_0) ;;
    *) echo "usage: $0 f16|q8_0" >&2; exit 1 ;;
esac

DROPIN_DIR=/etc/systemd/system/ollama.service.d
DROPIN_FILE="$DROPIN_DIR/flowai-kv-cache.conf"

current=""
if [ -f "$DROPIN_FILE" ]; then
    current=$(grep -oP 'OLLAMA_KV_CACHE_TYPE=\K\S+' "$DROPIN_FILE" || true)
fi

if [ "$current" = "$VALUE" ]; then
    exit 0
fi

mkdir -p "$DROPIN_DIR"
printf '[Service]\nEnvironment=OLLAMA_KV_CACHE_TYPE=%s\n' "$VALUE" > "$DROPIN_FILE"

systemctl daemon-reload
systemctl restart ollama
