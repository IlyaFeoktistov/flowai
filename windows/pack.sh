#!/usr/bin/env bash
# Собирает архив flowAI для передачи на Windows.
# Запуск: bash windows/pack.sh
# Результат: flowAI-windows.zip в папке проекта.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/flowAI-windows.tar.gz"

cd "$HERE"

tar -czf "$OUT" \
    --exclude="./.venv" \
    --exclude="./.git" \
    --exclude="./__pycache__" \
    --exclude="*/__pycache__" \
    --exclude="./generated" \
    --exclude="./memory.json" \
    --exclude="./.env" \
    --exclude="*.pyc" \
    --exclude="*.pyo" \
    .

echo "Готово: $OUT"
echo "Размер: $(du -sh "$OUT" | cut -f1)"
