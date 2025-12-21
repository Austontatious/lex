#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-backend}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-9000}
export LEXI_MEMORY_ROOT=${LEXI_MEMORY_ROOT:-./data/memory}

mkdir -p "$LEXI_MEMORY_ROOT"

exec uvicorn lexi.core.backend_core:app --host "$HOST" --port "$PORT"
