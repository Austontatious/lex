#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-backend}
export HOST=${HOST:-127.0.0.1}
export PORT=${PORT:-9000}
export LEX_MEMORY_PATH=${LEX_MEMORY_PATH:-./tmp/test_memory.jsonl}

mkdir -p "$(dirname "$LEX_MEMORY_PATH")"
touch "$LEX_MEMORY_PATH"

exec uvicorn lexi.core.backend_core:app --host "$HOST" --port "$PORT"
