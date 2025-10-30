#!/bin/sh
set -eu

pytest -q

PORT="${SMOKE_PORT:-8099}"
WAIT_SECONDS="${SMOKE_WAIT:-30}"

uvicorn lexi.core.backend_core:app --port "$PORT" --lifespan on --log-level warning &
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT

sleep "$WAIT_SECONDS"
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
