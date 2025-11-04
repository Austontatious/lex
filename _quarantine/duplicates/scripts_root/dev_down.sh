#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

for name in vllm comfy start_lexi; do
  pidfile="$LOG_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile" || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[dev_down] stopping $name (pid $pid)"
      kill "$pid" || true
    fi
    rm -f "$pidfile"
  fi
done
echo "[dev_down] done"

