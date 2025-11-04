#!/usr/bin/env bash
set -euo pipefail

# Simple dev launcher with logs + health checks
# - Starts vLLM (8008) and ComfyUI (8188) if not already running
# - Boots Lexi backend+frontend via start_lexi.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

# --- vLLM ---
if ! curl -fsS "http://127.0.0.1:8008/v1/models" >/dev/null 2>&1; then
  echo "[dev_up] starting vLLM on :8008"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
  nohup /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/models/Qwen/Qwen2.5-32B-AGI \
    --served-model-name Lexi \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.88 \
    --max-num-seqs 64 \
    --swap-space 12 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --download-dir /mnt/data/models \
    --disable-custom-all-reduce \
    --host 0.0.0.0 --port 8008 \
    >"$LOG_DIR/vllm.log" 2>&1 & echo $! >"$LOG_DIR/vllm.pid"
else
  echo "[dev_up] vLLM already responding on :8008"
fi

# --- ComfyUI ---
if ! curl -fsS "http://127.0.0.1:8188/queue" >/dev/null 2>&1; then
  echo "[dev_up] starting ComfyUI on :8188"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_COMFY:-4,5}
  pushd /mnt/data/comfy >/dev/null
  source venv/bin/activate
  nohup python3 main.py --listen 0.0.0.0 --port 8188 \
    >"$LOG_DIR/comfy.log" 2>&1 & echo $! >"$LOG_DIR/comfy.pid"
  popd >/dev/null
else
  echo "[dev_up] ComfyUI already responding on :8188"
fi

# --- Lexi backend + frontend ---
echo "[dev_up] starting Lexi (backend+frontend)"
if [ -d "${ROOT_DIR}/.venv" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi
nohup python3 "${ROOT_DIR}/start_lexi.py" \
  >"$LOG_DIR/start_lexi.log" 2>&1 & echo $! >"$LOG_DIR/start_lexi.pid"
sleep 2
echo "[dev_up] PIDs:"; cat "$LOG_DIR"/*.pid 2>/dev/null || true
echo "[dev_up] Logs: $LOG_DIR"

