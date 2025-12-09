#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1,2,3,4

exec /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged \
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
  --enable-auto-tool-choice \
  --tool-call-parser openai \
  --host 0.0.0.0 \
  --port 8008
