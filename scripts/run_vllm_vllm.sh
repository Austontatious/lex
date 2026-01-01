#!/usr/bin/env bash
set -e

# LEGACY: host-based vLLM fallback. Compose vLLM is the default path.

export CUDA_VISIBLE_DEVICES=0,1,2,3

exec /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged \
  --served-model-name Lexi \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-num-seqs 8 \
  --swap-space 12 \
  --max-model-len 32000 \
  --enforce-eager \
  --gpu-memory-utilization 0.88 \
  --host 0.0.0.0 \
  --port 8008
