# Lexi vLLM Service (`lexi-vllm.service`) — Legacy Host Fallback

This server runs the Lexi vLLM OpenAI-compatible API as a **legacy** systemd service.
The default path is now the Compose `vllm` service; use `COMPOSE_PROFILES=no-vllm docker compose up -d`
if you need to run this host service instead.

## Service basics

Service name:

    lexi-vllm.service

Script location:

    /mnt/data/Lex/scripts/run_vllm_vllm.sh

Working directory:

    /mnt/data/Lex

The script launches vLLM with:

    CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
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

## Start / Stop / Restart / Status

Start:

    sudo systemctl start lexi-vllm.service

Stop:

    sudo systemctl stop lexi-vllm.service

Restart:

    sudo systemctl restart lexi-vllm.service

Check status:

    sudo systemctl status lexi-vllm.service

## Logs

Follow logs in real time:

    sudo journalctl -u lexi-vllm.service -f

Show recent logs:

    sudo journalctl -u lexi-vllm.service --since "1 hour ago"

## Check that it’s listening

Port:

    8008 (OpenAI-compatible HTTP API)

Check with ss:

    ss -tulpn | grep 8008

Check with curl from the server:

    curl http://localhost:8008/v1/models

If that returns a JSON blob with model info, Lexi vLLM is up.

## Enable / Disable on boot

Enable (start automatically on boot):

    sudo systemctl enable lexi-vllm.service

Disable (no auto-start on boot):

    sudo systemctl disable lexi-vllm.service
