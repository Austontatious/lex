#!/usr/bin/env bash
set -euo pipefail

echo "== Processes =="
ps -ef | egrep "vllm\.entrypoints\.openai\.api_server|comfy.*main\.py|start_lexi\.py" | grep -v grep || true

echo "\n== Ports =="
{ ss -tulpn 2>/dev/null || netstat -tulpn 2>/dev/null; } | egrep ":8008|:8188|:8000|:3000" || true

echo "\n== Health =="
curl -fsS http://127.0.0.1:8008/v1/models | head -n 3 || echo "vLLM: down"
curl -fsS http://127.0.0.1:8188/queue | head -n 3 || echo "Comfy: down"
curl -fsS http://127.0.0.1:8000/lexi/health || echo "Backend: down"
echo

