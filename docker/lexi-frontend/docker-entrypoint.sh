#!/bin/sh
set -e

API_BASE="${LEX_API_BASE:-/lex}"
COMFY="${COMFY_URL:-http://host.docker.internal:8188}"
VLLM="${VLLM_URL:-http://host.docker.internal:8008/v1}"

cat <<EOF >/usr/share/nginx/html/runtime-config.js
window.__LEX_API_BASE="${API_BASE}";
window.__COMFY_URL="${COMFY}";
window.__VLLM_URL="${VLLM}";
EOF

exec "$@"
