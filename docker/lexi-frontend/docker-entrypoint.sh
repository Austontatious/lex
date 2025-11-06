#!/bin/sh
set -e

API_BASE="${LEX_API_BASE:-${LEX_API_BASE_PUBLIC:-https://api.lexicompanion.com/lexi}}"
COMFY="${COMFY_URL:-http://172.17.0.1:8188}"
VLLM="${VLLM_URL:-http://172.17.0.1:8008/v1}"

cat <<EOF >/usr/share/nginx/html/runtime-config.js
window.__LEX_API_BASE="${API_BASE}";
window.__COMFY_URL="${COMFY}";
window.__VLLM_URL="${VLLM}";
EOF

cat <<EOF >/usr/share/nginx/html/config.json
{ "API_BASE": "${API_BASE}" }
EOF

exec "$@"
