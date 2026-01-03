#!/usr/bin/env bash
set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:9000}"
GEN_URL="$BACKEND_URL/lexi/gen/avatar"
PROMPT="${PROMPT:-cozy cyberpunk librarian portrait, soft rim light}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-1216}"
STEPS="${STEPS:-20}"
GUIDANCE="${GUIDANCE:-3.5}"
SAMPLER="${SAMPLER:-euler}"
SCHEDULER="${SCHEDULER:-normal}"

TMP_HDRS="$(mktemp)"
TMP_BODY="$(mktemp)"
TMP_PAYLOAD="$(mktemp)"
trap 'rm -f "$TMP_HDRS" "$TMP_BODY" "$TMP_PAYLOAD"' EXIT

python3 - <<'PY' "$TMP_PAYLOAD" "$PROMPT" "$WIDTH" "$HEIGHT" "$STEPS" "$GUIDANCE" "$SAMPLER" "$SCHEDULER"
import json, sys
dst, prompt, width, height, steps, guidance, sampler, scheduler = sys.argv[1:]
body = {
    "prompt": prompt,
    "backend": "flux",
    "width": int(width),
    "height": int(height),
    "steps": int(steps),
    "guidance": float(guidance),
    "sampler": sampler,
    "scheduler": scheduler,
}
with open(dst, "w", encoding="utf-8") as f:
    json.dump(body, f)
PY

echo "==> Request 1 (no session header): expect backend to auto-create session"
HTTP_CODE=$(curl -sS -D "$TMP_HDRS" -o "$TMP_BODY" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  --data-binary @"$TMP_PAYLOAD" \
  "$GEN_URL" || true)

echo "HTTP: $HTTP_CODE"
grep -i '^x-lexi-session:' "$TMP_HDRS" || true
SESSION=$(awk -F': ' 'BEGIN{IGNORECASE=1}/^x-lexi-session:/{print $2}' "$TMP_HDRS" | tr -d '\r')
COOKIE=$(awk -F': ' 'BEGIN{IGNORECASE=1}/^set-cookie:/{print $2}' "$TMP_HDRS" | tr -d '\r')

echo "==> Body:"
cat "$TMP_BODY"; echo

if [[ -z "${SESSION:-}" ]]; then
  echo "WARN: No X-Lexi-Session header found; will try cookie fallback."
  SESSION=$(echo "$COOKIE" | sed -n 's/.*lex_session=\([^;]*\).*/\1/p' || true)
fi

if [[ -z "${SESSION:-}" ]]; then
  echo "ERROR: Could not obtain session id from header or cookie."
  exit 2
fi

echo "==> Session: $SESSION"

echo "==> Request 2 (reuse same session)"
HTTP_CODE2=$(curl -sS -o "$TMP_BODY" -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -H "X-Lexi-Session: $SESSION" \
  --data-binary @"$TMP_PAYLOAD" \
  "$GEN_URL" || true)

echo "HTTP2: $HTTP_CODE2"
cat "$TMP_BODY"; echo

echo "==> Tail backend logs (last 120 lines)"
docker logs --tail 120 lex-lexi-backend-1 || true

echo "==> Check Comfy queue/status"
docker exec lex-lexi-backend-1 sh -lc '
  COMFY_URL=${COMFY_URL:-http://comfy-sd:8188}
  if command -v jq >/dev/null 2>&1; then
    curl -s "$COMFY_URL/queue/status" | jq . || true
  else
    echo "jq missing; raw response:"
    curl -s "$COMFY_URL/queue/status" || true
  fi
'

echo "==> List latest outputs"
docker exec lex-lexi-backend-1 sh -lc '
  ls -lt /mnt/data/comfy/output 2>/dev/null | head -n 20 || true
'
