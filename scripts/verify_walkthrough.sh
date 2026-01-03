#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:9000}"

echo "[verify] checking /lexi/alpha/tour/script at ${BASE_URL}"
SCRIPT_JSON="$(curl -sS "${BASE_URL}/lexi/alpha/tour/script")"

INTRO="$(printf '%s' "$SCRIPT_JSON" | jq -r '.onboarding.intro // empty')"
DISCLAIMER="$(printf '%s' "$SCRIPT_JSON" | jq -r '.onboarding.disclaimer_short // empty')"
STEP_ID="$(printf '%s' "$SCRIPT_JSON" | jq -r '.onboarding.steps[0].id // empty')"

if [[ -z "$INTRO" ]]; then
  echo ":: error :: onboarding intro missing"
  exit 1
fi

if [[ -z "$DISCLAIMER" ]]; then
  echo ":: error :: onboarding disclaimer_short missing"
  exit 1
fi

if [[ "$STEP_ID" != "preview" ]]; then
  echo ":: error :: expected first step id 'preview' but got '${STEP_ID}'"
  exit 1
fi

echo "[verify] onboarding copy looks good."

echo "[verify] starting alpha session for memory test"
SESSION_JSON="$(curl -sS -X POST "${BASE_URL}/lexi/alpha/session/start" -H 'content-type: application/json' -d '{}')"
SESSION_ID="$(printf '%s' "$SESSION_JSON" | jq -r '.session_id // empty')"

if [[ -z "$SESSION_ID" ]]; then
  echo ":: error :: unable to start session"
  exit 1
fi

curl -sS -X POST "${BASE_URL}/lexi/alpha/tour/memory" \
  -H "Content-Type: application/json" \
  -H "X-Lexi-Session: ${SESSION_ID}" \
  -d '{"note":"walkthrough verification ping"}' \
  | jq .

curl -sS -X POST "${BASE_URL}/lexi/alpha/session/end" \
  -H "Content-Type: application/json" \
  -H "X-Lexi-Session: ${SESSION_ID}" \
  -d '{}' \
  > /dev/null

echo "[verify] memory endpoint responded successfully."
