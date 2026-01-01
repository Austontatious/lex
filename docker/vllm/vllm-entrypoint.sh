#!/usr/bin/env sh
set -e

# Ensure "python" exists so the canonical command works inside the image.
if ! command -v python >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    mkdir -p /usr/local/bin
    ln -sf "$(command -v python3)" /usr/local/bin/python
  fi
fi

exec "$@"
