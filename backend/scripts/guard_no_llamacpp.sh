#!/usr/bin/env bash
set -euo pipefail

if rg -n 'llama[_-]cpp[_-]python' -S requirements*.txt docker/**/requirements*.txt constraints*.txt 2>/dev/null; then
  echo "❌ llama-cpp-python detected in dependency files. Remove it before committing."
  exit 1
fi

echo "✅ No llama-cpp-python in dependency files."
