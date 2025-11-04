#!/usr/bin/env bash
set -euo pipefail

pkgs='facexlib|gfpgan|realesrgan|basicsr'
if rg -n "^\s*($pkgs)(==.*)?\s*$" docker/**/requirements*.txt requirements*.txt 2>/dev/null; then
  echo "❌ Face-restore deps found in backend requirements."
  exit 1
fi

echo "✅ No face-restore deps in backend requirements."
