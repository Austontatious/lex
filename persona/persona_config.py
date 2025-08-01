"""
DEPRECATED STRUCTURE: This module loads dynamic persona modes from JSON for backward compatibility.
"""
import json
import re
from pathlib import Path

# 🔧 Path to your persona_modes.json file
PERSONA_MODES_PATH = Path(__file__).parent / "persona_modes.json"

# ✅ Load JSON at runtime
with open(PERSONA_MODES_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# 📌 Dynamic replacements for legacy globals
PERSONA_MODES = {
    mode["id"]: mode["description"]
    for mode in data.get("modes", [])
}

MODE_TRIGGER_PATTERNS = {
    mode["id"]: re.compile(mode["trigger"], re.I)
    for mode in data.get("modes", [])
}

def get_avatar_prompt_for_mode(mode: str) -> str:
    return PERSONA_MODES.get(mode, {}).get("avatar_prompt", "")
    
def assemble_avatar_prompt(base_style: str, details: str) -> str:
    return ", ".join([base_style.strip(), details.strip()])


