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

# 📌 Central mode registry with structured data
PERSONA_MODE_REGISTRY = {}

PERSONA_AXES = data.get("axes", []) or ["joy", "anger", "affection", "energy", "warmth", "chaos"]

PERSONA_MODE_REGISTRY = {}
for mode in data.get("modes", []):
    PERSONA_MODE_REGISTRY[mode["id"]] = {
        "id": mode["id"],
        "name": mode.get("name", mode["id"]),
        "description": mode.get("description", ""),
        "avatar_prompt": mode.get("avatar_prompt", ""),
        "trigger": re.compile(mode.get("trigger", ""), re.I) if mode.get("trigger") else None,
        "imperative_required": mode.get("imperative_required", False),
        "axis_vector": mode.get("axis_vector"),
        "system_prompt": mode.get("system_prompt", ""),  # safe default even if empty
        "goal_vector": mode.get("goal_vector", ""),  # optional, used by set_mode()
        "traits": mode.get("traits", {}),  # optional per-mode traits
    }


def get_mode_description(mode_id: str) -> str:
    return PERSONA_MODE_REGISTRY.get(mode_id, {}).get("description", "")


def get_avatar_prompt_for_mode(mode_id: str) -> str:
    return PERSONA_MODE_REGISTRY.get(mode_id, {}).get("avatar_prompt", "")


def get_mode_trigger(mode_id: str):
    return PERSONA_MODE_REGISTRY.get(mode_id, {}).get("trigger")


def mode_requires_imperative(mode_id: str) -> bool:
    return PERSONA_MODE_REGISTRY.get(mode_id, {}).get("imperative_required", False)


def get_mode_axis_vector(mode_id: str):
    return PERSONA_MODE_REGISTRY.get(mode_id, {}).get("axis_vector")


def get_persona_axes() -> list:
    return PERSONA_AXES


def assemble_avatar_prompt(base_style: str, details: str) -> str:
    return ", ".join([base_style.strip(), details.strip()])
