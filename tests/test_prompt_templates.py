from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.lexi.config.prompt_templates import SYSTEM_PROMPT_PATH, load_system_prompt_template


def test_lexi_system_prompt_qwen3_moe_loads():
    path = Path(os.getenv("LEXI_SYSTEM_PROMPT_PATH", str(SYSTEM_PROMPT_PATH))).expanduser()
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()

    assert text
    assert "SOFT-REDIRECT KIT" in text or "Soft-Redirect Kit" in text
    forbidden = [
        "I'm sorry, I can't assist with that request",
        "I cannot help with that request",
    ]
    for phrase in forbidden:
        assert phrase not in text

    for placeholder in [
        "{now}",
        "{user_name}",
        "{mode}",
        "{traits}",
        "{recent_memories}",
        "{session_summary}",
        "{context_window_hint}",
    ]:
        assert placeholder in text

    loaded = load_system_prompt_template()
    assert loaded.strip()
