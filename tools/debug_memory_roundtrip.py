#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests


def sanitize_user_id(user_id: str) -> str:
    cleaned = (user_id or "").strip().lower()
    cleaned = cleaned.replace(" ", "-")
    cleaned = re.sub(r"[^a-z0-9_.@-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "anon"


def post_prompt(base_url: str, user_id: str, prompt: str) -> dict:
    resp = requests.post(
        f"{base_url}/lexi/process",
        json={"prompt": prompt},
        headers={"X-Lexi-User": user_id},
        timeout=30,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Request failed ({resp.status_code}): {resp.text}")
    return resp.json()


def main() -> int:
    base_url = os.getenv("LEXI_DEBUG_BASE", "http://127.0.0.1:8000").rstrip("/")
    user_id = os.getenv("LEXI_DEBUG_USER", "tester@example.com")
    memory_root = Path(
        os.getenv("LEXI_MEMORY_ROOT", "/mnt/data/Lex/data/memory")
    ).expanduser()
    os.environ.setdefault("LEXI_USER_ID_ENABLED", "1")
    os.environ.setdefault("LEXI_MEMORY_ROOT", str(memory_root))

    print(f"[debug] base_url={base_url}")
    print(f"[debug] user_id={user_id}")
    print(f"[debug] memory_root={memory_root}")

    first = post_prompt(base_url, user_id, "Remember that my dog is Juniper.")
    time.sleep(0.5)
    second = post_prompt(base_url, user_id, "What is my dog's name? Don't guess.")

    reply = (second.get("cleaned") or "").lower()
    if "juniper" not in reply:
        raise AssertionError(f"Expected 'Juniper' in reply, got: {second.get('cleaned')}")

    user_dir = memory_root / "users" / sanitize_user_id(user_id)
    ltm_path = user_dir / "ltm.jsonl"
    summary_path = user_dir / "session_summaries.json"

    if not user_dir.exists():
        raise AssertionError(f"User directory missing: {user_dir}")
    if not summary_path.exists():
        raise AssertionError(f"session_summaries.json missing: {summary_path}")

    summary_doc = json.loads(summary_path.read_text(encoding="utf-8"))
    rolling = (summary_doc.get("rolling_summary") or "")
    facts = summary_doc.get("facts") or {}
    if "juniper" not in rolling.lower() and "juniper" not in json.dumps(facts).lower():
        raise AssertionError("Expected Juniper in rolling_summary or facts.")

    if not ltm_path.exists():
        print("[warn] ltm.jsonl missing; session summaries still look good.")

    print("[ok] memory roundtrip succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
