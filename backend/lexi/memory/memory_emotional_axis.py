# /lex/memory/emotional_axis_memory.py

"""
emotional_axis_memory.py

Persists and retrieves per-user emotional/persona axis vectors ("baselines") across sessions.
Optionally logs all axis vector snapshots with timestamp for analytics and visualization.

- Baseline: long-term, slowly changing fingerprint per user
- Log: all snapshots by timestamp (optional, for analytics)

Usage:
    from lexi.memory.emotional_axis_memory import (
        get_user_axis_baseline,
        update_user_axis_baseline,
        log_user_axis_vector,
        load_user_axis_log
    )
"""

import os
import json
import time
from typing import Dict, Optional, List

STORAGE_PATH = os.path.join(os.path.dirname(__file__), "emotional_axis_store.json")
LOG_PATH = os.path.join(os.path.dirname(__file__), "emotional_axis_log.json")


def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# === Baseline vector: slow-changing fingerprint ===
def get_user_axis_baseline(user_id: str) -> Optional[Dict[str, float]]:
    store = _load_json(STORAGE_PATH)
    return store.get(user_id)


def update_user_axis_baseline(user_id: str, new_vector: Dict[str, float]) -> None:
    store = _load_json(STORAGE_PATH)
    store[user_id] = new_vector
    _save_json(STORAGE_PATH, store)


# === Logging axis snapshots by timestamp (for analytics/journaling) ===
def log_user_axis_vector(
    user_id: str, vector: Dict[str, float], timestamp: Optional[float] = None
) -> None:
    if timestamp is None:
        timestamp = time.time()
    log = _load_json(LOG_PATH)
    if user_id not in log:
        log[user_id] = []
    log[user_id].append({"timestamp": timestamp, "axis_vector": vector})
    _save_json(LOG_PATH, log)


def load_user_axis_log(user_id: str) -> List[Dict]:
    log = _load_json(LOG_PATH)
    return log.get(user_id, [])


# === CLI/test harness ===
if __name__ == "__main__":
    test_uid = "demo_user"
    v = {"joy": 0.7, "anger": 0.2, "affection": 0.8, "energy": 0.9, "warmth": 0.7, "chaos": 0.5}
    update_user_axis_baseline(test_uid, v)
    print("Baseline for user:", get_user_axis_baseline(test_uid))
    log_user_axis_vector(test_uid, v)
    print("Axis log for user:", load_user_axis_log(test_uid))
