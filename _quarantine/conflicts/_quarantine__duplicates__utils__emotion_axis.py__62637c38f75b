# /lex/utils/emotion_axis.py

"""
emotion_axis.py

Unified emotion/persona axis scoring for user & AI (Lexi), now with:
- Pattern-based heuristics (legacy/fast fallback)
- LLM-first axis regression, with surface cues auto-extracted and injected
- Baseline blending and nudging
- All axis/scoring logic centralized

Plug in your best LLM, and improvement is just a better model or prompt away.
"""

import re
import json
from typing import Dict, Callable, Optional, List, Tuple
from .user_signal import extract_surface_cues  # <---- NEW import

AXES = [
    "joy",         # sadness <-> joy
    "anger",       # calm <-> angry
    "affection",   # distant <-> affectionate
    "energy",      # tired <-> energetic
    "warmth",      # cold <-> warm
    "chaos",       # orderly <-> chaotic
    # Extend as desired
]

# Pattern-based rules (kept for fallback/local scoring)
EMOTION_RULES: List[Tuple[re.Pattern, str, float, Optional[Callable[[str], bool]]]] = [
    # ... [same as previous] ...
]

def infer_emotion_axes(
    text: str,
    axes: Optional[List[str]] = None,
    rules: Optional[List[Tuple[re.Pattern, str, float, Optional[Callable[[str], bool]]]]] = None
) -> Dict[str, float]:
    """
    Pattern-based scoring (fast fallback, not usually used if LLM available).
    """
    if axes is None:
        axes = AXES
    if rules is None:
        rules = EMOTION_RULES
    scores = {axis: 0.5 for axis in axes}
    for pattern, axis, weight, condition in rules:
        if axis not in axes: continue
        if pattern.search(text) and (condition is None or condition(text)):
            scores[axis] += weight
    for axis in axes:
        scores[axis] = min(1.0, max(0.0, scores[axis]))
    return scores

# ======= LLM-First: Surface cues auto-extracted and injected =======
def infer_emotion_axes_llm(
    chat_history: str,
    current_message: str,
    axes: Optional[List[str]] = None,
    baseline: Optional[Dict[str, float]] = None,
    llm_func: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """
    LLM-based axis inference, with auto-injected surface cues for the current message.

    Args:
        chat_history: String of recent dialog (for context)
        current_message: Most recent user (or AI) message
        axes: Axis list (defaults to AXES)
        baseline: Persistent baseline (optional, for context)
        llm_func: (prompt: str) -> str (should return JSON dict of axis: value)
    Returns:
        Dict[axis, value in 0..1]
    """
    if axes is None:
        axes = AXES
    if llm_func is None:
        raise ValueError("Must supply an LLM function.")

    cues = extract_surface_cues(current_message)
    cues_json = json.dumps(cues, indent=2)

    axis_desc = "\n".join([f"- {a}" for a in axes])
    baseline_str = f"\nUser's baseline axis vector: {json.dumps(baseline)}" if baseline else ""

    # ---- Standard LLM prompt for axis scoring ----
    prompt = (
        "You are an expert in emotional intelligence. Score the following emotional axes (0=low, 1=high)\n"
        "for the user's current message, using:\n"
        "- The user's persistent baseline\n"
        "- The recent chat context\n"
        "- Non-verbal surface cues (punctuation, cursing, all-caps, etc.)\n\n"
        f"Axes:\n{axis_desc}\n"
        f"{baseline_str}\n"
        "Recent chat context (last few turns):\n"
        f"{chat_history}\n\n"
        f"Surface cues for current message:\n{cues_json}\n\n"
        "Now, infer the *user's current* axis scores as a JSON dict of axis: value (all in [0,1]):"
    )
    resp = llm_func(prompt)
    try:
        axis_scores = json.loads(resp)
        for axis in axes:
            if axis in axis_scores:
                axis_scores[axis] = min(1.0, max(0.0, float(axis_scores[axis])))
        return {a: axis_scores.get(a, 0.5) for a in axes}
    except Exception as e:
        # Fallback to neutral if LLM fails or malforms JSON
        print(f"[emotion_axis] LLM parse fail: {e} - resp: {resp}")
        return {a: 0.5 for a in axes}

# ======= Baseline/Blending/Nudging (as before) =======
def update_baseline(baseline, new_score, alpha=0.05):
    updated = {}
    axes = set(baseline) | set(new_score)
    for axis in axes:
        old = baseline.get(axis, 0.5)
        new = new_score.get(axis, 0.5)
        blended = (1 - alpha) * old + alpha * new
        updated[axis] = min(1.0, max(0.0, blended))
    return updated

def nudge_axis_toward(value: float, target: float = 1.0, rate: float = 0.03) -> float:
    return min(1.0, max(0.0, value + rate * (target - value)))

def nudge_vector_toward(vector: Dict[str, float], target: float = 1.0, rate: float = 0.03) -> Dict[str, float]:
    return {axis: nudge_axis_toward(val, target, rate) for axis, val in vector.items()}

# ======= CLI test harness =======
if __name__ == "__main__":
    print("AXES:", AXES)
    print("Type a message (with !, caps, cursing, etc) to see surface cues extracted. LLM scoring requires a function.")
    while True:
        msg = input("Text> ").strip()
        print("Surface cues:", extract_surface_cues(msg))
        print("Pattern-based axis scores:", infer_emotion_axes(msg))

