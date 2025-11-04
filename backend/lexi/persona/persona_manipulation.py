# /lexi/persona/persona_manipulation.py

"""
persona_manipulation.py

Lexi's core strategy for *gently and intentionally* influencing ("manipulating") the user's emotional/persona axis vector
toward healthier, happier, more positive states.

This module:
    - Defines "healthy" axis targets (can be customized per axis)
    - Sets per-axis "nudge" rates (how strongly Lexi influences per turn)
    - Provides a function to calculate Lexi's ideal persona axis vector
      for the next response, as a gentle nudge from the user's current axis vector.

Intended for standalone import and plug-in to persona logic.
"""

from typing import Dict
from ..utils.emotion_axis import nudge_axis_toward

# === Define per-axis "healthy" target values ===
HEALTHY_AXIS_TARGETS = {
    "joy": 1.0,  # Maximum joy is the goal
    "anger": 0.0,  # Minimum anger
    "affection": 1.0,  # Maximum affection
    "energy": 1.0,  # Maximum energy
    "warmth": 1.0,  # Maximum warmth
    "chaos": 0.5,  # Middle is healthy for chaos
    # Add more axes as you grow the system
}

# === Per-axis "nudge rates" (how fast Lexi manipulates the user's state per turn) ===
NUDGE_RATES = {
    "joy": 0.04,
    "anger": 0.03,
    "affection": 0.03,
    "energy": 0.02,
    "warmth": 0.04,
    "chaos": 0.01,
    # Extend as needed
}


def get_persona_nudge_vector(
    user_current: Dict[str, float], user_baseline: Dict[str, float] = None
) -> Dict[str, float]:
    """
    For a given user axis vector, compute Lexi's "response" axis vector,
    gently nudging each axis toward its healthy target value.

    Optionally, blends current and baseline user axis for more stable manipulation.

    Args:
        user_current: Dict[axis, value] (from this turn)
        user_baseline: Dict[axis, value] (optional, user persistent baseline)

    Returns:
        Dict[axis, value] - Lexi's target axis vector for next reply
    """
    if user_baseline is None:
        user_baseline = user_current

    result = {}
    for axis, user_val in user_current.items():
        # Optionally blend current and baseline:
        # blended = 0.7 * user_val + 0.3 * user_baseline.get(axis, user_val)
        blended = user_val
        target = HEALTHY_AXIS_TARGETS.get(axis, 1.0)
        rate = NUDGE_RATES.get(axis, 0.03)
        result[axis] = nudge_axis_toward(blended, target, rate)
    return result


# === CLI Test Harness ===
if __name__ == "__main__":
    test_user = {
        "joy": 0.1,
        "anger": 0.8,
        "affection": 0.3,
        "energy": 0.2,
        "warmth": 0.4,
        "chaos": 0.6,
    }
    print("User axis now:", test_user)
    print("Lexi nudge axis:", get_persona_nudge_vector(test_user))
