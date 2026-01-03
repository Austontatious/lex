from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Canonical pose buckets (P1–P7) and camera buckets (C1–C7)
# Regex lists are intentionally “OR”-friendly for the intent layer.

POSE_BUCKETS = [
    {
        "code": "POSE_1",
        "label": "Standing Straight",
        "regex": r"(stand|standing straight|upright|neutral pose|straight posture|symmetrical stance|formal stance)",
        "feel": "professional",
    },
    {
        "code": "POSE_2",
        "label": "Standing Dynamic / Contrapposto",
        "regex": r"(contrapposto|hip tilt|weight on one leg|s-curve|glamour pose|model pose|leaning standing|arched hip)",
        "feel": "playful",
    },
    {
        "code": "POSE_3",
        "label": "Seated",
        "regex": r"(sit|sitting|seated|on a chair|on edge of bed|perched|cross[- ]legged sitting|upright sit)",
        "feel": "cozy",
    },
    {
        "code": "POSE_4",
        "label": "Kneeling / Crouched",
        "regex": r"(kneel|kneeling|on knees|crouched|low pose|one knee down|floor kneel|bed kneel)",
        "feel": "seductive",
    },
    {
        "code": "POSE_5",
        "label": "Lying Flat",
        "regex": r"(lying flat|laying flat|on back|on stomach|flat pose|reclined flat|spread out)",
        "feel": "seductive",
    },
    {
        "code": "POSE_6",
        "label": "Side-Lying / Curled",
        "regex": r"(side lying|curled|foetal pose|lounging on side|side recline|pin-up curl|side rest)",
        "feel": "cozy",
    },
    {
        "code": "POSE_7",
        "label": "Lounging / Ambiguous",
        "regex": r"(leaning|lounging|arched back|against wall|dramatic pose|artistic pose|cropped pose|foreshortened|twisted posture|complex pose|glam lounge)",
        "feel": "confident",
    },
]

CAMERA_BUCKETS = [
    {
        "code": "CAM_HIGH",
        "label": "High Angle",
        "regex": r"(high angle|looking down|bird’s eye|camera above|overhead \\(slight\\)|top-down angle)",
        "feel": "playful",
    },
    {
        "code": "CAM_SLIGHT_HIGH",
        "label": "Slight High Angle",
        "regex": r"(slight high angle|slightly above|gentle downward angle|soft high angle)",
        "feel": "cozy",
    },
    {
        "code": "CAM_EYE",
        "label": "Eye-Level",
        "regex": r"(eye level|straight on|direct angle|head[- ]on|neutral angle)",
        "feel": "cozy",
    },
    {
        "code": "CAM_SLIGHT_LOW",
        "label": "Slight Low Angle",
        "regex": r"(slight low angle|from below slightly|upward tilt small|low-ish angle)",
        "feel": "confident",
    },
    {
        "code": "CAM_LOW",
        "label": "Low Angle",
        "regex": r"(low angle|looking up|dramatic upward angle|camera below|powerful angle)",
        "feel": "confident",
    },
    {
        "code": "CAM_TOPDOWN",
        "label": "Top-Down / Overhead",
        "regex": r"(top-down|overhead|bird’s eye full|flat lay|looking down completely|above bed shot)",
        "feel": "seductive",
    },
    {
        "code": "CAM_UNKNOWN",
        "label": "Unknown / Artistic",
        "regex": r"(artistic angle|unclear angle|foreshortened camera|cropped view|stylized framing|ambiguous angle)",
        "feel": "playful",
    },
]


@dataclass(frozen=True)
class PoseCameraIntent:
    pose_bucket: Optional[str]
    camera_bucket: Optional[str]
    pose_feel: Optional[str]
    camera_feel: Optional[str]


def _first_match(buckets, text: str) -> tuple[Optional[str], Optional[str]]:
    for b in buckets:
        pat = re.compile(b["regex"], flags=re.IGNORECASE)
        if pat.search(text):
            return b["code"], b.get("feel")
    return None, None


def classify_pose_camera(text: str) -> PoseCameraIntent:
    text = text or ""
    pose_bucket, pose_feel = _first_match(POSE_BUCKETS, text)
    camera_bucket, camera_feel = _first_match(CAMERA_BUCKETS, text)
    return PoseCameraIntent(
        pose_bucket=pose_bucket,
        camera_bucket=camera_bucket,
        pose_feel=pose_feel,
        camera_feel=camera_feel,
    )


__all__ = [
    "POSE_BUCKETS",
    "CAMERA_BUCKETS",
    "PoseCameraIntent",
    "classify_pose_camera",
]
