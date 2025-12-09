#!/usr/bin/env python3
"""
Pose bucket + camera-angle bucket builder for Boudoir Glam keypoints.

Usage:
    python pose_bucket_builder.py \
        --input-dir ./BoudoirOutputWorking \
        --output-prefix boudoir_pose_buckets

Outputs:
    boudoir_pose_buckets.csv
    boudoir_pose_buckets.json

Pose buckets:
  POSE_1: standing straight
  POSE_2: standing / contrapposto / dynamic
  POSE_3: seated
  POSE_4: kneeling
  POSE_5: lying (back/stomach)
  POSE_6: lying (side)
  POSE_7: leaning / lounge / unknown

Camera buckets:
  CAM_HIGH
  CAM_SLIGHT_HIGH
  CAM_EYE
  CAM_SLIGHT_LOW
  CAM_LOW
  CAM_TOPDOWN   (horizontal / overhead-ish)
  CAM_UNKNOWN
"""

import os
import json
import math
import argparse
import csv
from typing import List, Tuple, Optional, Dict, Any

# COCO 18-keypoint order used by OpenPose
KEYPOINT_NAMES = [
    "nose",         # 0
    "neck",         # 1
    "r_shoulder",   # 2
    "r_elbow",      # 3
    "r_wrist",      # 4
    "l_shoulder",   # 5
    "l_elbow",      # 6
    "l_wrist",      # 7
    "r_hip",        # 8
    "r_knee",       # 9
    "r_ankle",      # 10
    "l_hip",        # 11
    "l_knee",       # 12
    "l_ankle",      # 13
    "r_eye",        # 14
    "l_eye",        # 15
    "r_ear",        # 16
    "l_ear",        # 17
]

CONF_THRESH = 0.05
Point = Tuple[float, float, float]  # x, y, conf


# ---------- basic helpers ----------

def load_keypoints(path: str) -> Optional[List[Point]]:
    """Load pose_keypoints_2d from an OpenPose JSON and return as list of (x,y,c)."""
    with open(path, "r") as f:
        data = json.load(f)

    people = data.get("people", [])
    if not people:
        return None

    arr = people[0].get("pose_keypoints_2d", [])
    if not arr or len(arr) % 3 != 0:
        return None

    pts: List[Point] = []
    for i in range(0, len(arr), 3):
        x, y, c = float(arr[i]), float(arr[i + 1]), float(arr[i + 2])
        pts.append((x, y, c))
    return pts


def get_point(pts: List[Point], idx: int) -> Optional[Point]:
    if idx < 0 or idx >= len(pts):
        return None
    x, y, c = pts[idx]
    if c < CONF_THRESH:
        return None
    return x, y, c


def avg_points(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, min(a[2], b[2]))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(a: Point, b: Point) -> float:
    """Angle in degrees between vector a->b and x-axis (0–180)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    if ang > 180:
        ang = 360 - ang
    if ang > 180:
        ang = 180
    return ang


def knee_angle(hip: Point, knee: Point, ankle: Point) -> float:
    """Angle at the knee joint in degrees (0–180)."""
    v1x, v1y = hip[0] - knee[0], hip[1] - knee[1]
    v2x, v2y = ankle[0] - knee[0], ankle[1] - knee[1]
    dot = v1x * v2x + v1y * v2y
    mag1 = math.hypot(v1x, v1y)
    mag2 = math.hypot(v2x, v2y)
    if mag1 == 0 or mag2 == 0:
        return 180.0
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosang))


# ---------- torso + pose shape ----------

def classify_torso_and_height(pts: List[Point]) -> Tuple[str, float, Optional[Point], Optional[Point]]:
    """
    Classify torso orientation: 'upright', 'leaning', 'horizontal', 'unknown'.
    Returns (torso_state, torso_angle_deg, neck_point, mid_hip_point)
    """
    neck = get_point(pts, 1)
    r_hip = get_point(pts, 8)
    l_hip = get_point(pts, 11)
    mid_hip = avg_points(r_hip, l_hip)
    if neck is None or mid_hip is None:
        return "unknown", 90.0, neck, mid_hip

    ang = angle_between(mid_hip, neck)  # 0 = horizontal, 90 = vertical
    if ang >= 60:
        state = "upright"
    elif 30 <= ang < 60:
        state = "leaning"
    else:
        state = "horizontal"
    return state, ang, neck, mid_hip


def classify_pose_shape(pts: List[Point]) -> str:
    """
    Rough shape bucket:
      POSE_1: standing straight
      POSE_2: standing / contrapposto / dynamic
      POSE_3: seated
      POSE_4: kneeling
      POSE_5: lying (back/stomach)
      POSE_6: lying (side)
      POSE_7: leaning / lounge / unknown upright
    """
    torso_state, torso_ang, neck, mid_hip = classify_torso_and_height(pts)

    # Estimate person "height" from keypoints to normalize thresholds
    ys = [p[1] for p in pts if p[2] >= CONF_THRESH]
    if not ys or neck is None or mid_hip is None:
        return "POSE_7"
    y_top, y_bottom = min(ys), max(ys)
    img_height = max(1.0, y_bottom - y_top)
    torso_len = distance(neck, mid_hip)

    r_knee = get_point(pts, 9)
    l_knee = get_point(pts, 12)
    r_ankle = get_point(pts, 10)
    l_ankle = get_point(pts, 13)

    def avg_y(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
        if a is None and b is None:
            return None
        if a is None:
            return b[1]
        if b is None:
            return a[1]
        return (a[1] + b[1]) / 2.0

    knee_y = avg_y(r_knee, l_knee)
    ankle_y = avg_y(r_ankle, l_ankle)
    hip_y = mid_hip[1]

    # Horizontal / lying cases
    if torso_state == "horizontal":
        # Side vs back: check vertical offset between left/right shoulders/hips
        r_sh = get_point(pts, 2)
        l_sh = get_point(pts, 5)
        r_hip = get_point(pts, 8)
        l_hip = get_point(pts, 11)

        side_score = 0.0
        if r_sh and l_sh:
            side_score += abs(r_sh[1] - l_sh[1])
        if r_hip and l_hip:
            side_score += abs(r_hip[1] - l_hip[1])

        if side_score > img_height * 0.1:
            return "POSE_6"  # side-lying
        return "POSE_5"      # back/stomach

    # Upright / leaning cases
    feet_below_hips = False
    if ankle_y is not None:
        feet_below_hips = (ankle_y - hip_y) > (torso_len * 0.7)

    knees_near_ground = False
    knees_near_hips = False
    if knee_y is not None:
        if ankle_y is not None:
            knees_near_ground = abs(knee_y - ankle_y) < torso_len * 0.3
        knees_near_hips = abs(knee_y - hip_y) < torso_len * 0.4

    # Evaluate knee angles if available
    knee_angles = []
    r_hip = get_point(pts, 8)
    l_hip = get_point(pts, 11)
    if r_hip and r_knee and r_ankle:
        knee_angles.append(knee_angle(r_hip, r_knee, r_ankle))
    if l_hip and l_knee and l_ankle:
        knee_angles.append(knee_angle(l_hip, l_knee, l_ankle))
    min_knee_angle = min(knee_angles) if knee_angles else 180.0

    if torso_state in ("upright", "leaning"):
        # Kneeling
        if knees_near_ground and knees_near_hips:
            return "POSE_4"

        # Seated
        if knees_near_hips and not feet_below_hips:
            return "POSE_3"

        # Standing vs leaning/lounging
        if feet_below_hips:
            # Very straight knees + torso ~vertical -> POSE_1
            if min_knee_angle > 160 and torso_state == "upright":
                return "POSE_1"
            # Otherwise dynamic / contrapposto
            return "POSE_2"

        # Upright-ish but feet not clearly below hips -> lounging
        return "POSE_7"

    # Unknown -> treat as lounging / misc
    return "POSE_7"


# ---------- camera angle classification ----------

def classify_camera_angle(pts: List[Point]) -> str:
    """
    Very rough camera angle buckets based on shoulder/hip foreshortening.

    Idea:
      - If subject is horizontal, call it CAM_TOPDOWN.
      - For upright/leaning:
          ratio = shoulder_span / hip_span
          shoulders >> hips  -> camera high
          shoulders ~ hips   -> eye level
          hips  >> shoulders -> camera low
    """
    torso_state, _, _, _ = classify_torso_and_height(pts)

    # Horizontal -> treat as top-down / overhead-ish.
    if torso_state == "horizontal":
        return "CAM_TOPDOWN"

    r_sh = get_point(pts, 2)
    l_sh = get_point(pts, 5)
    r_hip = get_point(pts, 8)
    l_hip = get_point(pts, 11)

    if not (r_sh and l_sh and r_hip and l_hip):
        return "CAM_UNKNOWN"

    shoulder_span = abs(r_sh[0] - l_sh[0])
    hip_span = abs(r_hip[0] - l_hip[0])

    if hip_span <= 1e-6:
        return "CAM_UNKNOWN"

    ratio = shoulder_span / hip_span

    # Tunable thresholds — this just gives you stable buckets to work with.
    if ratio >= 1.30:
        return "CAM_HIGH"
    if ratio >= 1.10:
        return "CAM_SLIGHT_HIGH"
    if ratio >= 0.90:
        return "CAM_EYE"
    if ratio >= 0.70:
        return "CAM_SLIGHT_LOW"
    return "CAM_LOW"


# ---------- IO + main ----------

def build_buckets(input_dir: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith("_keypoints.json")
    )

    for fname in files:
        fpath = os.path.join(input_dir, fname)
        pts = load_keypoints(fpath)

        if pts is None:
            shape_bucket = "POSE_7"
            camera_bucket = "CAM_UNKNOWN"
        else:
            try:
                shape_bucket = classify_pose_shape(pts)
            except Exception:
                shape_bucket = "POSE_7"
            try:
                camera_bucket = classify_camera_angle(pts)
            except Exception:
                camera_bucket = "CAM_UNKNOWN"

        records.append(
            {
                "filename": fname,
                "shape_bucket": shape_bucket,
                "camera_bucket": camera_bucket,
            }
        )

    return records


def write_outputs(records: List[Dict[str, Any]], output_prefix: str) -> None:
    csv_path = f"{output_prefix}.csv"
    json_path = f"{output_prefix}.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "shape_bucket", "camera_bucket"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[pose_bucket_builder] wrote {csv_path}")
    print(f"[pose_bucket_builder] wrote {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_keypoints.json files",
    )
    ap.add_argument(
        "--output-prefix",
        default="pose_buckets",
        help="Prefix for CSV/JSON outputs (default: pose_buckets)",
    )
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    print(f"[pose_bucket_builder] Scanning: {input_dir}")

    records = build_buckets(input_dir)
    write_outputs(records, args.output_prefix)

    # Small summary by bucket so you can sanity-check the distribution.
    from collections import Counter

    shape_counts = Counter(r["shape_bucket"] for r in records)
    cam_counts = Counter(r["camera_bucket"] for r in records)

    print("[pose_bucket_builder] Pose bucket counts:")
    for k, v in sorted(shape_counts.items()):
        print(f"  {k}: {v}")

    print("[pose_bucket_builder] Camera bucket counts:")
    for k, v in sorted(cam_counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

