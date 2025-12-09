from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoseRecord:
    filename: str
    shape_bucket: str
    camera_bucket: str
    keypoints_path: Path
    render_path: Path


@dataclass(frozen=True)
class PoseChoice:
    pose_id: str
    shape_bucket: str
    camera_bucket: str
    keypoints_path: Path
    render_path: Path


_POSE_CACHE: Dict[Tuple[Path, Path], List[PoseRecord]] = {}

# Default bucket → feel mapping (ordered by preference)
FEEL_POSE_BUCKETS: Dict[str, Tuple[Sequence[str], Sequence[str]]] = {
    "playful": (("POSE_2", "POSE_7"), ("CAM_HIGH", "CAM_SLIGHT_LOW", "CAM_EYE")),
    "seductive": (("POSE_4", "POSE_6"), ("CAM_LOW", "CAM_SLIGHT_LOW", "CAM_TOPDOWN")),
    "cozy": (("POSE_3", "POSE_6"), ("CAM_EYE", "CAM_SLIGHT_HIGH", "CAM_TOPDOWN")),
    "confident": (("POSE_2", "POSE_7"), ("CAM_LOW", "CAM_SLIGHT_LOW")),
}


def _load_records(csv_path: Path, render_dir: Path) -> List[PoseRecord]:
    cache_key = (csv_path.resolve(), render_dir.resolve())
    cached = _POSE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    records: List[PoseRecord] = []
    if not csv_path.exists():
        _POSE_CACHE[cache_key] = records
        return records

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("filename") or "").strip()
            if not fname:
                continue
            stem = fname.rsplit("_keypoints", 1)[0]
            render_name = f"{stem}_rendered.jpg"
            keypoints_path = csv_path.parent / fname
            render_path = render_dir / render_name
            records.append(
                PoseRecord(
                    filename=fname,
                    shape_bucket=(row.get("shape_bucket") or "").strip() or "POSE_7",
                    camera_bucket=(row.get("camera_bucket") or "").strip() or "CAM_UNKNOWN",
                    keypoints_path=keypoints_path,
                    render_path=render_path,
                )
            )

    _POSE_CACHE[cache_key] = records
    return records


def _filtered(
    records: Iterable[PoseRecord],
    shape_bucket: Optional[str],
    camera_bucket: Optional[str],
) -> List[PoseRecord]:
    if not shape_bucket and not camera_bucket:
        return list(records)
    out: List[PoseRecord] = []
    for r in records:
        if shape_bucket and r.shape_bucket != shape_bucket:
            continue
        if camera_bucket and r.camera_bucket != camera_bucket:
            continue
        out.append(r)
    return out


def choose_pose(
    *,
    csv_path: Path,
    render_dir: Path,
    feel: Optional[str] = None,
    shape_bucket: Optional[str] = None,
    camera_bucket: Optional[str] = None,
    pose_id: Optional[str] = None,
    rng_seed: Optional[int] = None,
) -> Optional[PoseChoice]:
    """
    Pick a pose render + metadata using the bucket CSV.

    Selection priority:
      1) exact pose_id match (filename stem)
      2) shape/camera buckets if provided
      3) feel mapping to buckets (playful, seductive, cozy, confident)
      4) fall back to any record
    """
    records = _load_records(csv_path, render_dir)
    if not records:
        return None

    rng = random.Random(rng_seed)

    if pose_id:
        target = pose_id.strip()
        for r in records:
            stem = r.filename.rsplit("_keypoints", 1)[0]
            if stem == target or r.filename == target:
                return PoseChoice(
                    pose_id=stem,
                    shape_bucket=r.shape_bucket,
                    camera_bucket=r.camera_bucket,
                    keypoints_path=r.keypoints_path,
                    render_path=r.render_path,
                )

    # Map feel → buckets if caller did not supply explicit buckets
    feel_norm = (feel or "").strip().lower()
    if feel_norm and not (shape_bucket or camera_bucket):
        buckets = FEEL_POSE_BUCKETS.get(feel_norm)
        if buckets:
            shapes, cams = buckets
            # Try every combination in order until we find hits
            for sb in shapes:
                for cb in cams:
                    hits = _filtered(records, sb, cb)
                    if hits:
                        r = rng.choice(hits)
                        stem = r.filename.rsplit("_keypoints", 1)[0]
                        return PoseChoice(
                            pose_id=stem,
                            shape_bucket=r.shape_bucket,
                            camera_bucket=r.camera_bucket,
                            keypoints_path=r.keypoints_path,
                            render_path=r.render_path,
                        )

    candidates = _filtered(records, shape_bucket, camera_bucket)
    if not candidates:
        candidates = records

    choice = rng.choice(candidates)
    stem = choice.filename.rsplit("_keypoints", 1)[0]
    return PoseChoice(
        pose_id=stem,
        shape_bucket=choice.shape_bucket,
        camera_bucket=choice.camera_bucket,
        keypoints_path=choice.keypoints_path,
        render_path=choice.render_path,
    )


__all__ = ["PoseChoice", "choose_pose", "FEEL_POSE_BUCKETS"]
