from pathlib import Path

from backend.lexi.sd.pose_selector import choose_pose


def _write_pose_csv(csv_path: Path, rows: list[tuple[str, str, str]]) -> None:
    csv_path.write_text("filename,shape_bucket,camera_bucket\n", encoding="utf-8")
    with csv_path.open("a", encoding="utf-8") as f:
        for fname, shape, cam in rows:
            f.write(f"{fname},{shape},{cam}\n")


def test_choose_pose_by_feel(tmp_path: Path):
    render_dir = tmp_path / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        ("pose_a_keypoints.json", "POSE_2", "CAM_EYE"),
        ("pose_b_keypoints.json", "POSE_4", "CAM_LOW"),
    ]
    _write_pose_csv(tmp_path / "buckets.csv", rows)
    for stem, _, _ in rows:
        name = Path(stem).stem.replace("_keypoints", "_rendered")
        render = render_dir / f"{name}.jpg"
        render.write_bytes(b"\x89PNG\r\n\x1a\n")  # tiny placeholder

    choice = choose_pose(
        csv_path=tmp_path / "buckets.csv",
        render_dir=render_dir,
        feel="playful",
        rng_seed=123,
    )

    assert choice is not None
    assert choice.shape_bucket == "POSE_2"
    assert choice.render_path.exists()
