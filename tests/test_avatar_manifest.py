import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))


def test_avatar_manifest_first_latest(monkeypatch, tmp_path):
    monkeypatch.setenv("LEXI_USER_DATA_ENABLED", "1")
    monkeypatch.setenv("LEX_USER_DATA_ROOT", str(tmp_path))
    from lexi.utils import avatar_manifest as am  # type: ignore

    importlib.reload(am)

    # create dummy image
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")

    manifest = am.record_avatar_event("tester@example.com", str(img), prompt="hello", traits={"hair": "blue"})
    assert manifest.get("first")
    assert manifest.get("latest")
    assert am.latest_avatar_path("tester@example.com")
    assert am.first_avatar_path("tester@example.com")

    # second event should keep first and update latest
    img2 = tmp_path / "img2.png"
    img2.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest2 = am.record_avatar_event("tester@example.com", str(img2), prompt="second")
    assert manifest2["first"]["basename"] == img.name
    assert manifest2["latest"]["basename"] == img2.name
