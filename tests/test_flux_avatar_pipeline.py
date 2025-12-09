from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from backend.lexi.sd.flux_prompt_builder import (
    FLUX_LEXI_HOT_BASE_PROMPT,
    FLUX_PORTRAIT_NEGATIVE,
    build_flux_avatar_prompt,
)
from backend.lexi.sd import sd_pipeline


def _write_png(path: Path, size: tuple[int, int] = (16, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=(123, 111, 255))
    img.save(path, format="PNG")
    return path


def test_build_flux_avatar_prompt_includes_traits():
    prompt = build_flux_avatar_prompt(
        traits={"hair": "long black", "outfit": "red dress", "vibe": "playful"},
        prompt_text="test prompt",
    )
    assert FLUX_LEXI_HOT_BASE_PROMPT.split(",")[0] in prompt
    assert "long black hair" in prompt
    assert "red dress" in prompt
    assert "playful mood" in prompt
    assert "test prompt" in prompt
    assert FLUX_PORTRAIT_NEGATIVE  # ensure constant is available


def test_generate_avatar_pipeline_img2img(monkeypatch, tmp_path):
    # Set up temp avatar directories
    monkeypatch.setattr(sd_pipeline, "IMAGE_DIR", tmp_path)
    monkeypatch.setattr(sd_pipeline, "PUBLIC_AVATAR_DIR", tmp_path)
    monkeypatch.setattr(sd_pipeline, "AV_PUBLIC_DIR", tmp_path)

    base = tmp_path / "lexi_base.png"
    _write_png(base)

    # Stub out network calls
    class FakeResp:
        def __init__(self, data=None, headers=None, ok=True):
            self._data = data or {}
            self._headers = headers or {"content-type": "application/json"}
            self.ok = ok

        def json(self):
            return self._data

        def raise_for_status(self):
            return None

        @property
        def headers(self):
            return self._headers

    def fake_get(url, *args, **kwargs):
        if "object_info" in url:
            return FakeResp(
                {
                    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": ""}}},
                    "VAELoader": {"input": {"required": {"vae_name": ""}}},
                    "DualCLIPLoader": {"input": {"required": {"type": ""}}},
                }
            )
        return FakeResp({})

    monkeypatch.setattr(sd_pipeline.requests, "get", fake_get)
    monkeypatch.setattr(sd_pipeline, "_post_graph", lambda graph: "pid1")
    monkeypatch.setattr(
        sd_pipeline,
        "_wait_for_images",
        lambda pid, timeout_s=240: [{"filename": "out.png", "subfolder": "", "type": "output"}],
    )

    def fake_download(filename: str, subfolder: str = "", ftype: str = "output", dst: Path | None = None):
        target = dst or (tmp_path / "dl.png")
        return _write_png(target)

    monkeypatch.setattr(sd_pipeline, "_download_image", fake_download)

    res = sd_pipeline.generate_avatar_pipeline(
        prompt="test prompt",
        mode="img2img",
        source_path=str(base),
        fresh_base=False,
    )

    assert res["ok"]
    assert res["meta"]["mode"] == "img2img"
    assert Path(res["file"]).exists()
