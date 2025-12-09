import json
import sys
from pathlib import Path

from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "backend"))

from lexi.session_logging import sanitize_for_log, log_turn  # type: ignore  # noqa: E402


def _dummy_request(tmp_path: Path) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/lexi/process",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }
    req = Request(scope)
    req.state.session_id = "sess_test"
    req.state.session_log_path = tmp_path
    return req


def test_sanitize_for_log_redacts_email():
    data = sanitize_for_log("Contact me at user@example.com for details")
    assert "content_hash" in data
    assert data["content_preview"]
    assert "<EMAIL>" in data["content_preview"]
    assert "user@example.com" not in data["content_preview"]


def test_log_turn_writes_redacted(tmp_path: Path):
    log_file = tmp_path / "log.ndjson"
    req = _dummy_request(log_file)
    log_turn(
        req,
        "user",
        "Here is my secret: user@example.com",
        turn_id=1,
        mode="default",
        persona="Lexi",
        tool_calls=[],
        safety={"decision": "allow"},
        latency_ms=5,
        model_meta={"model_name": "stub"},
    )
    lines = log_file.read_text().splitlines()
    assert lines, "log_turn should write a line"
    entry = json.loads(lines[-1])
    assert "<EMAIL>" in entry.get("content_preview", "")
    assert "content_hash" in entry
    assert entry["model"]["model_name"] == "stub"
