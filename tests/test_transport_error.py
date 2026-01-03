import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import importlib.util

ERROR_PATH = ROOT / "backend" / "lexi" / "utils" / "error_responses.py"
spec = importlib.util.spec_from_file_location("error_responses", ERROR_PATH)
err_mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(err_mod)  # type: ignore
SOFT_ERROR_MESSAGE = err_mod.SOFT_ERROR_MESSAGE  # type: ignore
soft_error_payload = err_mod.soft_error_payload  # type: ignore


def test_soft_error_payload_includes_user_friendly_message():
    payload = soft_error_payload(error_detail="boom", trace_id="t1")
    assert payload["cleaned"] == SOFT_ERROR_MESSAGE
    assert payload["raw"] == SOFT_ERROR_MESSAGE
    assert payload["choices"] == [{"text": SOFT_ERROR_MESSAGE}]
    assert payload["error_detail"] == "boom"
    assert payload["trace_id"] == "t1"
