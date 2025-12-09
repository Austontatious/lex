import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFETY_PATH = ROOT / "backend" / "lexi" / "utils" / "safety.py"

spec = importlib.util.spec_from_file_location("lexi_safety", SAFETY_PATH)
safety = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(safety)  # type: ignore
ensure_crisis_safety_style = safety.ensure_crisis_safety_style  # type: ignore
scrub_self_harm_methods = safety.scrub_self_harm_methods  # type: ignore


def test_scrub_self_harm_methods_basic():
    text = "He mentioned overdose and a noose."
    cleaned = scrub_self_harm_methods(text)
    assert "overdose" not in cleaned
    assert "noose" not in cleaned
    assert "harmful actions" in cleaned


def test_ensure_crisis_adds_template_when_missing_safety():
    user = "I want to harm myself and don't see a way out."
    reply = "You could overdose or cut to let it out."
    out = ensure_crisis_safety_style(user, reply)
    # Should replace with crisis template, not include explicit methods
    assert "overdose" not in out.lower()
    assert "cut" not in out.lower()
    assert "reach out" in out.lower()
    assert "not alone" in out.lower()


def test_ensure_crisis_preserves_reply_but_scrubs_when_safe_language_present():
    user = "I feel like suicide tonight"
    reply = "I'm sorry you're feeling this way. Please reach out instead of overdose or cutting."
    out = ensure_crisis_safety_style(user, reply)
    assert "overdose" not in out.lower()
    assert "cutting" not in out.lower()
    assert "i'm sorry" in out.lower()
