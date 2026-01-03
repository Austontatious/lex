from __future__ import annotations

import json
import re
import sys
from pathlib import Path
import types
import os

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure memory path points to a writable temp fixture before imports trigger persona init
tmp_mem = ROOT / "tmp" / "test_memory.jsonl"
tmp_mem.parent.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LEX_MEMORY_PATH", str(tmp_mem))

# Provide a stub for missing user_identity module before importing persona_core
fake_user_identity = types.SimpleNamespace(
    normalize_user_id=lambda uid: uid,
    sanitize_user_id=lambda uid: uid,
    user_bucket=lambda base, uid: None,
    user_id_feature_enabled=lambda: False,
    resolve_user_id=lambda request: None,
)
sys.modules.setdefault("backend.lexi.utils.user_identity", fake_user_identity)
sys.modules.setdefault("lexi.utils.user_identity", fake_user_identity)

from backend.lexi.persona import persona_core
from backend.lexi.persona.persona_core import LexiPersona, _hallucination_violation, needs_fresh_data


def test_smalltalk_today_does_not_trigger_tool():
    assert needs_fresh_data("Whatcha thinking about today?") is False
    assert needs_fresh_data("How's your morning today?") is False


def test_movies_flow_uses_tool_and_stays_grounded(monkeypatch):
    user_text = "What's new in theaters this week?"
    assert needs_fresh_data(user_text) is True

    tool_payload = {
        "results": [
            {
                "title": "Test Movie",
                "release_date": "2024-10-02",
                "runtime": None,
                "theatrical": True,
                "overview": "A grounded movie result for testing.",
                "showtimes_url": "https://example.com",
                "source": "stub",
            }
        ],
        "start_date": "2024-10-01",
        "end_date": "2024-10-07",
        "region": None,
    }

    def fake_movies_now(location, start_date, end_date, limit=12):
        # ensure inputs arrive from tool call
        assert start_date
        assert end_date
        return tool_payload

    class StubLoader:
        def __init__(self):
            self.calls = []

        def generate(self, payload, **kwargs):
            self.calls.append({"payload": payload, "kwargs": kwargs})
            if kwargs.get("tools"):
                return {
                    "text": "",
                    "finish_reason": "tool_calls",
                    "usage": {},
                    "raw": {
                        "choices": [
                            {
                                "message": {
                                    "tool_calls": [
                                        {
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "movies_now",
                                                "arguments": json.dumps(
                                                    {
                                                        "start_date": "2024-10-01",
                                                        "end_date": "2024-10-07",
                                                        "limit": 3,
                                                    }
                                                ),
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                }
            return {"text": 'Here is one: "Test Movie".', "finish_reason": "stop", "usage": {}}

    persona = LexiPersona()
    persona.loader = StubLoader()
    persona.model = persona.loader
    persona.tokenizer = persona.loader

    monkeypatch.setattr(persona_core, "movies_now", fake_movies_now)

    prompt_pkg = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_text},
        ]
    }
    reply, tool_called, finish_reason, titles, payload_out = persona._run_movies_tool_flow(
        prompt_pkg, user_text, {}
    )

    assert tool_called is True
    assert finish_reason == "stop"
    assert titles == ["Test Movie"]
    assert payload_out == tool_payload

    assert _hallucination_violation(reply, titles) is False
    quoted = re.findall(r'"(.+?)"', reply)
    assert set(quoted).issubset(set(titles))
