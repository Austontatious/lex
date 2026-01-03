import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..persona.persona_core import lexi_persona
from ..sd import sd_pipeline
from ..sd.sd_pipeline import generate_avatar_pipeline

router = APIRouter(tags=["Diagnostics"])
logger = logging.getLogger(__name__)


class TestOutcome(BaseModel):
    ok: bool
    time: Optional[float] = None
    result: Optional[str | bool | int | float] = None
    error: Optional[str] = None
    trace: Optional[List[str]] = None


class DiagnosticPayload(BaseModel):
    status: str
    uptime: float
    results: Dict[str, TestOutcome]


def _assemble_prompt(traits: Dict[str, str]) -> str:
    """Build a minimal SD prompt from the provided trait dictionary."""
    values = [v.strip() for v in (traits or {}).values() if isinstance(v, str) and v.strip()]
    if not values:
        values = ["portrait lighting", "gentle smile"]
    return ", ".join(values)


@router.get("/diagnostic", response_model=DiagnosticPayload)
async def run_self_diagnostic() -> DiagnosticPayload:
    """
    Run a suite of self-diagnostic tests covering GPU availability, prompt assembly,
    avatar generation, memory CRUD operations, persona state loading, file existence,
    and LLM backend responsiveness.
    """
    results: Dict[str, TestOutcome] = {}
    start_time = time.time()

    def safe_test(name: str, func: Callable[[], Any]) -> None:
        """Capture timing + result for each diagnostic helper."""
        try:
            t0 = time.time()
            res = func()
            elapsed = round(time.time() - t0, 3)
            ok = True
            result_field: Optional[str | bool | int | float] = None
            # If a dict with {ok: bool, error?: str}, treat ok False as failure
            if isinstance(res, dict) and "ok" in res:
                ok = bool(res.get("ok"))
                if not ok and res.get("error"):
                    raise RuntimeError(str(res.get("error")))
                result_field = "✅" if ok else "❌"
            else:
                result_field = res if isinstance(res, (str, bool, int, float)) else "✅"

            outcome = TestOutcome(ok=ok, time=elapsed, result=result_field)
        except Exception as exc:
            logger.exception("Diagnostic test '%s' failed", name)
            outcome = TestOutcome(
                ok=False,
                error=str(exc),
                trace=traceback.format_exc().splitlines()[-3:],
            )
        results[name] = outcome

    # External service URLs
    COMFY_URL = os.getenv("COMFY_URL", "http://host.docker.internal:8188").rstrip("/")
    VLLM_URL = os.getenv(
        "OPENAI_API_BASE", os.getenv("LLM_API_BASE", "http://host.docker.internal:8008/v1")
    ).rstrip("/")

    safe_test("GPU available", lambda: torch.cuda.is_available())
    safe_test("Avatar prompt build", lambda: _assemble_prompt({"hair": "red", "eyes": "green"}))
    safe_test(
        "Dummy avatar gen",
        lambda: generate_avatar_pipeline(
            prompt="simple placeholder portrait", steps=1, cfg_scale=1.5
        ),
    )

    # External reachability checks (don’t hard fail if down; just report)
    def _check_comfy() -> bool:
        r = requests.get(f"{COMFY_URL}/object_info", timeout=5)
        r.raise_for_status()
        return bool(r.json())

    def _check_vllm() -> bool:
        r = requests.get(f"{VLLM_URL}/models", timeout=5)
        r.raise_for_status()
        return r.headers.get("content-type", "").startswith("application/json")

    safe_test("Comfy reachable", _check_comfy)
    safe_test("vLLM reachable", _check_vllm)
    safe_test("Images dir writable", lambda: _touch_test_file(sd_pipeline.IMAGE_DIR))
    safe_test("Memory CRUD test", memory_test)
    safe_test("Persona load state", lambda: bool(lexi_persona._load_traits_state()))
    safe_test("Persona avatar path exists", lambda: Path(lexi_persona.get_avatar_path()).exists())
    safe_test(
        "LLM backend echo", lambda: "ping" in lexi_persona.chat("__test_diagnostic__").lower()
    )

    status = "PASS" if all(outcome.ok for outcome in results.values()) else "FAIL"
    uptime = round(time.time() - start_time, 3)
    return DiagnosticPayload(status=status, uptime=uptime, results=results)


def memory_test() -> bool:
    """
    Test basic Memory CRUD operations: save a shard, retrieve recent entries, delete shard.
    Returns True if the test value is found in recent memory entries.
    """
    test_shard = MemoryShard(role="test", content="ping")
    memory.save(test_shard)
    entries = memory.recent(limit=5)
    memory.delete(test_shard.created_at)
    return any("ping" in entry.content for entry in entries)


def _touch_test_file(dirpath: Path) -> bool:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / ".write_test"
    p.write_text("ok", encoding="utf-8")
    p.unlink(missing_ok=True)
    return True
