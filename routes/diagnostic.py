import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..persona.persona_core import lex_persona
from ..sd.sd_pipeline import generate_avatar_pipeline
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..routes.lex_persona import assemble_prompt

# Create router for diagnostic endpoints
router = APIRouter(tags=["Diagnostics"])
# Module-level logger
logger = logging.getLogger(__name__)

@router.get("/diagnostic")
async def run_self_diagnostic() -> JSONResponse:
    """
    Run a suite of self-diagnostic tests covering GPU availability, prompt assembly,
    avatar generation, memory CRUD operations, persona state loading, file existence,
    and LLM backend responsiveness. Returns status, uptime, and detailed results.
    """
    results: Dict[str, Any] = {}
    start_time = time.time()

    def safe_test(name: str, func: Callable[[], Any]) -> None:
        """
        Execute a test function safely, capturing its outcome and execution time.
        Populates the `results` dict with pass/fail status, timing, and result or error info.
        """
        try:
            t0 = time.time()
            res = func()
            elapsed = round(time.time() - t0, 3)
            results[name] = {
                "ok": True,
                "time": elapsed,
                "result": res if isinstance(res, (str, bool, int, float)) else "✅",
            }
        except Exception as exc:
            logger.exception("Diagnostic test '%s' failed", name)
            results[name] = {
                "ok": False,
                "error": str(exc),
                "trace": traceback.format_exc().splitlines()[-3:],
            }

    # Perform diagnostics
    safe_test("GPU available", lambda: torch.cuda.is_available())
    safe_test(
        "Avatar prompt build",
        lambda: assemble_prompt(traits={"hair": "red", "eyes": "green"})
    )

    safe_test(
        "Dummy avatar gen",
        lambda: generate_avatar(
            prompt="simple placeholder portrait",
            steps=1
        )
    )
    safe_test("Memory CRUD test", memory_test)
    safe_test(
        "Persona load state",
        lambda: bool(lex_persona._load_traits_state())
    )
    safe_test(
        "Persona avatar path exists",
        lambda: Path(lex_persona.get_avatar_path()).exists()
    )
    safe_test(
        "LLM backend echo",
        lambda: "ping" in lex_persona.chat("__test_diagnostic__").lower()
    )

    # Compile response
    status = "PASS" if all(entry.get("ok") for entry in results.values()) else "FAIL"
    uptime = round(time.time() - start_time, 3)
    payload = {
        "status": status,
        "uptime": uptime,
        "results": results,
    }
    return JSONResponse(payload)


def memory_test() -> bool:
    """
    Test basic Memory CRUD operations: save a shard, retrieve recent entries, delete shard.
    Returns True if the test value is found in recent memory entries.
    """
    test_shard = MemoryShard(role="test", content="ping")
    memory.save(test_shard)
    entries = memory.recent(limit=5)
    memory.delete(test_shard)
    return any("ping" in entry.content for entry in entries)

