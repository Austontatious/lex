import traceback, time, torch
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pathlib import Path

from ..persona_core import lex_persona
from lex.lex_stablediffusion import generate_avatar_pipeline
from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from .lex_persona import _assemble_prompt

router = APIRouter(tags=["Diagnostics"])

@router.get("/diagnostic")
async def run_self_diagnostic():
    results = {}
    start_time = time.time()

    def safe_test(name, func):
        try:
            t0 = time.time()
            result = func()
            results[name] = {
                "ok": True,
                "time": round(time.time() - t0, 3),
                "result": result if isinstance(result, (str, bool, int, float)) else "✅"
            }
        except Exception as e:
            results[name] = {
                "ok": False,
                "error": str(e),
                "trace": traceback.format_exc().splitlines()[-3:]  # Only show final 3 lines
            }

    # Begin tests
    safe_test("GPU available", lambda: torch.cuda.is_available())

    safe_test("Avatar prompt build", lambda: _assemble_prompt(traits={"hair": "red", "eyes": "green"}))

    safe_test("Dummy avatar gen", lambda: generate_avatar_pipeline(
        traits={"hair": "red", "eyes": "green"},
        use_seed=True,
        seed=42,
        test_mode=True  # <-- You’ll need to respect this flag in your pipeline
    ))

    safe_test("Memory CRUD test", memory_test)
    safe_test("Persona load state", lambda: lex_persona._load_traits_state() or True)
    safe_test("Persona avatar path exists", lambda: Path(lex_persona.get_avatar_path()).exists())
    safe_test("LLM backend echo", lambda: "ping" in lex_persona.chat("__test_diagnostic__").lower())

    return JSONResponse({
        "status": "PASS" if all(x["ok"] for x in results.values()) else "FAIL",
        "uptime": round(time.time() - start_time, 3),
        "results": results
    })

def memory_test():
    test_shard = MemoryShard(role="test", content="ping")  # Remove tags
    memory.save(test_shard)
    entries = memory.recent(limit=5)
    memory.delete(test_shard)
    return any("ping" in e.content for e in entries)

