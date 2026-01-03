from __future__ import annotations

# --------------------------------------------------------------------- #
#  Standard Library Imports                                             #
# --------------------------------------------------------------------- #
import asyncio
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict
import uuid

# --------------------------------------------------------------------- #
#  Third‑Party Imports                                                  #
# --------------------------------------------------------------------- #
from fastapi import Body, FastAPI, Request, APIRouter
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# --------------------------------------------------------------------- #
#  Path / Root Setup                                                    #
# --------------------------------------------------------------------- #
LEX_ROOT = Path(__file__).resolve().parent.parent
if str(LEX_ROOT) not in sys.path:
    sys.path.insert(0, str(LEX_ROOT))

# --------------------------------------------------------------------- #
#  Application + Logging                                                #
# --------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("lex.backend")

app = FastAPI(
    title="Lex Backend",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# --------------------------------------------------------------------- #
#  CORS / Static                                                        #
# --------------------------------------------------------------------- #
ORIGINS = os.getenv("LEX_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = LEX_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
else:
    logger.warning("Static directory not found at %s", static_dir)

# --------------------------------------------------------------------- #
#  Internal Imports (after app init to avoid circular surprises)        #
# --------------------------------------------------------------------- #
from ..persona.persona_core import lex_persona
from ..routes.lex import router as lex_router
from ..routes.lex_persona import router as persona_router
from ..routes.love_loop import router as love_router
from ..utils.live_token_viz import send_to_viz
from ..routes import gen
from ..routes.diagnostic import router as diagnostic_router
from .model_loader_core import initialize_model_loader
app.include_router(diagnostic_router)
app.include_router(persona_router, prefix="/persona")
# --------------------------------------------------------------------- #
#  Router Mounting (each once)                                          #
# --------------------------------------------------------------------- #
app.include_router(lex_router,     prefix="/lex")
app.include_router(persona_router, prefix="/lex/persona")
app.include_router(love_router,    prefix="/lex/love")
app.include_router(gen.router,     prefix="/lex")
# --------------------------------------------------------------------- #
#  Stub Router for Avatar Generation                                   #


# --------------------------------------------------------------------- #
#  Static Files (avatars)                                               #
# --------------------------------------------------------------------- #
AVATAR_STATIC_DIR = LEX_ROOT / "static" / "lex" / "avatars"
AVATAR_STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/lex/avatars",
    StaticFiles(directory=str(AVATAR_STATIC_DIR)),
    name="lex_avatars",
)

@app.get("/debug/avatar-path")
async def debug_avatar_path():
    return {
        "AVATAR_STATIC_DIR": str(AVATAR_STATIC_DIR),
        "file_exists": os.path.exists(AVATAR_STATIC_DIR / "default.png"),
        "cwd": os.getcwd(),
        "listing": os.listdir(AVATAR_STATIC_DIR),
    }

# --------------------------------------------------------------------- #
#  Startup / Shutdown Events                                            #
# --------------------------------------------------------------------- #
@app.on_event("startup")
async def _startup():
    logger.info("Application startup initiating.")
    # Visualizer ping (non-fatal)
    try:
        send_to_viz([0.33, -0.33], z=0.0, intensity=0.8)
        logger.info("Visualizer ping OK")
    except Exception as e:
        logger.warning("Visualizer ping failed: %s", e)

    # Warm-up model (optional / skip if env var set)
    if os.getenv("LEX_SKIP_WARMUP") != "1":
        try:
            logger.info("Preloading LLM (warm-up prompt).")
            ml = initialize_model_loader()
            ml.update_cfg(max_tokens=10, stop=["<|endoftext|>", "\n", "<|assistant|>"])
            await asyncio.to_thread(ml.generate, "<|system|> SYSTEM WARMUP: test token.")
            logger.info("Warm-up complete.")
        except Exception:
            logger.error("Warm-up failed: %s", traceback.format_exc().splitlines()[-1])

    logger.info("Application startup complete.")

@app.on_event("shutdown")
async def _shutdown():
    logger.info("Shutting down Lex backend.")

# --------------------------------------------------------------------- #
#  Health / Meta Endpoints                                              #
# --------------------------------------------------------------------- #
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"name": "Lex Backend", "version": "0.1.0"}

# --------------------------------------------------------------------- #
#  Simple Echo / Test (optional)                                        #
# --------------------------------------------------------------------- #
@app.post("/test")
async def test_endpoint(payload: Dict[str, Any] = Body(default_factory=dict)):
    return {"received": payload, "ok": True}

# --------------------------------------------------------------------- #
#  Global Exception Handler (Optional)                                  #
# --------------------------------------------------------------------- #
@app.middleware("http")
async def error_wrapper(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse({"error": "internal_error", "detail": str(exc)}, status_code=500)

# --------------------------------------------------------------------- #
#  Uvicorn Entrypoint (if run as script)                                #
# --------------------------------------------------------------------- #
def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    host = os.getenv("LEX_HOST", host)
    port = int(os.getenv("LEX_PORT", port))
    uvicorn.run(
        "lex.backend_core:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

if __name__ == "__main__":
    run()

