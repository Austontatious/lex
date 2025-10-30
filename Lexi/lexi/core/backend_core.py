# Lexi/lexi/core/backend_core.py
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

from ..alpha.session_manager import SessionRegistry
from ..config.now import ENABLE_NOW as CONFIG_ENABLE_NOW
from ..utils.now_utils import log_now
from ..boot.avatar_first_render import router as avatar_bootstrap_router

log = logging.getLogger("lexi.backend")

_start_now_scheduler = None
_refresh_now_feed = None
_env_now_enabled = os.getenv("LEXI_ENABLE_NOW", "0").lower() not in ("0", "false", "no", "off")
_now_enabled = CONFIG_ENABLE_NOW and _env_now_enabled

if _now_enabled:
    try:
        from .now_scheduler import start_now_scheduler as _start_now_scheduler
        from ..utils.now_ingest import refresh_now_feed as _refresh_now_feed
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("NOW scheduler disabled: %s", exc)
        _start_now_scheduler = None
        _refresh_now_feed = None
else:
    log.info("NOW feed disabled at startup; skipping scheduler import")

app = FastAPI(title="Lexi Backend", version="0.1.0")
app.include_router(avatar_bootstrap_router)
app.state.alpha_sessions = SessionRegistry()
# ---------------- CORS (dev-friendly) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # relax for local dev; tighten in prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Static files (/static/...) -------------
# Serve default avatars etc. from the repo's /static directory.
# We try a few likely locations; first one that exists wins.
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../Lexi
CANDIDATES = [
    PROJECT_ROOT / "static",                   # Lexi/static
    PROJECT_ROOT / "Lexi" / "lexi" / "static", # Lexi/lexi/static (if you keep it here)
    Path("/mnt/data/Lexi/static"),             # external mount, optional
]
for p in CANDIDATES:
    if p.exists():
        app.mount("/static", StaticFiles(directory=str(p)), name="static")
        break

# ------------- Optional: Serve built frontend (CRA build) -------------
# If a production build exists, serve it and provide a catch-all to index.html
FRONT_BUILD = PROJECT_ROOT / "Lexi" / "lexi" / "frontend" / "build"
if FRONT_BUILD.exists():
    app.mount("/app", StaticFiles(directory=str(FRONT_BUILD), html=True), name="app")

    @app.get("/{full_path:path}")
    async def _spa_catch_all(full_path: str):
        index = FRONT_BUILD / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"app": "lexi-backend", "version": "0.1.0"}

# ------------- Include routers under /lexi -------------
# IMPORTANT:
# - lexi_persona.py declares: router = APIRouter(prefix="/persona", ...)
# - Including it with prefix="/lexi" → final paths /lexi/persona, /lexi/persona/...
from ..routes.lexi import router as lexi_router
from ..routes.gen import router as gen_router
from ..routes.love_loop import router as love_router
from ..routes.now import router as now_router, tools as tools_router
from ..routes.alpha import router as alpha_router

app.include_router(lexi_router)
app.include_router(gen_router)
app.include_router(love_router)
app.include_router(now_router)     # <-- this exposes /now/...
app.include_router(tools_router)   # <-- ...and /tools/web_search
app.include_router(alpha_router)
# Core chat routes (optional)
try:
    from ..routes import lexi as lexi_routes
    app.include_router(lexi_routes.router, prefix="/lexi")
    log.info("Mounted lexi.routes.lexi at /lexi")
except Exception as e:
    log.info("lexi.routes.lexi not mounted: %s", e)

# Generation routes (optional)
try:
    from ..routes import gen
    app.include_router(gen.router, prefix="/lexi")
    log.info("Mounted lexi.routes.gen at /lexi")
except Exception as e:
    log.info("lexi.routes.gen not mounted: %s", e)

# Persona routes — FE calls /lexi/persona
try:
    from ..routes import lexi_persona
    app.include_router(lexi_persona.router, prefix="/lexi")
    log.info("Mounted lexi.routes.lexi_persona at /lexi/persona")
except Exception as e:
    log.warning("lexi.routes.lexi_persona not mounted: %s", e)


# Optional extras
try:
    from ..routes import love
    app.include_router(love.router, prefix="/lexi")
    log.info("Mounted lexi.routes.love at /lexi")
except Exception as e:
    log.info("lexi.routes.love not mounted: %s", e)

# ---------- Now Feed: scheduler + warm cache on startup ----------
@app.on_event("startup")
async def _startup_now_feed():
    # start periodic refresh job
    if not _start_now_scheduler or not _refresh_now_feed:
        log.info("NOW scheduler disabled; skipping warm-up")
        return

    _start_now_scheduler()
    # warm the cache immediately so /now has data on first request
    try:
        await _refresh_now_feed()
        log.info("Now Feed warmed successfully.")
    except Exception as e:
        log.warning("Now Feed warm-up failed: %s", e)

# ---------------- Health / Ready under /lexi ----------------
health = APIRouter()

@health.get("/health")
async def healthz():
    return {"ok": True}

@health.get("/ready")
async def readyz():
    return {"ready": True}

app.include_router(health, prefix="/lexi")

# ---------------- Root (optional) ----------------
@app.get("/")
async def root():
    return {"app": "lexi-backend", "version": "0.1.0"}
