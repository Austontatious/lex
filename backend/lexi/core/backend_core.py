# Lexi/lexi/core/backend_core.py
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import requests

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from ..alpha.session_manager import SessionRegistry
from ..boot.avatar_first_render import router as avatar_bootstrap_router
from ..config.config import REPO_ROOT, STATIC_ROOT, STATIC_URL_PREFIX
from ..config.now import ENABLE_NOW as CONFIG_ENABLE_NOW
from ..config.runtime_env import COMFY_URL
from ..session import session_middleware
from ..utils.now_utils import log_now
from ..sd.sd_pipeline import generate_avatar_pipeline
from ..sd.flux_prompt_builder import BASE_AVATAR_AESTHETIC

log = logging.getLogger("lexi.backend")


def _unique_id(route: APIRoute) -> str:
    tag = (route.tags[0] if route.tags else "default").lower().replace(" ", "_")
    verb = next(iter(route.methods)).lower()
    path = (
        route.path_format.strip("/")
        .replace("/", "_")
        .replace("{", "")
        .replace("}", "")
    )
    return f"{tag}__{verb}__{path or 'root'}"

_start_now_scheduler = None
_refresh_now_feed = None
_env_now_enabled = os.getenv("LEXI_ENABLE_NOW", "1").lower() not in ("0", "false", "no", "off")
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

app = FastAPI(
    title="Lexi Backend",
    version="0.1.0",
    generate_unique_id_function=_unique_id,
)
app.middleware("http")(session_middleware)
app.router.route_class = APIRoute
app.include_router(avatar_bootstrap_router)
app.state.alpha_sessions = SessionRegistry()
# ---------------- CORS (dev-friendly) ----------------
try:
    from ..config.config import settings  # type: ignore
    _cfg_settings = settings  # type: ignore
    _cors_origins = getattr(_cfg_settings, "CORS_ORIGINS", None)
except Exception:
    _cfg_settings = None  # type: ignore
    _cors_origins = None

if _cors_origins is None:
    try:
        from ..config.config import CORS_ORIGINS as _cors_origins  # type: ignore
    except Exception:
        _cors_origins = os.getenv("CORS_ORIGINS", "*")

DEFAULT_CORS_ORIGINS = [
    "https://lexicompanion.com",
    "https://www.lexicompanion.com",
    "https://api.lexicompanion.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if isinstance(_cors_origins, (set, tuple)):
    _cors_origins = list(_cors_origins)
elif isinstance(_cors_origins, str):
    if "," in _cors_origins:
        _cors_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]
    elif _cors_origins:
        _cors_origins = [_cors_origins.strip()]
    else:
        _cors_origins = []
elif not isinstance(_cors_origins, list):
    _cors_origins = []

# Wildcard origin cannot be used when allow_credentials=True; drop and fall back to defaults.
_cors_origins = [origin for origin in _cors_origins if origin and origin != "*"]

if not _cors_origins:
    _cors_origins = DEFAULT_CORS_ORIGINS.copy()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ------------- Static files (/static/...) -------------
# Serve default avatars and other assets from the canonical frontend public dir.
_static_root_candidate = None
if _cfg_settings is not None:
    try:
        candidate = getattr(_cfg_settings, "STATIC_ROOT", None)
        if candidate:
            _static_root_candidate = Path(candidate)
    except Exception:
        _static_root_candidate = None

STATIC_CANDIDATES = [
    _static_root_candidate,
    STATIC_ROOT,
    REPO_ROOT / "static",  # legacy fallback
    REPO_ROOT / "Lexi" / "lexi" / "static",  # legacy snapshot fallback
    Path("/mnt/data/Lexi/static"),  # external mount, optional
]
STATIC_CANDIDATES = [p for p in STATIC_CANDIDATES if p is not None]
static_dir = next((p for p in STATIC_CANDIDATES if p.exists()), STATIC_ROOT)
static_dir_str = str(static_dir)

app.mount("/static", StaticFiles(directory=static_dir_str), name="static_legacy")
if STATIC_URL_PREFIX != "/static":
    app.mount(STATIC_URL_PREFIX, StaticFiles(directory=static_dir_str), name="static_prefixed")

# ------------- Optional: Serve built frontend (Vite/CRA build) -------------
# If a production build exists, serve it and provide a catch-all to index.html
FRONT_BUILD_CANDIDATES = [
    REPO_ROOT / "frontend" / "dist",
    REPO_ROOT / "frontend" / "build",
]
FRONT_BUILD = next((p for p in FRONT_BUILD_CANDIDATES if p.exists()), None)
if FRONT_BUILD:
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
from ..routes.health import router as health_router
from ..routes.feedback import router as feedback_router
from ..routes.love_loop import router as love_router
from ..routes.now import router as now_router, tools as tools_router
from ..routes.alpha import router as alpha_router
from ..routes.onboarding import router as onboarding_router
from ..routes.account import router as account_router
from ..routes.identity import router as identity_router
from ..routes.user_data import router as user_data_router
from ..routes.vector import router as vector_router
from ..routes.avatar_edit import router as avatar_edit_router
from ..routes.analytics import router as analytics_router

app.include_router(lexi_router)
app.include_router(health_router)
app.include_router(feedback_router)
app.include_router(gen_router)
app.include_router(love_router)
app.include_router(now_router)  # <-- this exposes /now/...
app.include_router(tools_router)  # <-- ...and /tools/web_search
app.include_router(alpha_router)
app.include_router(onboarding_router, prefix="/lexi")
app.include_router(account_router)
app.include_router(identity_router)
app.include_router(user_data_router)
app.include_router(vector_router)
app.include_router(avatar_edit_router, prefix="/lexi")
app.include_router(analytics_router)
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


@app.on_event("startup")
async def _warmup_flux_backend():
    if os.getenv("LEXI_SKIP_FLUX_WARMUP", "0").lower() in ("1", "true", "yes", "on"):
        log.info("Flux warmup skipped via LEXI_SKIP_FLUX_WARMUP")
        return
    try:
        requests.get(f"{COMFY_URL}/system_stats", timeout=5)
    except Exception as exc:
        log.warning("Flux warmup system stats request failed: %s", exc)
    try:
        await asyncio.to_thread(
            generate_avatar_pipeline,
            prompt=BASE_AVATAR_AESTHETIC,
            traits={},
            steps=2,
            cfg_scale=1.5,
            seed=42,
            base_name="_warmup_flux",
            fresh_base=True,
        )
        log.info("Flux pipeline warmup completed")
    except Exception as exc:
        log.warning("Flux warmup skipped: %s", exc)


# ---------------- Health / Ready under /lexi ----------------
health = APIRouter()


@health.get("/health")
async def healthz():
    return {"ok": True}


@health.get("/ready")
async def readyz():
    return {"ready": True}


app.include_router(health, prefix="/lexi")



@app.get("/health", tags=["ops"])
def health_root():
    return {"ok": True}

# ---------------- Root (optional) ----------------
@app.get("/")
async def root():
    return {"app": "lexi-backend", "version": "0.1.0"}
