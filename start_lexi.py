#!/usr/bin/env python3
"""
Lexi unified launcher (no local models)

- Starts FastAPI backend
- Starts Vite/React frontend
- Does NOT boot SD/LLM; those are external services:
    * ComfyUI: http://host.docker.internal:8188
    * vLLM (OpenAI API): http://host.docker.internal:8008/v1

Env:
  COMFY_URL           (default: http://host.docker.internal:8188)
  OPENAI_API_BASE     (default: http://host.docker.internal:8008/v1)
  OPENAI_API_KEY      (default: "dummy")
  REQUIRE_COMFY       (default: "0") -> "1" to wait for Comfy to respond
  REQUIRE_VLLM        (default: "0") -> "1" to wait for /v1/models
  BACKEND_PORT_START  (default: 8000)
  FRONTEND_PORT_START (default: 3000)
"""
from __future__ import annotations

import os, sys, time, socket, subprocess, threading
from pathlib import Path
from typing import Optional, List
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))

PYTHON_BIN = sys.executable
NESTED_PKG = PROJECT_ROOT / "Lexi"
if str(NESTED_PKG) not in sys.path:
    sys.path.insert(0, str(NESTED_PKG))

# External services (updated defaults)
COMFY_URL = os.getenv("COMFY_URL", "http://host.docker.internal:8188").rstrip("/")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://host.docker.internal:8008/v1").rstrip("/")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "dummy")
# Prefer caller-provided LLM_MODEL; otherwise default to the served alias
LLM_MODEL = os.getenv("LLM_MODEL", "Lexi")

REQUIRE_COMFY = os.getenv("REQUIRE_COMFY", "0") == "1"
REQUIRE_VLLM  = os.getenv("REQUIRE_VLLM", "0")  == "1"

BACKEND_PORT_START  = int(os.getenv("BACKEND_PORT_START", "8000"))
FRONTEND_PORT_START = int(os.getenv("FRONTEND_PORT_START", "3000"))

TIMEOUT_SERVICE = 180
TIMEOUT_BACKEND = 120

LOG_FILE = (PROJECT_ROOT / "startup.log").open("w", buffering=1)

def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_FILE.write(msg + "\n")

def stream_output(pipe, prefix: str):
    for line in iter(pipe.readline, ""):
        if line.strip():
            out = f"[{prefix}] {line.rstrip()}"
            print(out)
            LOG_FILE.write(out + "\n")
    pipe.close()

def find_free_port(start: int, tries: int = 30) -> Optional[int]:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return None

def wait_for_http_ok(urls: List[str], timeout: int, expect_json: bool = False) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        for u in urls:
            try:
                r = requests.get(u, timeout=3)
                if r.status_code == 200:
                    if expect_json and "application/json" not in r.headers.get("content-type", ""):
                        continue
                    return True
            except requests.RequestException:
                pass
        time.sleep(1)
    return False



def launch_backend(port: int) -> subprocess.Popen:
    log(f"⚙️  Launching backend on port {port}")
    env = os.environ.copy()

    # ── External services ──
    env.setdefault("COMFY_URL", COMFY_URL)
    env.setdefault("OPENAI_API_BASE", OPENAI_API_BASE)
    env.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    env.setdefault("LLM_API_BASE", OPENAI_API_BASE)
    env.setdefault("LLM_API_KEY", OPENAI_API_KEY)
    if LLM_MODEL:
        env.setdefault("LLM_MODEL", LLM_MODEL)
        env.setdefault("LEX_MODEL_ID", LLM_MODEL)
    if "LEX_DATA_DIR" not in env:
        env["LEX_DATA_DIR"] = str(PROJECT_ROOT / "Lexi")
    env.setdefault("LEX_MEMORY_PATH", str(PROJECT_ROOT / "Lexi" / "memory" / "lexi_memory.jsonl"))

    # ── SDXL / Comfy defaults (filenames ONLY; Comfy reads from models/checkpoints) ──
    # These are safe defaults and can be overridden by real env at runtime.
    env.setdefault("COMFY_BASE_CKPT", "sd_xl_base_1.0.safetensors")
    env.setdefault("COMFY_REFINER_CKPT", "sd_xl_refiner_1.0.safetensors")
    env.setdefault("COMFY_UPSCALE", "true")  # or "false" if you prefer
    # Where to save images for the FE to serve:
    env.setdefault("LEX_IMAGE_DIR", str(PROJECT_ROOT / "Lexi" / "lexi" / "static" / "lexi" / "avatars"))
    trans_cache = env.get("TRANSFORMERS_CACHE")
    if "HF_HOME" not in env:
        env["HF_HOME"] = trans_cache or str(PROJECT_ROOT / "hf_cache")
    elif trans_cache and env["HF_HOME"] != trans_cache:
        log("⚠️  HF_HOME set; TRANSFORMERS_CACHE will be ignored in favor of HF_HOME.")
    try:
        Path(env["LEX_IMAGE_DIR"]).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"⚠️  Could not ensure LEX_IMAGE_DIR exists ({exc})")
    try:
        Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        log(f"⚠️  Could not ensure HF_HOME exists ({exc})")

    # Ensure Python can import the nested package
    env["PYTHONPATH"] = str(PROJECT_ROOT.resolve())

    proc = subprocess.Popen(
        [
            PYTHON_BIN, "-m", "uvicorn", "Lexi.lexi.core.backend_core:app",
            "--host", "0.0.0.0", "--port", str(port),
            "--timeout-keep-alive", "0",
            "--reload", "--reload-dir", str(PROJECT_ROOT / "Lexi"),
            "--workers", "1",
        ],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    threading.Thread(target=stream_output, args=(proc.stdout, "Backend"), daemon=True).start()
    return proc


def wait_for_backend(port: int, timeout: int = TIMEOUT_BACKEND) -> bool:
    url = f"http://127.0.0.1:{port}/"
    for _ in range(timeout):
        try:
            r = requests.get(url, timeout=2)
            if r.status_code in (200, 404):
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False

FRONTEND_DIR = NESTED_PKG / "lexi" / "frontend"

def write_frontend_env(back_port: int, front_port: int) -> None:
    env_vars = {
        "VITE_BACKEND_PORT": str(back_port),
        "VITE_BACKEND_URL": f"http://localhost:{back_port}/lexi",
        "VITE_FRONTEND_PORT": str(front_port),
        "VITE_API_URL": f"http://localhost:{back_port}/lexi",
        "VITE_COMFY_URL": COMFY_URL,
        "VITE_VLLM_URL": OPENAI_API_BASE,
        "REACT_APP_API_URL": f"http://localhost:{back_port}/lexi",
        "REACT_APP_BACKEND_URL": f"http://localhost:{back_port}/lexi",
        "VITE_USER_ID_ENABLED": os.getenv("LEXI_USER_ID_ENABLED", "0"),
    }
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    (FRONTEND_DIR / "public").mkdir(parents=True, exist_ok=True)
    for name in (".env.local", ".env"):
        (FRONTEND_DIR / name).write_text("\n".join(f"{k}={v}" for k, v in env_vars.items()) + "\n", encoding="utf-8")
    (FRONTEND_DIR / "public" / "runtime-config.js").write_text(
        f'window.__LEX_API_BASE="http://localhost:{back_port}/lexi";\n'
        f'window.__COMFY_URL="{COMFY_URL}";\n'
        f'window.__VLLM_URL="{OPENAI_API_BASE}";\n',
        encoding="utf-8",
    )
    log(f"📝 Wrote frontend env + runtime-config with ports {back_port}/{front_port}")

def launch_frontend(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["BROWSER"] = "none"
    cmd = ["npm", "run", "dev", "--prefix", str(FRONTEND_DIR)]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    threading.Thread(target=stream_output, args=(proc.stdout, "Frontend"), daemon=True).start()
    return proc

def main() -> None:
    log("🚀 LEXI STARTUP (external Comfy + vLLM)")

    if REQUIRE_COMFY:
        if not wait_for_http_ok([f"{COMFY_URL}/", f"{COMFY_URL}/queue"], timeout=TIMEOUT_SERVICE):
            log(f"⚠️  ComfyUI not reachable at {COMFY_URL}. Continuing…")

    if REQUIRE_VLLM:
        if not wait_for_http_ok([f"{OPENAI_API_BASE}/models"], timeout=TIMEOUT_SERVICE, expect_json=True):
            log(f"⚠️  vLLM not reachable at {OPENAI_API_BASE}. Continuing…")

    back_port = find_free_port(BACKEND_PORT_START) or BACKEND_PORT_START
    back = launch_backend(back_port)
    if not wait_for_backend(back_port):
        log("❌ Backend failed to start.")
        try: back.terminate()
        except Exception: pass
        return
    log("✅ Backend ready.")

    front_port = find_free_port(FRONTEND_PORT_START) or FRONTEND_PORT_START
    write_frontend_env(back_port, front_port)
    front = launch_frontend(front_port)

    try:
        while True:
            time.sleep(1)
            if back.poll() is not None:
                log("❌ Backend exited – shutting down.")
                break
            if front and front.poll() is not None:
                log("❌ Frontend exited – shutting down.")
                break
    except KeyboardInterrupt:
        log("👋 Ctrl-C received – terminating…")

    for p in (front, back):
        if p: p.terminate()
    for p in (front, back):
        if p:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill()
    log("🏁 All processes stopped.")

if __name__ == "__main__":
    main()
