# Lex — Local AI Companion Framework

Lex is an emotionally intelligent AI assistant designed for **local-first execution**, privacy, and full-stack customization. It combines a conversational backend, memory system, persona-aware avatar generation, and a beautiful frontend — all runnable on your own GPU hardware.

---

## ✨ Features

- **Memory-aware conversations** (short-term, long-term, internal monologue)
- **Dynamic persona modes** with mood shifting
- **Custom avatar generation** via Stable Diffusion XL + GFPGAN
- **Frontend interface** (React + Tailwind + Vite)
- **Offline-friendly** (no external API calls or cloud dependencies)
- **Extendable modular backend** with FastAPI
- **Natural language trait extraction** for avatar customization
- **Configurable prompt builder and visualizer tools**

---

## 🚀 Running Lex

> All commands below assume you are in the repository root (same folder as this README).

### Option A — Docker Compose (recommended)

This spins up the FastAPI backend plus the production React build. The backend expects to
reverse-proxy out to local ComfyUI and vLLM instances that run on the **host**, so the
compose file maps those endpoints to the Docker bridge IP (`172.17.0.1`).

```bash
cp .env.example .env          # customise ports/secrets if needed
docker compose up -d lexi-backend lexi-frontend
```

Key environment knobs (all overridable via `.env`):

| Variable | Purpose |
| --- | --- |
| `LLM_API_BASE`, `OPENAI_API_BASE`, `VLLM_BASE_URL`, `LITELLM_API_BASE` | bridge to your host vLLM at `172.17.0.1:8008/v1` |
| `COMFY_URL`, `COMFY_BASE_URL`, `IMAGE_API_BASE` | bridge to host ComfyUI at `172.17.0.1:8188` |
| `LEX_STATIC_ROOT` | path inside the container where static assets (avatars) live (`/app/static`) |
| `CORS_ORIGINS` | comma-separated list of allowed browser origins |

The backend returns avatar URLs such as `/lexi/static/avatars/default.png`. Traefik (or any
edge proxy) must forward both `/lexi/*` API calls and `/lexi/static/*` asset requests to the
`lexi-backend` container.

### Option B — Local venv + Vite dev server

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: customise `backend/lexi/config/config.py`

PYTHONPATH=backend uvicorn lexi.routes.lex:app --reload

cd frontend
npm install
npm run dev
```

---

## 📁 Repo Layout

- `backend/lexi/` — canonical Python package (FastAPI routes, persona, memory, SD helpers)
- `backend/scripts/` — local tooling & dev scripts
- `frontend/` — Vite/React client; serves static assets from `frontend/public/`
- `frontend/public/avatars/` — shared avatar outputs for backend ↔ frontend
- `_quarantine/` — preserved duplicates & conflict snapshots from legacy trees

---

## 🧰 Requirements

- Python 3.10+
- Node 18+ (for frontend)
- Docker Engine 24+ **or** native CUDA toolchain (12.1 suggested)
- NVIDIA GPU with ≥ 12GB VRAM (Stable Diffusion XL + vLLM)
- Host-side builds of **ComfyUI** (port `8188`) and **vLLM** (port `8008`) when using the compose stack

When running inside Docker, ensure the host firewall allows traffic from the Docker bridge
(`172.17.0.0/16`) to those services; otherwise avatar generation and LLM calls will return 502s.

---

## 🔒 Privacy, Sessions & Safety

Lex is designed with **user data privacy and local autonomy** in mind:

- No telemetry or remote logging
- All model inference happens locally
- No third-party API keys required (vLLM + Comfy run on your hardware)
- Dataset files, user logs, and experimental corpora are **excluded from this repo**

Browser clients establish a short-lived session via
`POST https://api.lexicompanion.com/lexi/alpha/session/start` and send the returned ID in the
`X-Lexi-Session` header on every request. When customising deployments, make sure your edge
proxy passes that header through and that CORS policies include it in `Access-Control-Allow-Headers`.

If you're contributing or cloning for dev purposes, be mindful not to commit:

- `*.json`, `*.jsonl`, `.env`, `*.safetensors`, checklist exports, or dataset archives

---

## 🧪 Status

> Lex is in active development. This repo is a working base for:

- AI personalization & interaction research
- Local-first AI assistant deployments
- Custom persona systems & avatar design pipelines

Expect occasional rapid iteration. Stable releases will be tagged when available.

---

## 📜 License

MIT (by default — update if you prefer something else)
