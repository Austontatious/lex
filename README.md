# Lex — Local AI Companion Framework

Lex is an emotionally intelligent AI assistant designed for **local-first execution**, privacy, and full-stack customization. It combines a conversational backend, memory system, persona-aware avatar generation, and a beautiful frontend — all runnable on your own GPU hardware.

---

## ✨ Features

- **Memory-aware conversations** (short-term, long-term, internal monologue)
- **Dynamic persona modes** with mood shifting
- **Custom avatar generation** via ComfyUI (Flux/SDXL workflows) with warn-only fallbacks
- **Frontend interface** (React + Tailwind + Vite)
- **Offline-friendly** (no external API calls or cloud dependencies)
- **Extendable modular backend** with FastAPI
- **Natural language trait extraction** for avatar customization
- **Configurable prompt builder and visualizer tools**

---

## 🚀 Running Lex

> All commands below assume you are in the repository root (same folder as this README).

### Environment Layout

- Copy `.env.example` to `.env.development` and `.env.production`.
- `docker compose up` (and `make dev`) read `.env.development` by default.
- Production runs layer `docker-compose.override.prod.yml` and `.env.production`.
- Secrets belong only in the real runtime `.env.*` files, Docker/Swarm secrets, or a vault.
- Session NDJSON logs land under `logs/sessions/YYYY-MM-DD/<session>.ndjson`. Mount this path in prod for durability.

- Host integrations rely on the Compose-provided gateway alias:
  - `LLM_API_BASE`, `OPENAI_API_BASE`, `SUMMARIZER_ENDPOINT` → `http://host.docker.internal:8008/v1`
  - `COMFY_URL`, `COMFY_BASE_URL`, `IMAGE_API_BASE`, `LEX_COMFY_URL` → `http://comfy-sd:8188` (bundled ComfyUI). Switch to `http://host.docker.internal:8188` only if you run Comfy on the host.
  - `LEX_SDXL_CHECKPOINT` must be the checkpoint **filename** (e.g. `flux1-kontext-dev.safetensors`) so Comfy’s `CheckpointLoaderSimple` can resolve it.
- `LEX_USE_COMFY_ONLY=1` keeps avatar preflight errors soft (warn JSON instead of 502). `LEX_AVATAR_DIR` (default `/app/static/avatars`) and `LEX_PUBLIC_BASE` influence the generated avatar URLs.
- The frontend defaults to `https://api.lexicompanion.com/lexi` whenever `window.location.hostname` ends with `lexicompanion.com`. Dev builds continue to use `/lexi` unless `/config.json` provides a different base.
- FastAPI ships with permissive CORS, but production should set `CORS_ORIGINS=https://lexicompanion.com` (append other origins with commas). The middleware already exposes `X-Lexi-Session`.
- Running the optional Cloudflare tunnel requires attaching `lex-cloudflared-1` to `lexnet` so service discovery finds `lexi-frontend` and `lexi-backend`.

### Local Dev (bundled Comfy + host vLLM)

```bash
cp .env.example .env.development
make dev
make backcurl     # curl $COMFY_URL/api/version from inside the backend container
```

- `docker-compose.yml` builds the `comfy-sd` image (Flux-ready ComfyUI) and pins GPUs `4,5` by default; adjust `NVIDIA_VISIBLE_DEVICES` as needed.
- Containers still map `host.docker.internal` → the host gateway so the backend reaches vLLM (`http://host.docker.internal:8008`) and other host-only services.
- `BASE_MODELS_DIR` defaults to `/mnt/data/models`; override it per machine in `.env.development`.
- `make logs`, `make ps`, and `make sh` are handy while iterating (`make help` lists everything).

### Production (bundled Comfy service)

```bash
cp .env.example .env.production
make prod
make backcurl
```

- `docker-compose.override.prod.yml` mounts `/var/lib/lex/models` into the backend as read-only `/models`.
- The backend favors the `/models` mount but still honors `LEX_SDXL_CHECKPOINT` (filename only) to pick the base checkpoint.
- Keep `host.docker.internal:host-gateway` unless every service runs in the same compose project. When everything lives inside Compose, attach the services to `lexnet` so DNS succeeds.
- Prefer the bundled `comfy-sd` service; if you have a host ComfyUI, set `COMFY_URL`, `COMFY_BASE_URL`, `IMAGE_API_BASE`, and `LEX_COMFY_URL` to `http://host.docker.internal:8188` (or whatever address you expose).

### Alternate Comfy setups

The default build wraps the upstream ComfyUI repo in `docker/comfyui/Dockerfile`. Add extra Python deps for custom nodes there, rebuild, then run `make dev` / `make prod`.
If you disable the `comfy-sd` service, remember to point every Comfy-related environment variable at your replacement endpoint.

### Manual dev (venv + Vite)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=backend uvicorn lexi.routes.lex:app --reload

cd frontend
npm install
npm run dev
```

### Sanity checks

```bash
# 1) Build/run in dev
make dev && make backcurl

# 2) Inspect container gateway
docker compose exec lexi-backend sh -lc 'ip route | awk "/default/ {print \$3}"'

# 3) Confirm backend → Comfy reachability
docker compose exec lexi-backend sh -lc 'curl -sS $COMFY_URL/api/version && echo'

# 4) Build/run prod (bundled Comfy service)
make prod && make backcurl

# 5) Trigger an avatar edit (warn-only paths still return HTTP 200)
curl -sS https://api.lexicompanion.com/lexi/process \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"make my avatar purple hair","intent":"avatar_edit"}' | jq

# 6) Confirm warn-only behaviour in logs
docker logs -f lex-lexi-backend-1 | grep "Avatar intent"
```

### Key API routes

- `POST /lexi/intent` → `{"text": str}` and returns `{"intent": str}`.
- `POST /lexi/process` expects `{"prompt": str, "intent"?: str}`; include `intent="avatar_edit"` to trigger the Comfy avatar pipeline.
- `GET /lexi/health` and `/lexi/ready` provide basic container status checks.

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
