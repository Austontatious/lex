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
- **Model & prompt tuned** for Lexi: Qwen3-30B-A3B (MoE) base with a Lexi DPO adapter, plus an updated system prompt (`backend/lexi/persona/prompt_templates/lexi_system_qwen3_moe.txt`) emphasizing sensory detail, warm SFW tone, and concise arcs. Override with `LEXI_SYSTEM_PROMPT_PATH` to hot-swap or roll back.

### Served model (merged)
- Merged checkpoint: `/mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged`
- Example vLLM launch (legacy host fallback):
  ```
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged \
    --served-model-name Lexi \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --max-num-seqs 8 \
    --swap-space 12 \
    --max-model-len 32000 \
    --enforce-eager \
    --gpu-memory-utilization 0.88 \
    --host 0.0.0.0 --port 8008
  ```

---

## 🚀 Running Lex

> All commands below assume you are in the repository root (same folder as this README).

### Environment Layout

- Copy `.env.example` to `.env.development` and `.env.production`.
- `docker compose up` (and `make dev`) read `.env.development` by default.
- Production runs layer `docker-compose.override.prod.yml` and `.env.production`.
- Compose profiles are controlled via `.env` (default `COMPOSE_PROFILES=default`); use `COMPOSE_PROFILES=no-vllm` to disable the vLLM container.
- Secrets belong only in the real runtime `.env.*` files, Docker/Swarm secrets, or a vault.
- Session NDJSON logs land under `logs/sessions/YYYY-MM-DD/<session>.ndjson`. Mount this path in prod for durability.

- vLLM (default Compose service):
  - `LLM_API_BASE`, `OPENAI_API_BASE`, `SUMMARIZER_ENDPOINT` → `http://vllm:8008/v1`
  - Host fallback (no-vllm profile) → `http://host.docker.internal:8008/v1`
- Comfy (default Compose service):
  - `COMFY_URL`, `COMFY_BASE_URL`, `IMAGE_API_BASE`, `LEX_COMFY_URL` → `http://comfy-sd:8188` (bundled ComfyUI). Switch to `http://host.docker.internal:8188` only if you run Comfy on the host.
  - `FLUX_MODELS_DIR`, `FLUX_DIFFUSION_DIR`, `FLUX_TEXT_ENCODER_DIR`, and `FLUX_VAE_PATH` must resolve to your Flux assets (e.g. `flux1-kontext-dev.safetensors`, `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`, `ae.safetensors`).
- `LEX_USE_COMFY_ONLY=1` keeps avatar preflight errors soft (warn JSON instead of 502). `LEX_AVATAR_DIR` (default `/app/frontend/public/avatars`) and `LEX_PUBLIC_BASE` influence the generated avatar URLs.
- Live “Now” feed + movies tool: set `TMDB_API_KEY` **or** `TMDB_READ_ACCESS_TOKEN` in `.env.*` (one is enough); leave `ENABLE_NOW` unset to keep the default `True`.
- `LEX_SKIP_COMFY_WARMUP=1` skips the Comfy warm-up call if you need ultra-fast cold starts (not recommended for production).
- The frontend defaults to `https://api.lexicompanion.com/lexi` whenever `window.location.hostname` ends with `lexicompanion.com`. Dev builds continue to use `/lexi` unless `/config.json` provides a different base.
- The bundled NGINX config now proxies `/lexi/*` to the backend and falls back to `index.html` for deep SPA routes.
- FastAPI ships with permissive CORS, but production should set `CORS_ORIGINS=https://lexicompanion.com` (append other origins with commas). The middleware already exposes `X-Lexi-Session`.
- Running the optional Cloudflare tunnel requires attaching `lex-cloudflared-1` to `lexnet` so service discovery finds `lexi-frontend` and `lexi-backend`.

### Local Dev (bundled Comfy + bundled vLLM)

```bash
cp .env.example .env.development
make dev
make backcurl     # curl $COMFY_URL/api/version from inside the backend container
```

- `docker-compose.yml` builds the `comfy-sd` image (Flux-ready ComfyUI) and pins GPU `4` by default; vLLM pins GPUs `0,1,2,3`.
- Escape hatch (host vLLM): `COMPOSE_PROFILES=no-vllm docker compose up -d`, then launch vLLM on the host.
- `BASE_MODELS_DIR` defaults to `/mnt/data/models`; override it per machine in `.env.development`.
- `make logs`, `make ps`, and `make sh` are handy while iterating (`make help` lists everything).

### Production (bundled Comfy service)

```bash
cp .env.example .env.production
make prod
make backcurl
```

- `docker-compose.override.prod.yml` mounts `/var/lib/lex/models` into the backend as read-only `/models`.
- The backend favors the `/models` mount; configure `FLUX_MODELS_DIR`, `FLUX_DIFFUSION_DIR`, `FLUX_TEXT_ENCODER_DIR`, and `FLUX_VAE_PATH` so the container sees your Flux checkpoint bundle.
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

### Verification

```bash
# Default (containerized vLLM)
docker compose up -d
curl -sS http://localhost:8008/v1/models
nvidia-smi

# Host fallback (no-vllm profile)
COMPOSE_PROFILES=no-vllm docker compose up -d
CUDA_VISIBLE_DEVICES=0,1,2,3 /mnt/data/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /mnt/data/models/Qwen/lexi-qwen3-30b-a3b-dpo-merged \
  --served-model-name Lexi \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --max-num-seqs 8 \
  --swap-space 12 \
  --max-model-len 32000 \
  --enforce-eager \
  --gpu-memory-utilization 0.88 \
  --host 0.0.0.0 --port 8008

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

# 7) Health probes
docker compose exec lexi-backend sh -lc 'curl -sS http://127.0.0.1:9000/lexi/healthz && echo'
docker compose exec lexi-backend sh -lc 'curl -sS http://127.0.0.1:9000/lexi/readyz && echo'
```

### Key API routes

- `POST /lexi/intent` → `{"text": str}` and returns `{"intent": str}`.
- `POST /lexi/process` expects `{"prompt": str, "intent"?: str}`; include `intent="avatar_edit"` to trigger the Comfy avatar pipeline (its `avatar_url` now reflects the per-IP static asset described below).
- `GET /lexi/persona/avatar` is now asynchronous: if an avatar already exists the backend returns `{"status":"done","avatar_url":...}` immediately; otherwise it enqueues a job and replies `{"status":"queued","job_id":"..."}`
- `GET /lexi/persona/avatar/status/{job_id}` reports `queued | running | done | error` and, once complete, includes the final `avatar_url`.
- `POST /lexi/alpha/session/start` returns `{"session_id": "...", ...}`. Keep the ID and send it back in the `X-Lexi-Session` header for persona/avatar routes.
- `POST /lexi/gen/avatar` requires `X-Lexi-Session`; otherwise the backend returns HTTP 401 with a hint to start a session.
- `GET /lexi/healthz` performs a lightweight liveness check (Comfy `/object_info` + avatar-dir write).
- `GET /lexi/readyz` validates the schema expectations and submits a no-op `/prompt` to ensure Comfy is ready.

### Optional: per-user IDs (email-or-name) — disabled by default

- Enable with `LEXI_USER_ID_ENABLED=1` on the backend. In Docker, the frontend entrypoint mirrors it into `runtime-config.js` so the prompt appears. (`start_lexi.py` is deprecated.)
- When enabled, the frontend asks for "email or name", stores it in `localStorage` (`LEXI_USER_ID`), and sends it on every request as `X-Lexi-User`. There is intentionally no validation; nicknames are fine.
- User ids are case-sensitive when mapped to storage; keep casing consistent (`Auston` and `auston` are different buckets).
- Backend binds that id to:
  - persona state under `backend/lexi/routes/users/<user>/lexi_persona_state.json`,
  - long-term memory under `Lex/memory/users/<user>/ltm.jsonl`,
  - session summaries under `Lex/memory/users/<user>/session_memory.json`,
  - session logs/metrics tagged with the normalized id.
- If the flag is off or the header is missing, everything continues to use the existing shared stores and UX. Users can also click "skip" and fall back to the shared bucket even when the flag is on.

### Experimental: vector memory + avatar manifest (disabled by default)

- Gate vector search/ingest behind `LEXI_VECTOR_ENABLED=1`. Storage lives at `LEXI_VECTOR_CHROMA_PATH` (default `/workspace/ai-lab/Lex/vector_store`) with collection name `LEXI_VECTOR_COLLECTION` (`lex_memory`). Embeddings use a local `sentence-transformers` model; no remote calls.
- Turn on per-user profile + avatar manifests with `LEXI_USER_DATA_ENABLED=1` (writes under `LEX_USER_DATA_ROOT`, default `<repo>/memory/users/<id>/...`). Keeps the first avatar and the latest edit with prompt/traits/seed metadata for continuity; when disabled it is a no-op.
- If only one of `LEXI_MEMORY_ROOT` or `LEX_USER_DATA_ROOT` is set, it is used for both memory and user data so the buckets align.
- See `docs/user_data_and_vectors.md` for how the flags combine and where artifacts are stored.

**Avatar job queue highlights**

- Every IP maps to `/lexi/static/avatars/<ip>.png`. If a render is missing, `GET /lexi/persona/avatar` enqueues a background job and returns `{"status":"queued","job_id":...}` immediately, eliminating Cloudflare 524s and proxy read timeouts.
- Poll `/lexi/persona/avatar/status/<job_id>` until it returns `{"status":"done","avatar_url":...}` (or `{"status":"error","error":...}`) and then use that URL directly in the UI.
- Completed jobs stick around in-memory for `LEXI_AVATAR_JOB_TTL` seconds (default 600). Override this env var if you need a different retention window.
- `frontend/src/lib/refreshAvatar.ts` implements the polling + swap logic so browsers always render the freshest PNG (note the built-in cache buster).
- While avatar generation runs, the avatar pane shows a muted looping video from `frontend/public/media/avatar-loading.mp4`; swap in your own clip there (keep it short, muted, ~720p and reasonably small—e.g., <10 MB) to avoid perceived freezes during long diffusion runs.

---

## 📁 Repo Layout

- `backend/lexi/` — canonical Python package (FastAPI routes, persona, memory, SD helpers)
- `backend/scripts/` — local tooling & dev scripts
- `frontend/` — Vite/React client; serves static assets from `frontend/public/`
- `frontend/public/avatars/` — shared avatar outputs for backend ↔ frontend
- `_quarantine/` — preserved duplicates & conflict snapshots from legacy trees

### Snapshot duplicates & why they exist

You’ll notice two almost-identical trees: `backend/lexi/...` and `Lexi/lexi/...`.
Docker builds, tests, and imports all target `backend/lexi`, so treat it as the **source of truth**.
The `Lexi/` tree is a frozen snapshot kept around to hotfix older deployments without rebasing.

When you patch backend code, edit `backend/lexi/...` first, then mirror the change into
`Lexi/lexi/...` only if the snapshot still needs to track it. Otherwise you’ll see confusing
behaviour where containers keep serving the old code even though your local tests pass.

Signs you touched the wrong tree:

- Docker builds succeed but your change never appears at runtime.
- `pytest` can’t see the change that “worked in dev”.
- Git diffs show noise under `_quarantine`.

Keep the duplication in mind whenever you modify routes, persona logic, or SD helpers.

### Avatar pipeline quick facts

- The backend always pushes graphs to ComfyUI; the legacy local diffusers path is disabled.
- Flux is the only supported avatar backend; SDXL workflows and refiners have been retired.
- `FLUX_*` environment variables must match the paths visible to the Comfy container (Compose mounts `/mnt/data/comfy` read-only into the backend to guarantee this).
- Flux tuning lives in `backend/lexi/sd/flux_defaults.py`; update that file once and every Comfy graph + backend default stays in sync.
- Prompt scaffolding lives in `backend/lexi/sd/flux_prompt_builder.py`, keeping the “Lexiverse” base text immutable while allowing trait/style deltas appended at runtime.
- Current defaults (Dec 2025): 768×1344 canvas, 24 steps, cfg 3.0, guidance 2.8/2.8, sampler `euler`, denoise 1.0; ControlNet enabled by default at 0.55 unless `controlnet_enabled=false` or `LEXI_CONTROLNET_ENABLED=0`.
- Prompt guardrails: positive includes “healthy feminine body proportions” and “wide framing, subject occupies ≤70% of vertical frame”; negatives add framing blockers plus “halo/outline/glow/rimlight/backlight/overprocessed HDR/bloom” and “overdefined muscles/bodybuilder physique.”
- Style prompt swapped to neutral studio light: “neutral studio key light + soft fill, no rimlight, no neon; natural skin texture, realistic specular highlights; filmic contrast, neutral color grade, accurate white balance.”
- ControlNet is automatically skipped if the model asset is missing; denoise stays 1.0 for txt2img even when CN is on.
- `backend/lexi/utils/request_ip.py` and `utils/ip_seed.py` derive deterministic per-IP seeds and filenames so every user keeps a consistent identity unless you override the seed manually.
- Populate those folders with the Flux bundle (e.g. `flux1-kontext-dev.safetensors`, `clip_l.safetensors`, `t5xxl_fp8_e4m3fn.safetensors`, `ae.safetensors`). Example:

```bash
mkdir -p /mnt/data/comfy/models/diffusion_models
wget https://example.com/flux1-kontext-dev.safetensors -O /mnt/data/comfy/models/diffusion_models/flux1-kontext-dev.safetensors
```
- On import we call `/object_info`, validate the schema, and warm up Comfy once. Set `LEXI_SKIP_FLUX_WARMUP=1` if you need to disable the warmup hit.
- The base avatar (`lexi_base.png`) is written under a file lock so simultaneous prompts do not corrupt the baseline image.
- The frontend no longer waits on long-running HTTP responses: avatar renders happen in the background, and the `/lexi/persona/avatar` + `/status/<job_id>` pair carries progress to the UI. If you need longer retention for completed jobs, tweak `LEXI_AVATAR_JOB_TTL`.
- Action item: img2img edits are temporarily disabled; all avatar updates are routed through txt2img. Revisit when we want controlled edit-mode back.

---

## 🧰 Requirements

- Python 3.10+
- Node 18+ (for frontend)
- Docker Engine 24+ **or** native CUDA toolchain (12.1 suggested)
- NVIDIA GPU with ≥ 12GB VRAM (Stable Diffusion XL + vLLM)
- Compose builds for **ComfyUI** (port `8188`) and **vLLM** (port `8008`) by default; host builds are only needed for the `no-vllm` fallback.

When running the host fallback, ensure the host firewall allows traffic from the Docker bridge
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
