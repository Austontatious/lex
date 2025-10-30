# Lexi — Local AI Companion Framework

Lexi is an emotionally intelligent AI assistant designed for **local-first execution**, privacy, and full-stack customization. It combines a conversational backend, memory system, persona-aware avatar generation, and a beautiful frontend — all runnable on your own GPU hardware.

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

## 🚀 Running Lexi

> ⚠️ This assumes you're working from the `lexi/` directory root.

### 1. Set up the Python backend

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your local paths

Copy and edit:

```bash
cp config.sample.py config.py
```

Update model paths, device settings, and runtime flags in `config.py`.

### 3. Run the backend

```bash
uvicorn routes.lexi:app --reload
```

---

### 4. Run the frontend (optional but pretty)

```bash
cd frontend
npm install
npm run dev
```

---

## 🐳 Containerised deployment

Prefer Docker? The repository root includes a Compose stack that packages the Gunicorn FastAPI backend and production frontend:

```bash
cd ..
cp .env.example .env   # configure credentials and runtime paths
make dev               # backend on :9000, frontend dev on :3000
make down              # stop everything
```

For a production rollout behind Traefik/Cloudflare, follow the detailed checklist in `README-ALPHA-DEPLOY.md`. The live profile registers two routers:

- `https://lexicompanion.com/api/*` → backend (after stripping `/api`)
- `https://lexicompanion.com/*`     → static frontend

---

## 🧰 Requirements

- Python 3.10+
- Node 18+ (for frontend)
- CUDA 12.1 (or compatible with your PyTorch build)
- A GPU with >= 12GB VRAM (for avatar generation)

---

## ⚙️ Environment

These environment variables configure the SDXL/ComfyUI pipeline and file locations. Sensible defaults are provided.

- `COMFY_URL` – ComfyUI server URL. Default: `http://127.0.0.1:8188`
- `COMFY_BASE_CKPT` – Base SDXL checkpoint filename. Default: `sd_xl_base_1.0.safetensors`
- `COMFY_REFINER_CKPT` – SDXL refiner checkpoint filename. Default: `sd_xl_refiner_1.0.safetensors`
- `COMFY_UPSCALE` – Enable simple latent upscale hop. Default: `false` (accepts `1|true|yes`)
- `LEX_IMAGE_DIR` – Output directory for generated avatars. Default: `<project>/Lexi/lexi/static/lexi/avatars`
- `LEX_WORKFLOW_FACE` – Path to the portrait workflow JSON. Default: `Lexi/Lexi/lexi/sd/workflows/face_workflow.json`
- `LEX_WORKFLOW_BODY` – Path to the full-body workflow JSON. Default: `Lexi/Lexi/lexi/sd/workflows/body_workflow.json`

Notes:
- Workflow paths can be absolute or relative. Relative paths are resolved against the project root so they’re stable regardless of CWD.
- Checkpoints and LoRAs must be visible to ComfyUI under its `models/` tree; pass only filenames (Comfy will resolve subfolders).

Example (bash):

```bash
export COMFY_URL="http://127.0.0.1:8188"
export COMFY_BASE_CKPT="sd_xl_base_1.0.safetensors"
export COMFY_REFINER_CKPT="sd_xl_refiner_1.0.safetensors"
export COMFY_UPSCALE="true"
export LEX_IMAGE_DIR="$(pwd)/Lexi/lexi/static/lexi/avatars"
export LEX_WORKFLOW_FACE="Lexi/Lexi/lexi/sd/workflows/face_workflow.json"
export LEX_WORKFLOW_BODY="Lexi/Lexi/lexi/sd/workflows/body_workflow.json"
```

---

## 🖼️ Avatar Pipeline

- Endpoints live under `/lexi/persona` and return absolute image URLs.
- Images are saved under `Lexi/lexi/static/lexi/avatars` (configurable via `LEX_IMAGE_DIR`).
- Workflows are JSON files under `Lexi/lexi/sd/workflows` and are patched at runtime (prompts, samplers, checkpoints, image inputs).

**New Look (txt2img base)**
- Chat: say “new look”, “start over”, or “fresh look”. The backend creates/overwrites `lexi_base.png`.
- API: POST `/lexi/persona/generate_avatar` with either:
  - `{"mode":"txt2img"}`
  - or `{"fresh_base": true}`

**Change Your Look (img2img edit)**
- Chat: say “change your look …” with a short edit.
- API: POST `/lexi/persona/generate_avatar` e.g. `{"mode":"img2img", "changes":"black leather jacket, studio gray background", "denoise":0.45}`

Notes
- The API converts relative `/static/...` paths to absolute URLs using the request’s host/port.
- Workflows without a `SaveImage` node are auto‑fixed so `/history/<prompt_id>` always reports images.
- Boot log prints `[Lexi SD] COMFY_URL=...`; export `COMFY_URL` in the same shell before starting if needed.
- Frontend derives API base from `window.location.origin` and uses a 240s timeout for avatar renders.

---

## 🔒 Privacy & Safety

Lexi is designed with **user data privacy and local autonomy** in mind:

- No telemetry or remote logging
- All model inference happens locally
- No third-party API keys required
- Dataset files, user logs, and experimental corpora are **excluded from this repo**

If you're contributing or cloning for dev purposes, be mindful not to commit:

- `*.json`, `*.jsonl`, `.env`, `*.safetensors`, or dataset archives

---

## 🧪 Status

> Lexi is in active development. This repo is a working base for:

- AI personalization & interaction research
- Local-first AI assistant deployments
- Custom persona systems & avatar design pipelines

Expect occasional rapid iteration. Stable releases will be tagged when available.

---

## 📜 License

MIT (by default — update if you prefer something else)
