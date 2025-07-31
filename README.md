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

> ⚠️ This assumes you're working from the `lex/` directory root.

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
uvicorn routes.lex:app --reload
```

---

### 4. Run the frontend (optional but pretty)

```bash
cd frontend
npm install
npm run dev
```

---

## 🧰 Requirements

- Python 3.10+
- Node 18+ (for frontend)
- CUDA 12.1 (or compatible with your PyTorch build)
- A GPU with >= 12GB VRAM (for avatar generation)

---

## 🔒 Privacy & Safety

Lex is designed with **user data privacy and local autonomy** in mind:

- No telemetry or remote logging
- All model inference happens locally
- No third-party API keys required
- Dataset files, user logs, and experimental corpora are **excluded from this repo**

If you're contributing or cloning for dev purposes, be mindful not to commit:

- `*.json`, `*.jsonl`, `.env`, `*.safetensors`, or dataset archives

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
