# Lexi Alpha — Production & Dev Deployment Guide

Lexi ships with two Docker Compose profiles:

| Profile  | Services                            | Use case                              |
|----------|-------------------------------------|---------------------------------------|
| `dev`    | `lexi-backend`, `lexi-frontend-dev` | Local development on host ports       |
| `live`   | `lexi-backend`, `lexi-frontend`     | Production containers behind Traefik  |

> **Important:** This stack no longer launches its own Traefik proxy.  
> A host-wide Traefik instance (or equivalent) terminates TLS and routes traffic to the Lexi containers over a shared Docker network named `edge`.

---

## 1. Prerequisites

- Docker Engine 24+ and Docker Compose v2
- Git clone of `https://github.com/Austontatious/lex`
- Domain: `lexicompanion.com` managed in Cloudflare
- Host-level Traefik container configured with:
  - `web` (HTTP/80) and `websecure` (HTTPS/443) entrypoints
  - ACME resolver named `le` (http-01 or dns-01)
  - Docker provider enabled
- Shared Docker network `edge`

```bash
docker network create edge   # run once on the host
# Attach Traefik if it is already running
docker network connect edge <traefik-container-name>
```

If you prefer Cloudflare DNS-01 challenges, create an API token with `Zone.DNS:Edit` and provide it via `CF_DNS_API_TOKEN` in `.env`.

---

## 2. Environment configuration

```bash
cp .env.example .env
```

Populate the following fields at minimum:

| Variable                | Description                                                         |
|-------------------------|---------------------------------------------------------------------|
| `CF_DNS_API_TOKEN`      | Cloudflare token (DNS-01). Leave blank if Traefik uses http-01.     |
| `TRAEFIK_ACME_EMAIL`    | Email address used for ACME registration.                           |
| `COMFY_URL`, `OPENAI_*` | Endpoints the backend should call for ComfyUI / vLLM.               |
| `LEX_API_BASE_PUBLIC`   | Public API base presented to the frontend (e.g. `/api`).            |
| `SD_BACKEND`, `FLUX_*`  | Stable Diffusion/FLUX model configuration paths.                    |

Secrets live only in `.env`; the file is git-ignored.

---

## 3. Local development

```bash
# Boots backend (Gunicorn on 0.0.0.0:8000) and frontend dev server on 3000
make dev

# Health checks
curl -fsS http://localhost:9000/lex/health          # backend via published port
open http://localhost:3000                          # frontend dev server
```

Use `make down` to stop all containers and `make logs` to tail combined logs.

---

## 4. Production deployment (behind Traefik)

1. Ensure the `edge` network exists and Traefik is attached (see prerequisites).
2. Export your `.env` values or run the stack from a shell where they are defined.
3. Launch the live profile:

```bash
docker compose --profile live up -d lexi-backend lexi-frontend
```

The containers register two routers with Traefik:

- `https://lexicompanion.com/api/*` → `lexi-backend` (port 8000) with `/api` stripped
- `https://lexicompanion.com/*` → `lexi-frontend` (port 80)

### Optional: Cloudflare Tunnel

`docker-compose.yml` still includes a `cloudflared` service (profile `live`).  
Enable it only if you need tunnel-based access:

```bash
docker compose --profile live up -d cloudflared
```

Mount `/mnt/data/Lex/cloudflared/config.yml` with your ingress settings and keep the Cloudflare credentials JSON under `/root/.cloudflared/` on the host.

---

## 5. Verification checklist

```bash
# Backend health through Traefik (API router strips /api prefix)
curl -fsS https://lexicompanion.com/api/health

# Frontend root
curl -I https://lexicompanion.com

# Container status
docker compose ps
docker compose logs -f lexi-backend
docker compose logs -f lexi-frontend
```

If you kept the tunnel running:

```bash
docker compose logs -f cloudflared
```

---

## 6. Troubleshooting

| Symptom                                   | Action                                                                 |
|-------------------------------------------|------------------------------------------------------------------------|
| `404` on `/api/health`                    | Ensure Traefik picked the API router (priority `100` vs frontend `10`).|
| ACME challenge failures                   | Verify Cloudflare record matches the chosen challenge (dns vs http).   |
| Traefik still serving old config          | Restart Traefik or run `docker network disconnect edge <container>` then reconnect. |
| Backend cannot reach Comfy/vLLM           | Confirm `COMFY_URL`/`OPENAI_API_BASE` endpoints from inside the container. |

---

## 7. Maintenance commands

```bash
make logs         # follow all service logs
make ps           # container status
make down         # stop all running services (dev or live)
docker compose build --no-cache  # rebuild images
```

Keep the `letsencrypt/` directory (ACME state) backed up if you rely on Traefik DNS-01 challenges.

---

Happy deploying! For deeper architecture details, see the root `README.md`.
