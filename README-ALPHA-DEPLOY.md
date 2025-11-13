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
| `LLM_API_BASE`, `OPENAI_API_BASE`, `LITELLM_API_BASE`, `VLLM_BASE_URL` | All point to the host vLLM bridge `http://172.17.0.1:8008/v1`. |
| `COMFY_URL`, `COMFY_BASE_URL`, `IMAGE_API_BASE` | Host bridge for ComfyUI (`http://172.17.0.1:8188`). |
| `LEX_API_BASE_PUBLIC`   | Public API base presented to the frontend (e.g. `/api`).            |
| `LEX_STATIC_ROOT`       | Directory inside the backend image where avatars & static assets live. |
| `CORS_ORIGINS`          | Comma-separated list of allowed browser origins (include `https://lexicompanion.com`). |
| `FLUX_*`                | Flux model configuration paths.                                  |
| `LEXI_AVATAR_JOB_TTL`   | Seconds to retain completed avatar jobs in memory (default `600`). |
| `LEXI_SKIP_FLUX_WARMUP` | Set to `1` to skip the Flux warm-up call at startup (default runs once). |

Secrets live only in `.env`; the file is git-ignored.

### Browser sessions & required headers

Frontend clients call `POST /lexi/alpha/session/start` once, persist the returned
`session_id`, and then send it on every request via
`X-Lexi-Session`. Make sure:

- The edge proxy forwards `X-Lexi-Session` to the backend.
- FastAPI’s CORS middleware allows the header (`allow_headers` must include `X-Lexi-Session`).
- Any Cloudflare Workers or other intermediaries also echo CORS headers on error paths; otherwise a 502 can masquerade as a “CORS blocked” message in DevTools.

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

# Avatar queue behaviour (should return queued/done quickly)
curl -fsS https://lexicompanion.com/api/lexi/persona/avatar
# If you see {"status":"queued","job_id":...}, poll the status endpoint:
curl -fsS https://lexicompanion.com/api/lexi/persona/avatar/status/<job_id>

# Avatar & static assets are served by the backend itself
curl -I https://lexicompanion.com/lexi/static/avatars/default.png

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
| Browser reports "CORS blocked" on `/lexi/process` | Usually an upstream 502. Check that ComfyUI (8188) and vLLM (8008) are reachable from inside the `lexi-backend` container and that `CORS_ORIGINS` includes the requesting origin. |
| `/lexi/persona/avatar` takes >100s and Cloudflare shows 524 | The route now returns immediately; if you still see long waits verify your proxy isn’t buffering responses and that the frontend is polling `/lexi/persona/avatar/status/<job_id>`. |
| Session header missing                    | Ensure your proxy forwards `X-Lexi-Session` and that `CORSMiddleware`’s `allow_headers` includes it. |

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
