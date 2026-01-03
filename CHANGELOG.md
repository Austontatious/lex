# Changelog

## 2026-01-03
- Hardened session cookie handling and feedback file writes (secure flag via scheme/forwarded headers; exclusive create with restrictive perms; removed /tmp fallback).
- Capped intent input length at the API boundary to avoid regex DoS risks.
- Fixed async reliability issues (async HTTP client in startup; retain background tasks to prevent GC; clarified SD pipeline mode handling).
- Cleaned frontend artifacts (restored CRA public template with runtime config; removed misplaced build outputs; fixed TS empty-type intersection).
- Added explicit Git safety rules for push-only operations in agent guidance.

## 2025-12-02
- Locked frontend runtime config to the canonical API URL by setting `LEX_API_BASE` and `LEX_API_BASE_PUBLIC` to `https://api.lexicompanion.com/lexi` in `docker-compose.override.yml` and `docker-compose.override.prod.yml` so entrypoint-generated `runtime-config.js` no longer picks up the dev localhost defaults.
- Verified only writers of `runtime-config.js` are the frontend entrypoint and the dev helper (`start_lexi.py`), and the live app resolves API_BASE from same-origin routing, so this change is safe but prevents future localhost leakage on restart.
