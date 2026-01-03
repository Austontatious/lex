# User ID, User Data, and Memory Tiers (opt-in)

Everything here is off by default. Flip the env flags to turn pieces on.

## Flags
- `LEXI_USER_ID_ENABLED=1` – accept `X-Lexi-User` (email or screenname) and normalize it. When off, all requests share the legacy buckets. (Defaults to on.)
- `LEXI_MEMORY_ROOT` – root folder for canonical per-user memory layout (default picks `/mnt/data/Lex/data/memory`, `/data/memory`, or `<repo>/data/memory`).
- `LEXI_MEMORY_MIGRATE_ON_START=1` – one-time migration of legacy memory files into the per-user bucket (creates `migration_done.flag`).
- `LEXI_USER_DATA_ENABLED=1` – enable per-user profile + avatar manifest storage under `LEX_USER_DATA_ROOT` (default `./memory/users/<id>/`).
- `LEXI_VECTOR_ENABLED=1` – enable Chroma ingest/search for memories. Uses local `sentence-transformers` (no network calls). Store lives at `LEXI_VECTOR_CHROMA_PATH` (default `/workspace/ai-lab/Lex/vector_store`), collection `LEXI_VECTOR_COLLECTION` (default `lex_memory`).
- `LEXI_LOG_RETENTION_DAYS` – future pruning knob for session logs (no-op today; set for ops policy).
- `LEXI_USER_API_MAX` / `LEXI_USER_API_WINDOW_SEC` – rate-limit /lexi/user/* endpoints (default 60 calls per 60s per IP).

Flags are read at call-time so you can toggle without restarting the backend. Vector enablement is also mirrored to the frontend via `.env.production` defaults.
If only one of `LEXI_MEMORY_ROOT` or `LEX_USER_DATA_ROOT` is set, it is used for both memory and user data so the buckets align.

Health check: `GET /lexi/vector/health` reports enabled status, path, collection, and current count (503 if unavailable).

## Memory tiers
- Tier 1 (ephemeral): in-session context window assembled from recent turns (prompt builder only; not persisted).
- Tier 2 (persisted per-user JSON): rolling session summaries + facts in `session_summaries.json`, plus `ltm.jsonl` for long-form notes.
- Tier 3 (vector, optional): Chroma-backed embeddings for semantic search, same user namespace.

## Canonical on-disk layout
```
<LEXI_MEMORY_ROOT>/
  users/
    <user_id_sanitized>/
      ltm.jsonl
      session_summaries.json
      session_index.json
```
`<user_id_sanitized>` preserves case. `Auston` and `auston` map to different buckets.

## What gets persisted
- Profiles: `users/<id>/profile.json` with `id`, `created_at`, `last_seen`, optional `email`, `display_name`, `attributes`.
- Avatar manifest: `users/<id>/avatars_manifest.json` keeps **first** and **latest** avatar events plus prompt/traits/mode/seed/session metadata.
- Memory (Tier 2): `ltm.jsonl` (long-form notes) + `session_summaries.json` (rolling summary + facts).
- Memory (Tier 3): optional vector ingest mirrors the same namespace when `LEXI_VECTOR_ENABLED` is set.

## Memory tools
- `POST /tools/memory_get_profile` returns rolling summary + facts for the current user.
- `POST /tools/memory_search_ltm` accepts `{ "query": "...", "k": 5 }` and returns LTM hits for the current user.
- The persona tool planner knows about `memory_search_ltm`, but invocation is still placeholder and results are not auto-injected.

## Flow wiring (when enabled)
- Requests resolve a stable `user_id` (header, session registry, or anon fallback).
- Persona binding (`LexiPersona.set_user`) swaps memory + session paths into the per-user bucket and tags vector ingest.
- Avatar generation stores first/latest into the manifest and reuses the latest as img2img source when possible.
- Memory writes update `session_summaries.json` and append to `ltm.jsonl` with the same `user_id` namespace.

## Safe defaults
- All helpers short‑circuit when flags are off or IDs are missing.
- Vector ingest returns quietly if Chroma or the embed model is unavailable.
- Avatar manifest ignores HTTP URLs as img2img sources; prefers local/static files.

## Turning it on (staging)
1. Set `LEXI_USER_ID_ENABLED=1 LEXI_USER_DATA_ENABLED=1` (and optionally `LEXI_VECTOR_ENABLED=1`) in the backend env.
2. Set `LEXI_MEMORY_ROOT` to a writable path; first run will create the `users/<id>/` structure.
3. (Optional) Set `LEXI_MEMORY_MIGRATE_ON_START=1` to import legacy `lex_memory.jsonl`/`session_memory.json`.
4. (Optional) Set `VITE_USER_ID_ENABLED=1` to prompt the frontend for email/name so the header is sent automatically.
5. Exercise persona + chat; check `<LEXI_MEMORY_ROOT>/users/<id>/` for new memory files.
