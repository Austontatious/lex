# User ID, User Data, and Vector Memory (opt-in)

Everything here is off by default. Flip the env flags to turn pieces on.

## Flags
- `LEXI_USER_ID_ENABLED=1` – accept `X-Lexi-User` (email or screenname) and normalize it. When off, all requests share the legacy buckets.
- `LEXI_USER_DATA_ENABLED=1` – enable per-user profile + avatar manifest storage under `LEX_USER_DATA_ROOT` (default `./memory/users/<id>/`).
- `LEXI_VECTOR_ENABLED=1` – enable Chroma ingest/search for memories. Uses local `sentence-transformers` (no network calls). Store lives at `LEXI_VECTOR_CHROMA_PATH` (default `/workspace/ai-lab/Lex/vector_store`), collection `LEXI_VECTOR_COLLECTION` (default `lex_memory`).

## What gets persisted
- Profiles: `users/<id>/profile.json` with `id`, `created_at`, `last_seen`, optional `email`, `display_name`, `attributes`.
- Avatar manifest: `users/<id>/avatars_manifest.json` keeps **first** and **latest** avatar events plus prompt/traits/mode/seed/session metadata.
- Memory: long‑term JSONL (`ltm.jsonl`) and session summaries are already per-user when `LEXI_USER_ID_ENABLED` is on; vector ingest mirrors the same namespace when `LEXI_VECTOR_ENABLED` is set.

## Flow wiring (when enabled)
- Requests with `X-Lexi-User` set:
  - `session_middleware` tags `request.state.user_id` and touches `last_seen`.
  - Persona binding (`LexiPersona.set_user`) swaps memory + session paths into the per-user bucket and tags vector ingest.
  - Avatar generation stores first/latest into the manifest and reuses the latest as img2img source when possible.
  - Memory writes push shards/summaries into Chroma with `user_id` metadata.

## Safe defaults
- All helpers short‑circuit when flags are off or IDs are missing.
- Vector ingest returns quietly if Chroma or the embed model is unavailable.
- Avatar manifest ignores HTTP URLs as img2img sources; prefers local/static files.

## Turning it on (staging)
1. Set `LEXI_USER_ID_ENABLED=1 LEXI_USER_DATA_ENABLED=1 LEXI_VECTOR_ENABLED=1` in the backend env.
2. Ensure `LEXI_VECTOR_CHROMA_PATH` points to a writable path; first run will create the store.
3. (Optional) Set `VITE_USER_ID_ENABLED=1` to prompt the frontend for email/name so the header is sent automatically.
4. Exercise persona + chat; check `memory/users/<id>/` for profile/manifest files and Chroma for ingested vectors.
