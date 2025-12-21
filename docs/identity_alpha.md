# Lexi Alpha Identity (Device + Handle, No Login)

Lexi Alpha uses a low-friction identity system that binds memory + avatars to a
canonical user id while still letting users pick a friendly handle.

## Canonical identity
- Storage identity is always `user_<uuid>`.
- Routes must use `request.state.user_id` (set by middleware).
- No route should recompute user ids from headers.

## Headers
Send these on every request:
- `X-Lexi-Device`: stable device UUID (stored by the client).
- `X-Lexi-Handle`: free-form handle ("Auston", "Auston H.").
- `X-Lexi-User`: canonical `user_<uuid>` when already known.
- `X-Lexi-Session`: optional session id (cookie-backed) for continuity.

If `X-Lexi-Device` is missing, the server generates one and echoes it as a
response header. Clients must persist it.

## Resolution order
1) Canonical `X-Lexi-User` header.
2) Device binding (`X-Lexi-Device`).
3) Session binding (`X-Lexi-Session`).
4) Handle binding (`X-Lexi-Handle`).
   - 0 candidates: create user + bind device.
   - 1 candidate: bind device.
   - 2+ candidates: return `needs_disambiguation`.
5) Fallback: create a new user (source `anon_temp`).

## Collision UX
When a handle maps to multiple users, the backend flags:
- `needs_disambiguation: true`
- `candidates: [...]`

The UI should render the prompt:
"I already know an Auston — is that you or should I call you something else?"

## Identity endpoints
- `GET /lexi/whoami`
  - Returns `user_id`, `device_id`, `handle_norm`, `source`, and paths.
- `POST /lexi/identity/select`
  - Body: `{ "handle": "Auston", "selected_user_id": "user_...", "merge_others": false }`
  - Binds device to the selected user.
- `POST /lexi/identity/rename`
  - Body: `{ "old_handle": "Auston", "new_handle": "AustonHorras" }`
  - Adds the new handle to the current user.

## Storage paths
- Identity DB: `LEXI_IDENTITY_DB_PATH` or `<LEX_USER_DATA_ROOT>/identity/identity.db`.
- Memory: `<LEXI_MEMORY_ROOT>/users/<user_id>/...`
- User data: `<LEX_USER_DATA_ROOT>/users/<user_id>/...`

## Notes
- Session ids are never used as storage identities.
- Canonical user ids are stable across devices after selection.
- User ids are case-sensitive when mapped to storage; `Auston` and `auston` are different buckets.
