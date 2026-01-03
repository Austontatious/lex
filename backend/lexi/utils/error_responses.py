from __future__ import annotations

SOFT_ERROR_MESSAGE = (
    "Something just went wrong on my side and I couldn't finish that reply, "
    "but I'm still here with you. If you feel up to it, can you tell me a little more "
    "about what's going on, or we can just breathe together for a moment."
)


def soft_error_payload(error_detail: str | None = None, trace_id: str | None = None) -> dict:
    payload = {
        "cleaned": SOFT_ERROR_MESSAGE,
        "raw": SOFT_ERROR_MESSAGE,
        "choices": [{"text": SOFT_ERROR_MESSAGE}],
    }
    if error_detail:
        payload["error_detail"] = error_detail
    if trace_id:
        payload["trace_id"] = trace_id
    return payload


__all__ = ["SOFT_ERROR_MESSAGE", "soft_error_payload"]
