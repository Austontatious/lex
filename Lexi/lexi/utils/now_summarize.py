import asyncio

import httpx

from .now_utils import log_now
from ..config.now import ENABLE_NOW, settings_now

SYSTEM = ("You are a concise assistant. Summarize items into 1–2 sentences. "
          "Then produce 2–4 conversational bullet points. Be neutral; avoid clickbait.")

def _msgs(text: str):
    return [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":f"Summarize for small talk, then bullets:\n\n{text}\n"}
    ]

_summarizer_disabled_logged = False


async def summarize_for_smalltalk(text: str):
    global _summarizer_disabled_logged
    if not ENABLE_NOW:
        if not _summarizer_disabled_logged:
            log_now("[summarize] skipped (ENABLE_NOW!=1)")
            _summarizer_disabled_logged = True
        return _fallback(text)

    delay = 0.5
    content: str | None = None
    last_exc: Exception | None = None

    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    str(settings_now.SUMMARIZER_ENDPOINT),
                    json={
                        "model": settings_now.SUMMARIZER_MODEL,
                        "messages": _msgs(text),
                        "temperature": 0.2,
                        "max_tokens": 220,
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                break
        except Exception as exc:
            last_exc = exc
            log_now(f"[summarize] attempt {attempt}/3 failed: {exc}")
            if attempt == 3:
                content = None
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 4.0)

    if content is None:
        return _fallback(text, last_exc)

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    summary = lines[0] if lines else ""
    bullets = [ln.lstrip("-• ").strip() for ln in lines[1:5]]
    return summary, bullets[:4]


def _fallback(text: str, error: Exception | None = None):
    if error:
        log_now(f"[summarize] fallback due to {error}")
    snippet = text[:220] + ("..." if len(text) > 220 else "")
    return snippet, []
