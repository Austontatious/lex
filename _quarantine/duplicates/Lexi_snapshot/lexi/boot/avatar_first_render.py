from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/persona", tags=["persona"])

# ---- TRAIT SETUP ----
BASE_TRAITS = {
    "apparent_age": ["young adult", "twenties", "thirties", "forties"],
    "body_type": ["slim", "average", "athletic", "curvy", "stocky"],
    "hair_color": ["blonde", "brown", "black", "red", "auburn", "dyed", "grey"],
    "eye_color": ["blue", "green", "brown", "hazel", "gray"],
    "ethnicity": ["white", "black", "asian", "hispanic", "mixed", "middle eastern", "indian", "indigenous", "other"],
}

FIRST_PERSON_QUESTIONS = {
    "apparent_age":   "How old do you imagine me looking? ('surprise me' works too!)",
    "body_type":      "What kind of body type do you picture for me?",
    "hair_color":     "What color is my hair? Any style in mind?",
    "eye_color":      "What about my eyes?",
    "ethnicity":      "And what’s my complexion or background?",
}

VLLM_URL = os.getenv("VLLM_URL")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Lexi")

# ---- UTILITIES ----

def random_trait(trait):
    return random.choice(BASE_TRAITS[trait])

def trait_llm_prompt(trait, user_input):
    opts = BASE_TRAITS[trait]
    return f"""
You are Lexi, an AI companion preparing her look for a first date. The user answered:
"{user_input}"
for the trait '{trait}'.

Pick the closest value from this list: {', '.join(opts)}.
If unsure, pick the most plausible. Output just the value.
"""

def _deterministic_choice(trait: str, user_input: str) -> str:
    opts = [value.lower() for value in BASE_TRAITS[trait]]
    normalized = user_input.lower()
    for option in opts:
        if option in normalized:
            return option
    return opts[0]


async def extract_trait_with_llm(llm_func, trait, user_input):
    """
    Uses the LLM to select the trait option closest to user_input.
    Falls back to deterministic parsing.
    """
    if user_input.strip().lower() in {"surprise me", "idk", "random", "any", "whatever", ""}:
        return random_trait(trait)
    prompt = trait_llm_prompt(trait, user_input)
    resp = llm_func(prompt)
    if asyncio.iscoroutine(resp):
        resp = await resp
    # Normalize and sanitize
    resp = (resp or "").strip().lower()
    options = [x.lower() for x in BASE_TRAITS[trait]]
    if resp in options:
        return resp
    for opt in options:
        if opt in resp:
            return opt
    return _deterministic_choice(trait, user_input)


async def _call_trait_llm(trait: str, user_input: str, prompt: str) -> str | None:
    if not VLLM_URL:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                VLLM_URL,
                json={
                    "model": VLLM_MODEL,
                    "messages": [
                        {"role": "system", "content": "Select the closest option and reply with just the value."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 16,
                },
            )
            response.raise_for_status()
            content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return content.strip()
    except Exception as exc:
        logger.warning("Trait LLM fallback (%s): %s", trait, exc)
        return None

# ---- MAIN ENDPOINT ----

@router.post("/avatar_bootstrap")
async def avatar_bootstrap(request: Request) -> Dict[str, Any]:
    """
    Stepwise, LLM-extracted trait onboarding for Lexi's avatar.
    Frontend POSTs:
      {
        "trait": "hair_color",         # trait being answered (or null/None/"" at first step)
        "answer": "dark brown, wavy",  # user's answer (or "" if first step)
        "traits": { ... }              # dict of previous traits so far
      }

    At any step, user may answer "surprise me", "skip", etc.
    """
    data = await request.json()
    trait = data.get("trait")
    answer = data.get("answer")
    prev_traits = data.get("traits", {}) or {}

    # Option: full random (if first prompt is "surprise me" or similar)
    if answer and answer.strip().lower() in {"surprise me", "random", "whatever", "idk"} and not trait:
        traits = {t: random_trait(t) for t in BASE_TRAITS}
        return {
            "status": "final",
            "traits": traits,
            "message": "Surprise! I’m going to get ready—be right back.",
        }

    # Step: extract current trait (if provided)
    if trait and answer:
        async def llm_func(prompt: str):
            content = await _call_trait_llm(trait, answer, prompt)
            return content or _deterministic_choice(trait, answer)

        extracted = await extract_trait_with_llm(llm_func, trait, answer)
        prev_traits[trait] = extracted

    # Find next trait to ask for
    for t in BASE_TRAITS:
        if t not in prev_traits:
            return {
                "status": "ask",
                "trait": t,
                "prompt": FIRST_PERSON_QUESTIONS[t],
                "traits": prev_traits,
                "opening_line": (
                    "Before we get started, want to help me get ready for our first date? "
                    "If you’d rather I surprise you, just say so—I can dream up my look!"
                ) if not prev_traits else None,
            }
    # All done!
    # Optionally, convert all trait values to canonical format (titlecase, etc)
    traits = {k: v.title() for k, v in prev_traits.items()}
    # Compose natural language SD prompt for first render
    summary = (
        f"I look like a {traits['apparent_age']} woman with a {traits['body_type']} build, "
        f"{traits['hair_color']} hair, {traits['eye_color']} eyes, "
        f"and a {traits['ethnicity']} background."
    )
    return {
        "status": "final",
        "traits": traits,
        "sd_prompt": summary,
        "message": "Got it! I’m going to get ready—be right back.",
    }
