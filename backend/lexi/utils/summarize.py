# lexi/utils/summarize.py

from typing import Callable, Any


def _coerce_summary(raw: Any) -> str:
    """
    Accept the various shapes our LLM wrappers may return and normalize to string.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        # common shapes: { "text": "..."} or OpenAI-like { "choices": [{"message": {"content": "..."}}]}
        for key in ("text", "summary", "content"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val
        choices = raw.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, str) and choice.strip():
                    return choice
                if isinstance(choice, dict):
                    for key in ("text", "message", "content"):
                        val = choice.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                        if isinstance(val, dict):
                            content = val.get("content")
                            if isinstance(content, str) and content.strip():
                                return content
    if isinstance(raw, (list, tuple)):
        for item in raw:
            coerced = _coerce_summary(item)
            if coerced.strip():
                return coerced
    # Fallback: ensure we always return a string
    return str(raw)


def summarize_pair(llm_generate: Callable[[str], Any], user_msg: str, ai_msg: str) -> str:
    """
    Summarizes a user+AI exchange into one concise, context-preserving sentence.
    llm_generate: function that takes a prompt string and returns a summary string.
    """
    prompt = (
        "Summarize the following conversation turn in ONE short sentence, "
        "keeping all key context and emotional nuance for the next exchange.\n\n"
        f"User: {user_msg}\nAI: {ai_msg}\n\nSummary:"
    )
    summary = llm_generate(prompt)
    normalized = _coerce_summary(summary)
    return normalized.strip()
