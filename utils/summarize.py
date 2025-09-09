# lex/utils/summarize.py

from typing import Callable

def summarize_pair(llm_generate: Callable[[str], str], user_msg: str, ai_msg: str) -> str:
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
    return summary.strip()

