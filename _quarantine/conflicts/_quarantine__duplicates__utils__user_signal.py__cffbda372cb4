# /lex/utils/user_signal.py

"""
user_signal.py

Utility for extracting "surface" emotional cues from user or AI messages.
These features are NOT based on meaning, but on HOW something is said:
    - Misspellings
    - Punctuation (!!! ??? ...)
    - ALL CAPS (shouting/intensity)
    - Cursing
    - Message length
    - [Future: timing between messages, emoji, etc.]

Returns a dict of features for use in persona/emotion axis inference.
"""

import re
from typing import Dict

CURSE_WORDS = [
    "fuck", "shit", "damn", "bitch", "asshole", "bastard", "dick", "crap",
    "piss", "cock", "pussy", "motherfucker", "cunt", "fag", "slut"
]

def extract_surface_cues(text: str) -> Dict[str, float]:
    """
    Analyze text for surface emotional features.

    Returns dict with:
      - cursing: [0,1] (intensity)
      - exclamation: [0,1] (count)
      - question: [0,1]
      - all_caps: [0,1]
      - misspell: [0,1] (rough proxy, see note)
      - msg_length: [0,1] (normalized to 300 chars)
      - trailing_ellipsis: [0,1]
    """
    cues = {}
    lower = text.lower()
    cues['cursing'] = sum(w in lower for w in CURSE_WORDS) / max(1, len(CURSE_WORDS))
    cues['exclamation'] = min(text.count("!"), 6) / 6.0
    cues['question'] = min(text.count("?"), 4) / 4.0
    cues['all_caps'] = sum(1 for w in text.split() if w.isupper() and len(w) > 2) / max(1, len(text.split()))
    # NOTE: Real misspell detection needs spellchecker, but here's a fast proxy
    cues['misspell'] = sum(1 for w in text.split() if len(w) > 3 and not w.islower() and w.isalpha()) / max(1, len(text.split()))
    cues['msg_length'] = min(len(text), 300) / 300.0
    cues['trailing_ellipsis'] = 1.0 if text.strip().endswith("...") else 0.0
    return cues

if __name__ == "__main__":
    while True:
        msg = input("Text> ")
        print(extract_surface_cues(msg))

