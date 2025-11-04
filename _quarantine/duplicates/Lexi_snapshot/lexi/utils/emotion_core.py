import re
from typing import Dict, Pattern, Callable, Optional

# Define the list of emotions we track
EMOTIONS = [
    'sadness',
    'anger',
    'joy',
    'love',
    'frustration',
    'neutral',
]

# A pattern entry consists of a compiled regex, the emotion it affects,
# the weight to add when matched, and an optional condition function
PatternEntry = tuple[Pattern[str], str, float, Optional[Callable[[str], bool]]]

PATTERN_ENTRIES: list[PatternEntry] = [
    # If the prompt contains "fuck" (case-insensitive) and does not contain "love"
    (re.compile(r"\bfuck\b", re.IGNORECASE), 'anger', 0.5, lambda text: 'love' not in text.lower()),
    # Expressions of hopelessness
    (re.compile(r"\bwhy even try\b|\bpointless\b", re.IGNORECASE), 'sadness', 0.8, None),
    # "I'm fine" may signal hidden sadness or frustration
    (re.compile(r"\bi'm fine\b", re.IGNORECASE), 'sadness', 0.4, None),
    (re.compile(r"\bi'm fine\b", re.IGNORECASE), 'frustration', 0.4, None),
    # Laughter indicators
    (re.compile(r"\b(?:lol|lmao)\b", re.IGNORECASE), 'joy', 0.3, None),
    # Dismissive tone
    (re.compile(r"\bwhatever\b", re.IGNORECASE), 'frustration', 0.3, None),
    # Gratitude or affection
    (re.compile(r"\bthank you\b|\bi appreciate\b", re.IGNORECASE), 'love', 0.5, None),
]

def infer_emotion(prompt: str) -> Dict[str, float]:
    """
    Infer a distribution of emotions from the given prompt.

    Uses pattern-based heuristics to score emotions: sadness, anger,
    joy, love, frustration, and neutral. Scores are normalized to sum to 1.0.
    """
    # Initialize all scores to zero
    scores: Dict[str, float] = {emotion: 0.0 for emotion in EMOTIONS}

    # Apply each pattern rule
    for pattern, emotion, weight, condition in PATTERN_ENTRIES:
        if pattern.search(prompt) and (condition is None or condition(prompt)):
            scores[emotion] += weight

    # Normalize scores so they sum to 1.0; default to neutral if no matches
    total = sum(scores.values())
    if total == 0.0:
        scores['neutral'] = 1.0
    else:
        for emotion in scores:
            scores[emotion] /= total

    return scores

