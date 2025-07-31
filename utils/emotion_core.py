from typing import Dict

def infer_emotion(prompt: str) -> Dict[str, float]:
    """
    Returns a dictionary like {'sadness': 0.7, 'anger': 0.2, 'joy': 0.1}
    """
    # Stub: start with keyword & tone patterning
    score = {
        'sadness': 0.0,
        'anger': 0.0,
        'joy': 0.0,
        'love': 0.0,
        'frustration': 0.0,
        'neutral': 0.0
    }

    # Quick keywords (expand as needed)
    if "fuck" in prompt and "love" not in prompt:
        score['anger'] += 0.5
    if "why even try" in prompt or "pointless" in prompt:
        score['sadness'] += 0.8
    if "i'm fine" in prompt.lower():
        score['sadness'] += 0.4
        score['frustration'] += 0.4
    if "lol" in prompt.lower() or "lmao" in prompt.lower():
        score['joy'] += 0.3
    if "whatever" in prompt.lower():
        score['frustration'] += 0.3
    if "thank you" in prompt.lower() or "i appreciate":
        score['love'] += 0.5

    # Normalize:
    total = sum(score.values())
    if total == 0:
        score['neutral'] = 1.0
    else:
        for k in score:
            score[k] /= total

    return score

