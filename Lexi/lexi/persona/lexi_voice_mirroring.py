"""
Lexi Voice Mirroring Kernel

Drop-in module that:
  1) Extracts a lightweight per-user linguistic profile from recent messages.
  2) Maintains an exponentially-decayed moving average profile across turns.
  3) Produces a compact style control prompt + post-processing rules to steer generation.

No external deps. Pure Python + regex.

Integration sketch (persona_core.py):
  from lexi_voice_mirroring import StyleMemory, analyze_user_style, make_style_directives, apply_postprocessing

  style_mem = StyleMemory(store=your_memory_store)  # instantiate somewhere globally

  def chat(self, user_text: str, ...):
      # 1) update profile from last N user turns
      recent_user_utts = self.memory.get_recent_user_utts(k=12)
      profile = analyze_user_style(recent_user_utts, prev_profile=style_mem.get(self.user_id))
      style_mem.set(self.user_id, profile)

      # 2) build style directives for the LLM
      directives = make_style_directives(profile, persona_traits=self.traits)

      # 3) call your generator with the directives prepended/merged into your system prompt
      sys_prompt = base_sys_prompt + "\n" + directives
      raw = llm.generate(sys_prompt=sys_prompt, messages=messages, **sampler_from_profile(profile))

      # 4) post-process to mirror punctuation/case tics without overdoing it
      final = apply_postprocessing(raw, profile)
      return final

"""

from __future__ import annotations
import math
import re
import statistics
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import random

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF]")
_URL_RE = re.compile(r"https?://\S+")
_CODE_RE = re.compile(r"`{1,3}.*?`{1,3}", re.S)

SLANG = {
    "kinda",
    "sorta",
    "gonna",
    "wanna",
    "tho",
    "tho.",
    "imo",
    "irl",
    "btw",
    "lol",
    "lmao",
    "idk",
    "ikr",
    "brb",
    "af",
    "tbh",
    "omg",
}
HEDGES = {"maybe", "perhaps", "seems", "appears", "somewhat", "a bit", "sort of", "kind of"}
MODALS = {"can", "could", "may", "might", "should", "would", "must", "shall", "will"}
INTENSIFIERS = {"very", "really", "so", "extremely", "super", "totally", "literally"}
PROFANITY = {"fuck", "shit", "damn", "ass", "bitch", "bastard"}
FORMALISM = {"therefore", "however", "moreover", "consequently", "hence", "thus"}
AFFECTION = {"love", "dear", "sweet", "babe", "honey", "cutie"}
CARE_WORDS = {"let me", "i've got you", "i hear you", "with you", "we'll"}


@dataclass
class StyleProfile:
    avg_sentence_len: float = 12.0  # words per sentence
    clause_density: float = 0.25  # commas/semicolons per sentence
    question_ratio: float = 0.15
    exclam_ratio: float = 0.05
    ellipses_ratio: float = 0.05
    emoji_rate: float = 0.02  # emojis per token
    allcaps_rate: float = 0.005
    lowercase_pref: float = 0.1  # leading lowercase lines / total
    slang_density: float = 0.05
    hedge_density: float = 0.05
    modal_density: float = 0.10
    intensifier_density: float = 0.05
    profanity_density: float = 0.01
    formality_score: float = 0.4
    type_token_ratio: float = 0.5

    # Emotional axes (0..1)
    energy: float = 0.5
    warmth: float = 0.5
    chaos: float = 0.2

    # Punctuation habits
    commas_per_100w: float = 5.0
    dashes_per_100w: float = 2.0

    # Derived reading ease (rough FK proxy)
    reading_grade: float = 9.0

    def to_dict(self) -> Dict:
        return asdict(self)


# ----------------------------- Utilities -----------------------------


def _strip_noise(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _CODE_RE.sub("", text)
    return text


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


# --------------------------- Core Extraction ---------------------------


def analyze_user_style(
    user_utts: List[str],
    prev_profile: Optional[Dict] = None,
    decay: float = 0.8,
) -> Dict:
    """Build/update a StyleProfile from recent user utterances.
    - decay: EMA factor for smoothing with previous profile (if present).
    Returns a plain dict for easy JSON persistence.
    """
    joined = _strip_noise("\n".join(user_utts[-12:]))
    if not joined:
        base = StyleProfile()
        return base.to_dict()

    sents = _sentences(joined)
    words = _words(joined)
    n_words = max(1, len(words))

    avg_sentence_len = (
        statistics.mean([max(1, len(_words(s))) for s in sents]) if sents else len(words)
    )
    clause_density = sum(s.count(",") + s.count(";") for s in sents) / max(1, len(sents))
    question_ratio = sum(1 for s in sents if s.strip().endswith("?")) / max(1, len(sents))
    exclam_ratio = sum(1 for s in sents if s.strip().endswith("!")) / max(1, len(sents))
    ellipses_ratio = joined.count("...") / max(1, len(sents))
    emoji_rate = len(_EMOJI_RE.findall(joined)) / n_words

    lines = [l for l in joined.splitlines() if l.strip()]
    lowercase_pref = sum(1 for l in lines if l and l[0].islower()) / max(1, len(lines))
    allcaps_rate = sum(1 for w in words if len(w) > 2 and w.isupper()) / n_words

    # Lexical densities
    def dens(set_words: set) -> float:
        return sum(1 for w in words if w in set_words) / n_words

    slang_density = dens(SLANG)
    hedge_density = dens(HEDGES)
    modal_density = dens(MODALS)
    intensifier_density = dens(INTENSIFIERS)
    profanity_density = dens(PROFANITY)
    formality_score = min(1.0, dens(FORMALISM) * 8.0 + 0.2 * (1.0 - slang_density))

    # Punct tics
    commas = joined.count(",")
    dashes = joined.count("—") + joined.count("--")
    commas_per_100w = 100 * commas / n_words
    dashes_per_100w = 100 * dashes / n_words

    # TTR (vocab breadth)
    type_token_ratio = len(set(words)) / n_words

    # Rough reading grade (FK proxy): sentences per 100w & syllables proxy via word length
    avg_words_per_sentence = n_words / max(1, len(sents))
    avg_chars_per_word = statistics.mean([len(w) for w in words]) if words else 4.5
    reading_grade = max(
        1.0, 0.39 * avg_words_per_sentence + 11.8 * (avg_chars_per_word / 5.0) - 15.59
    )

    # Emotional axes from markers
    energy = min(1.0, 0.2 + 0.9 * (exclam_ratio + intensifier_density * 2 + emoji_rate * 4))
    warmth = min(1.0, 0.4 + 0.6 * (dens(AFFECTION) * 4 + lowercase_pref * 0.1))
    chaos = max(
        0.0, min(1.0, profanity_density * 2 + dashes_per_100w * 0.02 + ellipses_ratio * 0.5)
    )

    profile = StyleProfile(
        avg_sentence_len=avg_sentence_len,
        clause_density=clause_density,
        question_ratio=question_ratio,
        exclam_ratio=exclam_ratio,
        ellipses_ratio=ellipses_ratio,
        emoji_rate=emoji_rate,
        allcaps_rate=allcaps_rate,
        lowercase_pref=lowercase_pref,
        slang_density=slang_density,
        hedge_density=hedge_density,
        modal_density=modal_density,
        intensifier_density=intensifier_density,
        profanity_density=profanity_density,
        formality_score=formality_score,
        type_token_ratio=type_token_ratio,
        commas_per_100w=commas_per_100w,
        dashes_per_100w=dashes_per_100w,
        reading_grade=reading_grade,
        energy=energy,
        warmth=warmth,
        chaos=chaos,
    ).to_dict()

    if prev_profile:
        # Exponential moving average smoothing
        smoothed = {}
        for k, v in profile.items():
            pv = prev_profile.get(k, v)
            smoothed[k] = decay * pv + (1 - decay) * v
        return smoothed
    return profile


# ------------------------ Prompt Construction ------------------------


def _bucket(x: float, edges: Tuple[float, float, float]) -> str:
    a, b, c = edges
    if x <= a:
        return "low"
    if x <= b:
        return "medium"
    return "high"


_DEF_DIRECTIVE_TEMPLATE = (
    "You are Lexi. Mirror the user's linguistic style subtly and supportively.\n"
    "Guidelines (do not restate to the user):\n"
    "- Sentences: target {sent_len_desc} length; clause density {clause_desc}.\n"
    "- Tone: energy {energy_b}, warmth {warmth_b}, chaos {chaos_b}.\n"
    "- Vocabulary: formality {formality_b}; slang {slang_b}; modal/hedge usage {modals_b}.\n"
    "- Punctuation: lightly mirror user's tics (ellipses {ell_b}, exclamations {ex_b}, emojis {emoji_b}).\n"
    "- Formatting: respect lowercase tendency={lower_b}, avoid overuse of ALL CAPS.\n"
    "- Keep empathy authentic; prioritize clarity if style conflicts with safety.\n"
)


def make_style_directives(profile: Dict, persona_traits: Optional[Dict] = None) -> str:
    p = profile
    sent_len_desc = (
        "short" if p["avg_sentence_len"] < 9 else "medium" if p["avg_sentence_len"] < 17 else "long"
    )
    clause_desc = _bucket(p["clause_density"], (0.15, 0.35, 0.6))

    energy_b = _bucket(p["energy"], (0.35, 0.6, 0.8))
    warmth_b = _bucket(p["warmth"], (0.35, 0.6, 0.8))
    chaos_b = _bucket(p["chaos"], (0.15, 0.35, 0.6))

    formality_b = _bucket(p["formality_score"], (0.35, 0.6, 0.8))
    slang_b = _bucket(p["slang_density"], (0.03, 0.07, 0.12))
    modals_b = _bucket(p["modal_density"] + p["hedge_density"], (0.08, 0.16, 0.3))

    ell_b = _bucket(p["ellipses_ratio"], (0.02, 0.08, 0.15))
    ex_b = _bucket(p["exclam_ratio"], (0.02, 0.07, 0.12))
    emoji_b = _bucket(p["emoji_rate"], (0.005, 0.015, 0.035))
    lower_b = (
        "high" if p["lowercase_pref"] > 0.5 else ("medium" if p["lowercase_pref"] > 0.2 else "low")
    )

    base = _DEF_DIRECTIVE_TEMPLATE.format(
        sent_len_desc=sent_len_desc,
        clause_desc=clause_desc,
        energy_b=energy_b,
        warmth_b=warmth_b,
        chaos_b=chaos_b,
        formality_b=formality_b,
        slang_b=slang_b,
        modals_b=modals_b,
        ell_b=ell_b,
        ex_b=ex_b,
        emoji_b=emoji_b,
        lower_b=lower_b,
    )

    if persona_traits:
        extras = []
        if persona_traits.get("playful"):
            extras.append("Favor playful metaphors when appropriate.")
        if persona_traits.get("analytical"):
            extras.append("Prefer crisp structure and explicit reasoning steps.")
        if persona_traits.get("romance"):
            extras.append("Weave gentle affection into responses when invited.")
        if extras:
            base += "- Persona blend: " + " ".join(extras) + "\n"
    return base


# ------------------------ Sampler Suggestions ------------------------


def sampler_from_profile(p: Dict) -> Dict:
    """Map style into decoding params. Adjust to your generator API."""
    # Energy → temperature/top_p, Formality → reduce randomness, Chaos → add slight creativity
    temp = 0.7 + 0.3 * (p["energy"] - 0.5) + 0.1 * (p["chaos"] - 0.2)
    temp = max(0.4, min(1.2, temp))
    top_p = 0.85 + 0.1 * (p["energy"] - 0.5) - 0.1 * (p["formality_score"] - 0.5)
    top_p = max(0.7, min(0.98, top_p))
    max_tokens = 180 if p["avg_sentence_len"] < 10 else (260 if p["avg_sentence_len"] < 18 else 360)
    return {"temperature": round(temp, 2), "top_p": round(top_p, 2), "max_tokens": max_tokens}


# --------------------------- Post-processing ---------------------------


def _maybe_add_emoji(text: str, p: Dict) -> str:
    if p["emoji_rate"] < 0.005:
        return text
    # add a soft emoji at the end with low probability if not present
    if not _EMOJI_RE.search(text) and len(text) < 280 and p["warmth"] > 0.55:
        return text.rstrip() + " " + "🙂"
    return text


def _mirror_punct(text: str, p: Dict) -> str:
    t = text
    # Exclamations: add at most one if user's exclam habit is medium/high and none exists
    if p["exclam_ratio"] > 0.07 and "!" not in t and len(t) < 220:
        t = re.sub(r"([.!?])$", "!", t)
    # Ellipses: if user uses them often, rarely mirror with a trailing beat
    if p["ellipses_ratio"] > 0.12 and not t.endswith("...") and len(t) < 260:
        if t.endswith(".") and random.random() < 0.35:
            t = t[:-1] + "..."
    return t


def _respect_lowercase(text: str, p: Dict) -> str:
    if p["lowercase_pref"] > 0.5:
        # Make salutations and short interjections lowercase while keeping proper nouns
        lines = text.splitlines()
        new_lines = []
        for ln in lines:
            if len(ln) <= 80:
                new_lines.append(ln[:1].lower() + ln[1:])
            else:
                new_lines.append(ln)
        return "\n".join(new_lines)
    return text


def apply_postprocessing(text: str, profile: Dict) -> str:
    t = text.strip()
    t = _mirror_punct(t, profile)
    t = _respect_lowercase(t, profile)
    t = _maybe_add_emoji(t, profile)
    return t


# ------------------------------ Memory ------------------------------


class StyleMemory:
    """Simple in-memory (or pluggable) store for per-user style profiles.
    Replace `store` with your memory system (redis, db, Lexi memory API, etc.).
    """

    def __init__(self, store: Optional[Dict[str, Dict]] = None):
        self._store = store or {}

    def get(self, user_id: str) -> Optional[Dict]:
        return self._store.get(user_id)

    def set(self, user_id: str, profile: Dict) -> None:
        self._store[user_id] = profile


# ------------------------------ Tests ------------------------------

if __name__ == "__main__":
    sample = [
        "lol ok — i'm kinda into this vibe... can we try a softer outfit? maybe pastel?",
        "Also: the last avatar felt too sharp. like, literally pointy? can we tone it down!!",
        "i'm thinking cozy + playful. idk, you tell me :)",
    ]
    prof = analyze_user_style(sample)
    print("PROFILE:", prof)
    directives = make_style_directives(prof, persona_traits={"playful": True})
    print("\nDIRECTIVES:\n", directives)
    print("\nSAMPLER:", sampler_from_profile(prof))
    mock_gen = (
        "Sure — soft pastels, rounded edges, and cozy textures. I’ll make it feel playful and warm."
    )
    print("\nPOST:", apply_postprocessing(mock_gen, prof))
