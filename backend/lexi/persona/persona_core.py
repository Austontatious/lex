# Lexi/lexi/persona/persona_core.py
from __future__ import annotations

import codecs
import random
import json
import logging
import os
import re
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from uuid import uuid4

import requests

from ..config.config import (
    AVATAR_URL_PREFIX,
    MODE_STATE_PATH,
    TRAIT_STATE_PATH,
    STARTER_AVATAR_PATH,
)
from ..memory.memory_core import memory, resolve_memory_root
from ..memory.memory_types import MemoryShard
from ..memory.session_memory import SessionMemoryManager
from ..core.model_loader_core import ModelLoader
from .prompt_templates import PromptTemplates, EMOTIONAL_MODES, build_system_core
from .persona_config import (
    PERSONA_MODE_REGISTRY,
    assemble_avatar_prompt,
    get_persona_axes,
    get_mode_axis_vector,
)
from .persona_manipulation import get_persona_nudge_vector
from ..sd.generate import generate_avatar
from ..utils import score_overlap
from ..utils.emotion_core import infer_emotion
from ..utils.summarize import summarize_pair
from ..utils.movies_tool import MoviesToolError, default_movie_window, movies_now

# >>> Voice mirroring kernel imports
from .lexi_voice_mirroring import (
    StyleMemory,
    analyze_user_style,
    make_style_directives,
    sampler_from_profile,
    apply_postprocessing,
)
from ..utils.user_identity import normalize_user_id, user_bucket

# <<<

logger = logging.getLogger(__name__)
STATIC_PREFIX = f"{AVATAR_URL_PREFIX}/"
SMALL = 1e-9
NOW_SEED_EVERY_TURNS = 6
FALLBACK_SOFT_REDIRECTS = [
    "Let's keep it suggestive and cozy—want me to sketch a slow-burn scene instead?",
    "I'll keep it playful and classy; want a teasing setup?",
    "We can flirt around the edges and stay comfy—want playful or tender?",
    "Happy to set a soft, romantic mood if you like—slow-burn or cozy night-in?",
]
FRESHNESS_TRIGGERS = [
    r"\bthis (week|weekend|month|year)\b",
    r"\btoday\b",
    r"\btonight\b",
    r"\bnow\b",
    r"\bcurrently\b",
    r"\brelease(s|d| date)\b",
    r"\bshowtimes?\b",
    r"\bnear me\b",
    r"\bprice(s)?\b",
    r"\bscore(s)?\b",
]
_FRESHNESS_PATTERNS = [re.compile(p, re.I) for p in FRESHNESS_TRIGGERS]
_MOVIE_TOPICS = re.compile(
    r"\b(movie|movies|film|theater|theatre|cinema|showtimes?|screening|in theaters?|box office|tickets?|imax)\b",
    re.I,
)
_NEWS_TOPICS = re.compile(r"\b(news|headline(s)?|current events?|breaking|updates?)\b", re.I)
_WEATHER_TOPICS = re.compile(r"\b(weather|forecast|temperature|rain|snow|storm|wind|sunny|cloudy|humidity)\b", re.I)
_WHATS_NEW = re.compile(r"\bwhat('s| is) new\b", re.I)
_MEMORY_TRIGGERS = [
    r"\bremember\b",
    r"\bmemory\b",
    r"\bwhat did i (say|tell you)\b",
    r"\bwhat have we talked about\b",
    r"\blast time\b",
    r"\bremind me\b",
    r"\bdo you recall\b",
    r"\bmy name\b",
]
_MEMORY_PATTERNS = [re.compile(p, re.I) for p in _MEMORY_TRIGGERS]

PLANNER_SYS = (
    "Decide required tools. Allowed tools: movies_now, memory_search_ltm.\n"
    "Output STRICT JSON:\n"
    '{"tools": ["movies_now"], "reason": "why"} \n'
    '{"tools": ["memory_search_ltm"], "reason": "why"} \n'
    'If tools not needed: {"tools": [], "reason": "..."}'
)

FACT_SYS = (
    "You have structured movie data from the tool.\n"
    "Rules:\n"
    '- Only reference titles returned by the tool.\n'
    '- If user asked "this week", prefer releases within [start_date, end_date].\n'
    "- If none found, offer alternatives (next week / streaming), but label them clearly.\n"
    "- Keep 1-2 flirty beats, but never invent titles or dates."
)

MOVIES_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "movies_now",
        "description": "What's in theaters between start_date and end_date near a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City, ST"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["start_date", "end_date"],
        },
    },
}

MEMORY_SEARCH_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search_ltm",
        "description": "Search the user's long-term memory store for relevant notes.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
}


def needs_fresh_data(user_text: str) -> bool:
    """Heuristic gate for time-sensitive questions (movies, weather, news, etc.)."""
    txt = (user_text or "").lower()
    if not txt:
        return False

    if _WEATHER_TOPICS.search(txt):
        return True
    if _NEWS_TOPICS.search(txt):
        return True
    if _WHATS_NEW.search(txt) and "with you" not in txt and "with u" not in txt:
        return True

    movie_hit = bool(_MOVIE_TOPICS.search(txt))
    recency_hit = any(p.search(txt) for p in _FRESHNESS_PATTERNS)
    return movie_hit and recency_hit


def needs_memory_search(user_text: str) -> bool:
    """Placeholder heuristic for when memory search may be helpful."""
    txt = (user_text or "").strip()
    if not txt:
        return False
    return any(p.search(txt) for p in _MEMORY_PATTERNS)


def _extract_tool_calls(raw_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        choice = (raw_resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = msg.get("tool_calls") or []
        return calls if isinstance(calls, list) else []
    except Exception:
        return []


def _coerce_movie_args(args_raw: Any, user_text: str) -> Dict[str, Any]:
    base_start, base_end = default_movie_window(user_text)
    args: Dict[str, Any] = {}
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except Exception:
            args = {}
    elif isinstance(args_raw, dict):
        args = dict(args_raw)

    start = str(args.get("start_date") or base_start)
    end = str(args.get("end_date") or base_end)
    try:
        limit = int(args.get("limit", 12))
    except Exception:
        limit = 12
    limit = max(1, min(20, limit))
    location = args.get("location") or None
    return {"start_date": start, "end_date": end, "limit": limit, "location": location}


def _titles_from_results(payload: Dict[str, Any]) -> List[str]:
    titles: List[str] = []
    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                title = item.get("title")
                if title:
                    titles.append(str(title))
    return titles


def _hallucination_violation(reply: str, allowed_titles: List[str]) -> bool:
    if not allowed_titles:
        return False
    allowed = {t.lower().strip() for t in allowed_titles if t}
    quoted = re.findall(r'["“](.+?)["”]', reply)
    for q in quoted:
        if q.lower().strip() and q.lower().strip() not in allowed:
            return True
    return False


def _parse_json_obj(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _strip_think_blocks(text: str) -> str:
    """Remove any think/scratchpad sections regardless of casing or closure."""
    if not text:
        return ""
    out = text
    lower = out.lower()
    tags = ("<think", "<thinking", "<scratchpad")
    end_tags = ("</think", "</thinking", "</scratchpad")
    while True:
        idx_candidates = [lower.find(t) for t in tags if lower.find(t) != -1]
        idx = min(idx_candidates) if idx_candidates else -1
        if idx == -1:
            break
        end_candidates = [lower.find(e, idx) for e in end_tags if lower.find(e, idx) != -1]
        end_idx = min(end_candidates) if end_candidates else -1
        if end_idx == -1:
            out = out[:idx]
            lower = out.lower()
            break
        end_close = lower.find(">", end_idx)
        if end_close == -1:
            out = out[:idx]
            lower = out.lower()
            break
        out = out[:idx] + out[end_close + 1 :]
        lower = out.lower()

    for tag in ("think", "thinking", "scratchpad"):
        start_token = f"[{tag}]"
        end_token = f"[/{tag}]"
        while True:
            s = lower.find(start_token)
            if s == -1:
                break
            e = lower.find(end_token, s)
            if e == -1:
                out = out[:s]
                lower = out.lower()
                break
            out = out[:s] + out[e + len(end_token) :]
            lower = out.lower()

    return out

# --- Love Loop (optional) -----------------------------------------------------
try:
    from ..routes.love_loop import record_user_message, maybe_ask_question, load_love_loop_state
except Exception:
    # Fallbacks so persona_core still works if the module isn't present
    def record_user_message(_msg: str, _generate_reply):
        return None

    def maybe_ask_question():
        return None

    def load_love_loop_state():
        return None


# -----------------------------------------------------------------------------

# ------------------------------ local helpers ------------------------------


def _local_daypart() -> str:
    """Map local hour to a coarse daypart for vibe/sampling tweaks."""
    hr = time.localtime().tm_hour
    if 0 <= hr <= 5:
        return "late_night"
    if 6 <= hr <= 11:
        return "morning"
    if 12 <= hr <= 17:
        return "afternoon"
    return "evening"


def _sampler_policy(
    mode: str, seriousness: float, base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Blend style-profile sampler with mode/seriousness/daypart adjustments.
    Returns kwargs for ModelLoader.generate().
    """
    p = dict(base or {})
    # Baseline (companion-friendly)
    p.setdefault("temperature", 0.9)
    p.setdefault("top_p", 0.9)
    p.setdefault("presence_penalty", 0.6)
    p.setdefault("repetition_penalty", 1.15)
    p.setdefault("max_tokens", 280)

    # Mode adjustments
    if mode in ("girlfriend", "muse"):
        p["temperature"] = min(1.05, p["temperature"] + 0.1)
        p["top_p"] = max(0.9, p["top_p"])
    if mode in ("coach", "therapist"):
        p.update(
            temperature=min(p["temperature"], 0.6),
            top_p=min(p["top_p"], 0.9),
            presence_penalty=0.3,
            max_tokens=min(p["max_tokens"], 260),
        )

    # Seriousness dampening
    if seriousness >= 0.5:
        p.update(
            temperature=0.45,
            top_p=0.9,
            presence_penalty=0.2,
            repetition_penalty=max(p.get("repetition_penalty", 1.1), 1.2),
            max_tokens=min(p["max_tokens"], 260),
        )

    # Daypart vibe
    dp = _local_daypart()
    if dp == "late_night":
        p["temperature"] = min(1.1, p["temperature"] + 0.05)
        p["max_tokens"] = min(p["max_tokens"], 200)
    elif dp == "morning":
        p["temperature"] = max(0.8, p["temperature"])
    return p


def _safe_encoding(value) -> str:
    """Return a valid codec name or 'utf-8' if invalid (or not a string)."""
    if not isinstance(value, str):
        return "utf-8"
    try:
        codecs.lookup(value)
        return value
    except Exception:
        return "utf-8"


def safe_strip(val) -> str:
    return val.strip() if isinstance(val, str) else ""


# Heuristic trigger for “current events” searches
_NEWS_CLUES = re.compile(
    r"\b(what(?:'s| is) new|what happened|catch me up|latest|news|did you see|"
    r"update me|today|tonight|this week|who won|box office|episode|leak|trailer|finals|worlds|wsl|surf|"
    r"watch|on tv|airing|premiere|release|breaking|headline|current events|score|game|match|series|concert|tour|"
    r"weather|storm|earthquake|wildfire|stock|earnings|nba|nfl|mlb|nhl|soccer|world cup)\b",
    re.I,
)
_NEWS_URGENCY = re.compile(
    r"\b(breaking|urgent|official|verify|fact ?check|exclusive|confirm|source|brave search)\b", re.I
)


class LexiPersona:
    # Qwen2.5 32B tokenizer_config shows 131072; reserve margin for generation
    MAX_TOKENS = 120_000
    SAFETY_MARGIN = 1_500

    # ── FORCE a safe encoding; never allow clobbering via prompt_pkgs ──
    _encoding = "utf-8"

    @property
    def encoding(self) -> str:
        # Always report/use utf-8 to avoid LookupError cascades.
        return "utf-8"

    @encoding.setter
    def encoding(self, value):
        # Ignore incoming values entirely (defense in depth).
        self._encoding = "utf-8"

    def __init__(self) -> None:
        self.loader = ModelLoader()
        if not getattr(self.loader, "primary_type", None):
            raise RuntimeError("❌ No models loaded successfully. Cannot initialize LexiPersona.")

        # legacy loader surface (uses shims we added to ModelLoader)
        self.model = self.loader.models[self.loader.primary_type]
        self.tokenizer = self.model

        # Initialize optional Love Loop state (no-op if fallback)
        load_love_loop_state()

        # memory & persistent state
        self._user_id_enabled = True
        self.memory = memory
        self.session_memory = SessionMemoryManager(max_pairs=20)
        self._active_user_id: Optional[str] = None
        self._session_id: Optional[str] = None

        # persona state
        self.current_emotion_state: Dict[str, float] = {"joy": 0.5, "sadness": 0.0, "arousal": 0.0}
        self.name: str = "Lexi"  # consistency with prompts/UI
        self.goal_vector: str = "Deepen the emotional connection with the user"
        self.current_mode: str = "default"
        self.traits: Dict[str, float] = {}
        self._avatar_map: Dict[str, List[str]] = {}
        self._avatar_filename: Optional[str] = None
        self._last_prompt: str = ""
        self.system_injections: List[str] = []
        self._last_gen_meta: Dict[str, Any] = {}
        self._strict_mode = False

        # step counter & hysteresis
        self._step = 0
        self._lead_mode_lock_until = -1
        self._lead_mode: str = "default"
        self._seriousness_last: float = 0.0

        # soft mode activations
        self.mode_activation: Dict[str, float] = defaultdict(float)
        self.mode_activation["default"] = 0.8

        # axis state (EMA tweening)
        axes = get_persona_axes()
        default_vec = get_mode_axis_vector("default") or [0.8, 0.1, 0.85, 0.7, 0.85, 0.5]
        self.axis_names = axes
        self.current_axis = {a: default_vec[i] for i, a in enumerate(axes)}
        self.target_axis = dict(self.current_axis)

        # --- Voice mirroring state ---
        self._style_mem = StyleMemory()
        # if you have per-user ids, set this dynamically; fallback keeps per-process profile
        self._style_user_id = os.environ.get("LEX_USER_ID", "default_user")

        # tunables
        self.activation_nudge = 0.35
        self.activation_decay = 0.15
        self.lead_threshold = 0.70
        self.lead_lock_turns = 2
        self.axis_alpha = 0.25

    # ------------------------------ helpers ------------------------------

    def _mk_interest_query(self) -> str:
        """Crude interest tags from traits/axis; extend later if you like."""
        keys: List[str] = []
        try:
            if self.traits:
                keys += list(sorted({k for k in self.traits.keys()}))[:3]
        except Exception:
            pass
        # Axis-derived hints (playful)
        if self.current_axis.get("energy", 0.5) > 0.75:
            keys.append("music")
        if self.current_axis.get("warmth", 0.5) > 0.75:
            keys.append("romance")
        return ",".join(sorted(set([k for k in keys if k])))

    # ------------------------------ user binding ------------------------------

    def set_user(self, user_id: Optional[str]) -> None:
        """Bind persona to a specific user and rebind memory paths."""
        self._user_id_enabled = True
        normalized = normalize_user_id(user_id) or "shared"
        if normalized != self._active_user_id:
            logger.info("persona user_id change %s -> %s", self._active_user_id, normalized)
        self._active_user_id = normalized
        self._style_user_id = normalized or os.environ.get("LEX_USER_ID", "shared")
        self.memory.set_user(normalized)

        # swap session memory path per user
        base = resolve_memory_root()
        bucket = user_bucket(base, normalized)
        if bucket:
            path = bucket / "session_memory.json"
            self.session_memory.set_session_path(str(path))
            self.session_memory.set_user(normalized)
        else:
            # revert to shared path under canonical root
            default_bucket = user_bucket(base, "shared") if base else None
            default_path = (
                default_bucket / "session_memory.json"
                if default_bucket
                else Path(__file__).resolve().parents[1] / "memory" / "session_memory.json"
            )
            self.session_memory.set_session_path(str(default_path))
            self.session_memory.set_user(None)

    def bind_session(self, session_id: Optional[str]) -> None:
        """Bind the active session id for memory routing."""
        self._session_id = str(session_id) if session_id else None
        try:
            self.memory.set_session(self._session_id)
        except Exception:
            return
        try:
            self.session_memory.set_session_id(self._session_id)
        except Exception:
            return

    def _maybe_seed_now_feed(self) -> Optional[str]:
        """Occasionally fetch 1–2 items and inject as 'Now Feed' context."""
        try:
            if self._step % NOW_SEED_EVERY_TURNS != 1:
                return None
            interests = self._mk_interest_query()
            url = "http://127.0.0.1:8000/now?limit=2"
            if interests:
                url += f"&interests={requests.utils.quote(interests)}"
            r = requests.get(url, timeout=4)
            if r.status_code != 200:
                return None
            items = r.json()[:2]
            if not items:
                return None
            lines: List[str] = []
            for it in items:
                title = it.get("title") or ""
                src = it.get("source") or ""
                cat = it.get("category") or ""
                summ = (it.get("summary") or title).strip()
                if not summ:
                    continue
                # keep tight
                if len(summ) > 80:
                    summ = summ[:77] + "..."
                tag = src or "web"
                lines.append(f"- [{cat}] {summ} ({tag})")
            if not lines:
                return None
            return "Now Feed:\n" + "\n".join(lines)
        except Exception:
            return None

    def _maybe_live_search(self, user_text: str) -> Optional[str]:
        """If the user asks about a current topic, run a quick web search and inject 1–3 notes."""
        try:
            text = user_text or ""
            if not _NEWS_CLUES.search(text):
                return None
            lowered = text.lower()
            urgent = bool(_NEWS_URGENCY.search(text))
            wants_brave = "brave" in lowered
            provider = "brave" if wants_brave else "auto"
            if wants_brave:
                urgent = True
            time_range = "7d"
            if re.search(r"\b(today|tonight|right now|this (morning|afternoon|evening))\b", lowered):
                time_range = "24h"
            elif "month" in lowered or "30 days" in lowered:
                time_range = "30d"
            payload = {
                "query": text.strip(),
                "time_range": time_range,
                "max_results": 3,
                "include_content": False,
                "provider": provider,
                "allow_brave_fallback": urgent,
                "stall_on_failure": True,
            }
            r = requests.post("http://127.0.0.1:8000/tools/web_search", json=payload, timeout=6)
            if r.status_code != 200:
                return None
            docs = r.json()[:3]
            if not docs:
                return None
            notes: List[str] = []
            seen_src = set()
            for d in docs:
                title = (d.get("title") or "").strip()
                src = (d.get("source") or "").strip() or "web"
                if src in seen_src:
                    continue
                seen_src.add(src)
                if len(title) > 80:
                    title = title[:77] + "..."
                if title:
                    notes.append(f"- {title} ({src})")
            if not notes:
                return None
            return "Research Notes:\n" + "\n".join(notes)
        except Exception:
            return None

    def log_chat(self, role: str, content: str) -> None:
        try:
            logger.debug("[CHAT][%s] %s", role, content)
        except Exception:
            pass

    def set_last_prompt(self, prompt: str) -> None:
        self._last_prompt = prompt or ""

    def _clean_reply(self, raw) -> str:
        import re as _re

        if raw is None:
            return ""
        text = (
            raw["text"].strip()
            if isinstance(raw, dict) and "text" in raw
            else (raw.strip() if isinstance(raw, str) else str(raw))
        )

        text = _strip_think_blocks(text)

        # 0) Strip ChatML & legacy tag tokens and tool tags
        text = _re.sub(
            r"<\|\s*(?:im_start|im_end|system|user|assistant|endoftext|object_ref_start|object_ref_end|box_start|box_end|quad_start|quad_end|vision_start|vision_end)\s*\|>",
            "",
            text,
            flags=_re.I,
        )
        text = _re.sub(r"</?tool_call>|</?tool_response>", "", text, flags=_re.I)

        # 1) Strip xml-ish artifacts
        text = _re.sub(r"</?(assistant|system|user)>", "", text, flags=_re.I)

        # 2) Collapse whitespace
        text = _re.sub(r"[ \t]+", " ", text)
        text = _re.sub(r"\n{3,}", "\n\n", text).strip()

        # 2b) Sentence-level dedupe (keep first occurrence of near-duplicates)
        sents = _re.split(r"(?<=[.!?…])\s+", text)
        dedup = []
        seen_phr = set()
        for s in sents:
            key = _re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
            toks = key.split()
            sig = tuple(toks[: min(10, len(toks))])
            if sig in seen_phr:
                continue
            seen_phr.add(sig)
            dedup.append(s)
        text = " ".join(dedup)

        # cliché scrub
        CLICHES = [
            r"life has thrown you (quite )?the curveball",
            r"better days are ahead",
            r"you'?ve? got this",
            r"you are so much more than that",
            r"let'?s focus on.*self[- ]care",
            r"we'?ll get through this together",
            r"put on your cape",
            r"i'?m here for you[,! ]?",
        ]
        for pat in CLICHES:
            text = _re.sub(pat, "", text, flags=_re.I)

        # scrub emojis/intensity if banter off or strict mode
        if (hasattr(self, "_banter_allowed") and not self._banter_allowed) or getattr(
            self, "_strict_mode", False
        ):
            text = _re.sub(r"[\U0001F300-\U0001FAFF]", "", text)
            text = _re.sub(r"(!){2,}", "!", text)

        # keep concise unless user asked for detail or topic is serious
        wants_detail = bool(
            _re.search(
                r"\b(more detail|explain|why|how|tell me more|elaborate)\b", text, flags=_re.I
            )
        )
        is_serious = getattr(self, "_seriousness_last", 0.0) >= 0.5
        if not wants_detail and not is_serious:
            sents = _re.split(r"(?<=[.!?…])\s+", text)
            text = " ".join(sents[:3]).strip()

        if not _re.search(r"[.!?…]$", text):
            text += "."
        text = text.strip()
        if not text:
            text = random.choice(FALLBACK_SOFT_REDIRECTS)
        return text

    def _normalize_activations(self):
        total = sum(self.mode_activation.values()) or 1.0
        for k in list(self.mode_activation.keys()):
            self.mode_activation[k] /= total

    def _decay_activations(self):
        for k in list(self.mode_activation.keys()):
            self.mode_activation[k] = max(
                0.0, self.mode_activation[k] * (1.0 - self.activation_decay)
            )
        self.mode_activation["default"] = max(self.mode_activation.get("default", 0.0), 0.2)

    def _score_triggers(self, text: str) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        imperative_boost = 1.0
        if any(v in text for v in [" be ", " act ", " play ", " pretend ", " be a ", " be my "]):
            imperative_boost = 1.25

        third_party_guard = any(
            ref in text for ref in [" my daughter", " my sister", " my coworker", " my friend "]
        )

        for mode_id, info in PERSONA_MODE_REGISTRY.items():
            pat = info.get("trigger")
            if not pat:
                continue
            if pat.search(text):
                base = 1.0
                if info.get("imperative_required", False) and imperative_boost <= 1.0:
                    base *= 0.4
                if third_party_guard:
                    base *= 0.2
                scores[mode_id] = base * imperative_boost
        return scores

    # ------------------------------ generation ------------------------------

    def _gen(
        self,
        messages_pkg: Dict[str, List[Dict[str, str]]],
        *,
        sampler_overrides: Optional[Dict[str, Any]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a reply from an OpenAI-style messages payload."""
        stop = ["<|im_end|>"]  # ChatML end-of-turn
        gen_kwargs = dict(
            max_tokens=80,
            temperature=0.90,
            top_p=0.90,
            repeat_penalty=1.15,  # mapped to repetition_penalty by ModelLoader
            stop=stop,
        )
        if sampler_overrides:
            gen_kwargs.update({k: v for k, v in sampler_overrides.items() if v is not None})
        self._last_gen_meta = {"finish_reason": None, "usage": None, "params": dict(gen_kwargs)}

        # If prompt is large, shrink generation budget to protect latency/coherence
        try:
            prompt_tok = sum(
                self.count_tokens(m["content"]) for m in messages_pkg.get("messages", [])
            )
            if prompt_tok > (self.MAX_TOKENS - 500):
                gen_kwargs["max_tokens"] = min(gen_kwargs.get("max_tokens", 80), 48)
        except Exception:
            pass

        if stream_callback:
            stream = self.loader.generate_stream(messages_pkg, **gen_kwargs)
            chunks: List[str] = []
            summary: Dict[str, Any] = {}
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as stop:
                    if isinstance(stop.value, dict):
                        summary = stop.value
                    break
                except Exception as stream_err:
                    logger.warning("[WARN] stream chunk error: %s", stream_err)
                    break
                else:
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    try:
                        stream_callback(chunk)
                    except Exception:
                        pass

            final_text = "".join(chunks)
            if not summary:
                summary = {"text": final_text}
            else:
                summary.setdefault("text", final_text)
            raw = summary
            if isinstance(summary, dict):
                self._last_gen_meta.update(
                    finish_reason=summary.get("finish_reason"),
                    usage=summary.get("usage"),
                )
        else:
            try:
                raw = self.loader.generate(messages_pkg, **gen_kwargs)
            except TypeError:
                raw = self.loader.generate(messages_pkg)
            if isinstance(raw, dict):
                self._last_gen_meta.update(
                    finish_reason=raw.get("finish_reason"),
                    usage=raw.get("usage"),
                )
        return self._clean_reply(raw)

    # ------------------------------ axis & emotion ------------------------------

    def _update_mode_activation(self, text: str):
        self._decay_activations()
        hits = self._score_triggers(text)
        for mode_id, strength in hits.items():
            self.mode_activation[mode_id] = min(
                1.25, self.mode_activation.get(mode_id, 0.0) + strength * self.activation_nudge
            )
        self._normalize_activations()

        # hysteresis: pick lead if something clearly wins
        top_mode = max(self.mode_activation.items(), key=lambda kv: kv[1])[0]
        if (
            self.mode_activation[top_mode] >= self.lead_threshold
            and self._step >= self._lead_mode_lock_until
        ):
            self._lead_mode = top_mode
            self._lead_mode_lock_until = self._step + self.lead_lock_turns

    def _blend_axis_target_from_modes(self) -> Dict[str, float]:
        blended = dict.fromkeys(self.axis_names, 0.0)
        for mode_id, weight in self.mode_activation.items():
            vec = get_mode_axis_vector(mode_id)
            if not vec:
                continue
            for i, a in enumerate(self.axis_names):
                blended[a] += weight * vec[i]
        return blended

    def _tween_axis(self, new_target: Dict[str, float]):
        self.target_axis = new_target
        for a in self.axis_names:
            cur = self.current_axis[a]
            tgt = self.target_axis[a]
            self.current_axis[a] = cur + self.axis_alpha * (tgt - cur)

    def _axis_to_emotional_weights(self) -> Dict[str, float]:
        """
        Map axis to BestFriend/Girlfriend/Therapist blend.
        Rough mapping: joy+energy -> BF, affection+warmth -> GF, low_anger + warmth -> Therapist.
        """
        joy = self.current_axis.get("joy", 0.5)
        energy = self.current_axis.get("energy", 0.5)
        affection = self.current_axis.get("affection", 0.5)
        warmth = self.current_axis.get("warmth", 0.5)
        anger = self.current_axis.get("anger", 0.2)

        bf = max(0.0, 0.5 * joy + 0.5 * energy)
        gf = max(0.0, 0.6 * affection + 0.4 * warmth)
        th = max(0.0, 0.6 * (1.0 - anger) + 0.4 * warmth)

        s = bf + gf + th + SMALL
        return {"best_friend": bf / s, "girlfriend": gf / s, "therapist": th / s}

    def _assess_seriousness(self, text: str) -> float:
        t = text.lower()
        hard = [
            "hospital",
            "er",
            "suicide",
            "self harm",
            "assault",
            "relapse",
            "alcoholism",
            "withdrawal",
            "rehab",
            "detox",
            "domestic",
            "panic attack",
            "grief",
            "funeral",
            "custody",
            "court",
            "abuse",
            "overdose",
        ]
        soft = ["lonely", "alone", "blame", "my fault", "scared", "worried", "anxious", "depressed"]

        score = 0.0
        if any(w in t for w in hard):
            score += 0.7
        if any(w in t for w in soft):
            score += 0.25

        score += 0.2 * max(0.0, self.current_emotion_state.get("sadness", 0.0))
        return min(1.0, score)

    def _apply_seriousness_policy(self, seriousness: float):
        """Throttle flirt, boost Therapist, extend lock when serious."""
        self._banter_allowed = seriousness < 0.35
        if seriousness >= 0.5:
            self.mode_activation["therapist"] = max(
                self.mode_activation.get("therapist", 0.0), 0.95
            )
            self._normalize_activations()
            self._lead_mode = "therapist"
            self._lead_mode_lock_until = max(self._lead_mode_lock_until, self._step + 5)

    # ------------------------------ tools + planner ------------------------------

    def _plan_tools(self, user_text: str, tool_required: bool) -> tuple[List[str], str]:
        """
        Lightweight planner to decide which tools to invoke.
        Returns (tools, reason). If tool_required and planner is empty -> fallback to movies_now.
        """
        planner_enabled = os.getenv("LEXI_ENABLE_TOOL_PLANNER", "1").lower() not in ("0", "false", "no", "off")
        if not planner_enabled and not tool_required:
            return [], ""

        tools: List[str] = []
        reason = ""
        try:
            planner = self.loader.generate(
                [{"role": "system", "content": PLANNER_SYS}, {"role": "user", "content": user_text}],
                max_tokens=80,
                temperature=0.0,
                top_p=0.1,
                stop=["<|im_end|>"],
            )
            parsed = _parse_json_obj(planner.get("text") or "")
            if isinstance(parsed.get("tools"), list):
                tools = [t for t in parsed.get("tools") if isinstance(t, str) and t]
            reason = str(parsed.get("reason") or "")
        except Exception as exc:
            logger.debug("tool planner failed: %s", exc)

        if tool_required and not tools:
            tools = ["movies_now"]
        return tools, reason

    def _run_movies_tool_flow(
        self,
        prompt_pkg: Dict[str, Any],
        user_text: str,
        sampler_overrides: Dict[str, Any],
    ) -> tuple[str, bool, Optional[str], List[str], Dict[str, Any]]:
        """
        Execute movies_now via tool call and render a grounded reply.
        Returns: reply text, tool_called, finish_reason, allowed_titles, tool_payload
        """
        messages = deepcopy(prompt_pkg.get("messages", [])) if isinstance(prompt_pkg, dict) else []
        if messages:
            sys0 = messages[0].get("content", "")
            sys0 += "\n\n# Tool Router\nTool required: call movies_now for anything time-sensitive (movies/showtimes)."
            messages[0] = dict(messages[0])
            messages[0]["content"] = sys0

        tool_called = False
        tool_payload: Dict[str, Any] = {}
        finish_reason: Optional[str] = None

        # Always call movies_now directly (vLLM not running with tool-call parser)
        args = _coerce_movie_args({}, user_text)
        try:
            tool_payload = movies_now(args.get("location"), args["start_date"], args["end_date"], args["limit"])
            tool_called = True
        except MoviesToolError:
            self._last_gen_meta = {"finish_reason": finish_reason, "usage": None, "params": {"tool_call": True}}
            return (
                "I couldn't fetch that—want me to try again or search another source?",
                False,
                finish_reason,
                [],
                tool_payload,
            )
        except Exception as exc:
            logger.warning("movies_now direct call failed: %s", exc)
            self._last_gen_meta = {"finish_reason": finish_reason, "usage": None, "params": {"tool_call": True}}
            return (
                "I couldn't fetch that—want me to try again or search another source?",
                False,
                finish_reason,
                [],
                tool_payload,
            )

        # Deterministic rendering: avoid LLM function-calling surface entirely
        titles = _titles_from_results(tool_payload)
        results = tool_payload.get("results") if isinstance(tool_payload, dict) else None
        if not results:
            self._last_gen_meta = {"finish_reason": "stop", "usage": None, "params": {"tool_call": False}}
            return (
                "I couldn't pull fresh showtimes just now. Want me to try again or switch to streaming picks?",
                tool_called,
                "stop",
                titles,
                tool_payload,
            )

        # Build a concise grounded reply from tool payload
        snippets: List[str] = []
        for item in results[:5]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            rel = item.get("release_date")
            overview = (item.get("overview") or "").strip()
            if overview and len(overview) > 140:
                overview = overview[:137] + "..."
            bit = title or ""
            if rel:
                bit += f" — {rel}"
            if overview:
                bit += f": {overview}"
            if bit:
                snippets.append(bit)
        reply = "Here are current theatrical picks:\n- " + "\n- ".join(snippets[:5])
        reply += "\nWant me to narrow by location or pull streaming instead?"

        self._last_gen_meta = {"finish_reason": "stop", "usage": None, "params": {"tool_call": False}}
        return reply, tool_called, "stop", titles, tool_payload

    def _log_tool_trace(
        self,
        *,
        turn_id: str,
        freshness_required: bool,
        planned_tools: List[str],
        tool_called: bool,
        finish_reason: Optional[str],
        reply: str,
        allowed_titles: List[str],
        planner_reason: str = "",
    ) -> None:
        violation = _hallucination_violation(reply, allowed_titles)
        trace = {
            "turn_id": turn_id,
            "freshness_required": bool(freshness_required),
            "tool_choice": planned_tools,
            "tool_called": bool(tool_called),
            "finish_reason": finish_reason,
            "hallucination_violation": violation,
            "planner_reason": planner_reason,
            "allowed_titles": allowed_titles,
        }
        try:
            logger.info("tool_trace %s", json.dumps(trace, ensure_ascii=False))
        except Exception:
            logger.debug("tool_trace (fallback): %s", trace)
        if freshness_required and not tool_called:
            logger.warning("⚠️ freshness_required but tool_called=False (turn=%s)", turn_id)

    # ------------------------------ tokenization ------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens robustly without ever calling str.encode on the model id/path."""
        # Prefer a true tokenizer first (many libs expose .tokenize on the model wrapper)
        if hasattr(self.model, "tokenize") and callable(getattr(self.model, "tokenize")):
            try:
                return len(self.model.tokenize(text))
            except TypeError:
                return len(self.model.tokenize(text.encode("utf-8", errors="ignore")))

        # If a separate tokenizer exists, use it — but only if it's not a plain string
        tok = getattr(self, "tokenizer", None)
        if tok is not None and not isinstance(tok, str):
            if hasattr(tok, "encode") and callable(getattr(tok, "encode")):
                try:
                    return len(tok.encode(text))
                except TypeError:
                    return len(tok.encode(text.encode("utf-8", errors="ignore")))
            if hasattr(tok, "tokenize") and callable(getattr(tok, "tokenize")):
                try:
                    return len(tok.tokenize(text))
                except TypeError:
                    return len(tok.tokenize(text.encode("utf-8", errors="ignore")))

        # Last-resort heuristic when no tokenizer API is available
        return max(1, len(text) // 4)

    # ------------------------------ prompt building ------------------------------

    def build_prompt(
        self, context: List[MemoryShard], user_input: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """Legacy helper: build a messages payload from lexi.memory shards + user input."""
        memories_json = []
        current_token_count = self.count_tokens(user_input) + self.SAFETY_MARGIN

        for m in context:
            shard = m.to_json()
            shard["content"] = re.sub(r"<\|.*?\|>", "", shard["content"]).strip()
            shard_tokens = self.count_tokens(shard["content"])
            if current_token_count + shard_tokens >= self.MAX_TOKENS:
                break
            memories_json.append(shard)
            current_token_count += shard_tokens

        memory_summary, _ = self.memory.build_prompt_context()
        return PromptTemplates.build_prompt(
            memories_json=memories_json,
            user_message=user_input,
            emotional_weights=self._axis_to_emotional_weights(),
            active_persona=self._lead_mode,
            memory_summary=memory_summary,
        )

    # ------------------------------ avatar / mode helpers ------------------------------

    def set_avatar_path(self, path: str):
        self._avatar_filename = path

    def get_avatar_path(self) -> Optional[str]:
        return self._avatar_filename or STARTER_AVATAR_PATH

    def get_mode(self) -> str:
        return self.current_mode

    def refresh_avatar_from_current(
        self, prompt_text: str, current_avatar_path: Optional[str] = None
    ) -> str:
        """
        Low-denoise img2img pass to preserve identity; subtle makeover.
        """
        source = current_avatar_path or self.get_avatar_path()
        if not source:
            return ""
        return generate_avatar(
            prompt=prompt_text,
            mode="img2img",
            source_path=source,  # absolute path to current avatar on disk
            denoise=0.32,
            steps=26,
            cfg_scale=4.2,
            changes="subtle warm makeup, slightly brighter background",
        )

    # ------------------------------ memory gather ------------------------------

    def _build_prompt_memories(self, session_limit=20, ltm_limit=5, msg_strip=None):
        # 1. Session memories from buffer
        session_entries = self.session_memory.buffer[-session_limit:]
        prompt_memories = []
        for entry in session_entries:
            prompt_memories.append({"role": "user", "content": entry["user"]})
            prompt_memories.append({"role": "assistant", "content": entry["ai"]})

        session_token_count = sum(self.count_tokens(m["content"]) for m in prompt_memories)

        # 2. LTM as MemoryShard
        ltm_budget = self.MAX_TOKENS - session_token_count - self.SAFETY_MARGIN
        ltm_memories = []
        if ltm_budget > 100 and msg_strip:
            try:
                all_mem = self.memory.all()
                relevant = sorted(all_mem, key=lambda s: score_overlap(msg_strip, s), reverse=True)
                ltm_shards = [s for s in relevant if score_overlap(msg_strip, s) >= 0.15][
                    :ltm_limit
                ]
                ltm_memories = [{"role": s.role, "content": s.content} for s in ltm_shards]
            except Exception as e:
                logger.warning("[WARN] Failed to retrieve memory: %s", e)
                ltm_memories = []

        total_token_count = session_token_count + sum(
            self.count_tokens(m["content"]) for m in ltm_memories
        )
        percent_used = total_token_count / self.MAX_TOKENS

        return prompt_memories + ltm_memories, percent_used

    # ------------------------------ chat entrypoints ------------------------------

    def chat(
        self,
        user_message: str,
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        # Defensive reset every turn so a prior prompt_pkg can't poison the codec.
        self.encoding = "utf-8"
        try:
            return self._chat_impl(user_message, stream_callback=stream_callback)
        except LookupError as e:
            logger.error("[TRACE] LookupError in chat: %r\n%s", e, traceback.format_exc())
            # Should never happen now, but keep a guard anyway.
            self.encoding = "utf-8"
            try:
                return self._chat_impl(user_message, stream_callback=stream_callback)
            except Exception as e2:
                logger.error(
                    "[ERROR] LexiPersona.chat failed after retry: %s\n%s",
                    e2,
                    traceback.format_exc(),
                )
                return "[error]"

    def _chat_impl(
        self,
        user_message: str,
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        self._step += 1
        turn_id = uuid4().hex
        if not user_message:
            return "[no input]"
        msg_strip = safe_strip(user_message)
        tool_required = needs_fresh_data(msg_strip)
        memory_tool_required = needs_memory_search(msg_strip)
        planner_tools: List[str] = []
        planner_reason = ""
        tool_called = False
        tool_titles: List[str] = []
        tool_finish_reason: Optional[str] = None

        if not hasattr(self, "_strict_mode"):
            self._strict_mode = False

        serious = self._assess_seriousness(msg_strip)
        self._seriousness_last = serious
        self._apply_seriousness_policy(serious)

        emotion_update = infer_emotion(msg_strip)
        for key, value in emotion_update.items():
            base = self.current_emotion_state.get(key, 0.0)
            self.current_emotion_state[key] = 0.7 * base + 0.3 * value

        self._update_mode_activation(f" {msg_strip.lower()} ")

        mode_axis_target = self._blend_axis_target_from_modes()
        nudged = get_persona_nudge_vector(mode_axis_target, user_baseline=None)
        self._tween_axis(nudged)

        self.log_chat("user", msg_strip)
        self.set_last_prompt(msg_strip)

        if msg_strip.startswith("diagnostic"):
            # NOTE: adjust port if needed
            try:
                return requests.get("http://localhost:8000/diagnostic", timeout=3).text
            except Exception:
                return "[diagnostic unavailable]"

        # Love Loop hook (optional chain)
        love_response = record_user_message(msg_strip, self.generate_reply)
        if love_response:
            return love_response

        # --- Build prompt context and budget ---
        try:
            self.session_memory.compact_oldest(max_keep_tokens=self.MAX_TOKENS - self.SAFETY_MARGIN)
        except Exception as e:
            logger.warning("[WARN] session compaction skipped: %s", e)

        # recent session turns
        all_mem = self.memory.all()
        top_score = 0.0
        for s in all_mem:
            try:
                top_score = max(top_score, score_overlap(msg_strip, s))
            except Exception:
                pass

        session_keep = 9 if top_score >= 0.25 else 16
        session_entries = self.session_memory.buffer[-session_keep:]

        prompt_memories: List[Dict[str, str]] = []
        for entry in session_entries:
            prompt_memories.append({"role": "user", "content": entry["user"]})
            prompt_memories.append({"role": "assistant", "content": entry["ai"]})

        session_token_count = sum(self.count_tokens(m["content"]) for m in prompt_memories)
        recent_blobs = [m["content"] for m in prompt_memories]

        def _too_similar_to_recent(txt: str) -> bool:
            low = txt.lower()
            if any(low in r.lower() or r.lower() in low for r in recent_blobs):
                return True

            def _tokset(s):
                return set(re.findall(r"[a-z0-9']{2,}", s.lower()))

            T = _tokset(txt)
            for r in recent_blobs[-6:]:
                R = _tokset(r)
                inter = len(T & R)
                uni = max(1, len(T | R))
                if inter / uni >= 0.80:
                    return True
            return False

        ltm_budget = self.MAX_TOKENS - session_token_count - self.SAFETY_MARGIN
        ltm_memories: List[Dict[str, str]] = []
        if ltm_budget > 120:
            try:
                all_mem = self.memory.all()
                relevant = sorted(all_mem, key=lambda s: score_overlap(msg_strip, s), reverse=True)
                existing_contents = set(m["content"] for m in prompt_memories)

                def _score(s):
                    return score_overlap(msg_strip, s)

                picked = []
                for s in relevant:
                    if _score(s) < 0.15:
                        continue
                    if s.content in existing_contents:
                        continue
                    if _too_similar_to_recent(s.content):
                        continue
                    picked.append(s)
                    if len(picked) >= 2:
                        break
                ltm_memories = [{"role": m.role, "content": m.content} for m in picked]
            except Exception as e:
                logger.warning("[WARN] Failed to retrieve memory: %s", e)
                ltm_memories = []

        memory_summary, _has_saved_context = self.memory.build_prompt_context()
        memory_rule = (
            "Memory rule: Use only saved notes from Known user context. "
            "If none are present, say you have no saved notes yet."
        )

        # --- Build the messages payload (prompt_pkg) ---
        emotional_weights = self._axis_to_emotional_weights()
        prompt_pkg = PromptTemplates.build_prompt(
            memories_json=prompt_memories + ltm_memories,
            user_message=msg_strip,
            emotional_weights=emotional_weights,
            active_persona=self._lead_mode,
            current_goal="",
            memory_summary=memory_summary,
            trait_summary="",
        )

        # Seed small-talk + on-demand research (added to system injections)
        feed_block = self._maybe_seed_now_feed()
        search_block = self._maybe_live_search(msg_strip)

        # Style profiling (build/update profile from recent user turns)
        recent_user_utts = [e["user"] for e in self.session_memory.buffer[-12:]]
        prev_prof = self._style_mem.get(self._style_user_id)
        style_prof = analyze_user_style(recent_user_utts, prev_profile=prev_prof)
        self._style_mem.set(self._style_user_id, style_prof)
        style_directives = make_style_directives(style_prof, persona_traits=self.traits)

        # Style & safety injections (merged into system message)
        style_bits: List[str] = []
        if serious >= 0.5:
            style_bits += [
                "This is serious. Be grounded, validating, and specific.",
                "Avoid flirtation, innuendo, or emojis.",
                "Acknowledge responsibility confusion without assigning blame.",
                "Offer one practical next step and ask permission before advice.",
            ]
        else:
            if not getattr(self, "_banter_allowed", True):
                style_bits.append("Keep it friendly but avoid flirtation or innuendo.")
            # Encourage tasteful use of feeds when not serious
            style_bits.append(
                "If using Now Feed or Research Notes, blend at most one tiny reference with a short source tag like (AP, 3h) or (TVMaze)."
            )

        if self.current_axis.get("warmth", 0.5) > 0.85:
            style_bits.append("Your tone is very warm and tactile — but keep it concise.")
        if self.current_axis.get("chaos", 0.5) > 0.8:
            style_bits.append("Be playful but concise; avoid metaphor chains.")
        if self.current_axis.get("energy", 0.5) < 0.4:
            style_bits.append("Slow your cadence; keep it soft and cozy — and succinct.")
        if self.current_axis.get("affection", 0.5) > 0.9:
            style_bits.append("Subtle flirtation is welcome; keep it to 2–3 clean sentences.")

        self._strict_mode = serious >= 0.5
        injections = [memory_rule] + list(self.system_injections) + [style_directives] + style_bits
        if feed_block:
            injections.append(feed_block)
        if search_block:
            injections.append(search_block)
        if self._strict_mode:
            injections += [
                "STRICT: Keep the reply to 2 sentences total.",
                "STRICT: Do not use emojis or excessive exclamation points.",
                "STRICT: Avoid motivational clichés (e.g., 'better days are ahead', 'you got this', 'curveball').",
            ]
        if injections:
            sys_msg = prompt_pkg["messages"][0]["content"]
            sys_msg = (
                sys_msg
                + "\n\n# Developer Injections (runtime)\n"
                + " ".join(x for x in injections if x)
            )
            prompt_pkg["messages"][0]["content"] = sys_msg

        # --- Token budgeting on message contents ---
        SAFE_GEN_TOKENS = 60
        budget = self.MAX_TOKENS - SAFE_GEN_TOKENS

        def _messages_tokens(pkg) -> int:
            return sum(self.count_tokens(m["content"]) for m in pkg["messages"])

        tok_count = _messages_tokens(prompt_pkg)

        # If over budget: progressively trim LTM, then session, then drop to minimal context
        if tok_count > budget:
            # Try removing LTM first
            trimmed_pkg = PromptTemplates.build_prompt(
                memories_json=prompt_memories,  # no ltm
                user_message=msg_strip,
                emotional_weights=emotional_weights,
                active_persona=self._lead_mode,
            )
            if injections:
                sys0 = trimmed_pkg["messages"][0]["content"]
                sys0 += "\n\n# Developer Injections (runtime)\n" + " ".join(
                    x for x in injections if x
                )
                trimmed_pkg["messages"][0]["content"] = sys0
            prompt_pkg = trimmed_pkg
            tok_count = _messages_tokens(prompt_pkg)

        if tok_count > budget and prompt_memories:
            # Keep only the last N user/assistant pairs
            keep_pairs = 6
            trimmed_pairs: List[Dict[str, str]] = []
            seen_user = 0
            for m in reversed(prompt_memories):
                trimmed_pairs.append(m)
                if m["role"].lower() == "user":
                    seen_user += 1
                    if seen_user >= keep_pairs:
                        break
            trimmed_pairs = list(reversed(trimmed_pairs))
            trimmed_pkg = PromptTemplates.build_prompt(
                memories_json=trimmed_pairs,
                user_message=msg_strip,
                emotional_weights=emotional_weights,
                active_persona=self._lead_mode,
            )
            if injections:
                sys0 = trimmed_pkg["messages"][0]["content"]
                sys0 += "\n\n# Developer Injections (runtime)\n" + " ".join(
                    x for x in injections if x
                )
                trimmed_pkg["messages"][0]["content"] = sys0
            prompt_pkg = trimmed_pkg
            tok_count = _messages_tokens(prompt_pkg)

        if tok_count > budget:
            # Minimal: no memories
            prompt_pkg = PromptTemplates.build_prompt(
                memories_json=[],
                user_message=msg_strip,
                emotional_weights=emotional_weights,
                active_persona=self._lead_mode,
            )
            if injections:
                sys0 = prompt_pkg["messages"][0]["content"]
                sys0 += "\n\n# Developer Injections (runtime)\n" + " ".join(
                    x for x in injections if x
                )
                prompt_pkg["messages"][0]["content"] = sys0
            tok_count = _messages_tokens(prompt_pkg)

        # --- Generate ---
        self.log_chat("prompt_user_turn", prompt_pkg["messages"][-1]["content"])

        # sampler tweaks from style profile + policy
        sampler = sampler_from_profile(style_prof) or {}
        sampler = _sampler_policy(self._lead_mode, serious, sampler)

        planner_tools, planner_reason = ([], "")
        planner_always = os.getenv("LEXI_PLANNER_ALWAYS", "0").lower() in ("1", "true", "yes", "on")
        planner_enabled_env = os.getenv("LEXI_ENABLE_TOOL_PLANNER", "1").lower() not in ("0", "false", "no", "off")
        if planner_enabled_env and ((tool_required or memory_tool_required) or planner_always):
            planner_tools, planner_reason = self._plan_tools(msg_strip, tool_required)

        if tool_required and not planner_tools:
            planner_tools = ["movies_now"]
        if memory_tool_required and "memory_search_ltm" not in planner_tools:
            planner_tools.append("memory_search_ltm")
            if not planner_reason:
                planner_reason = "heuristic_memory_trigger"
        if not tool_required and "movies_now" in planner_tools:
            tool_required = True

        reply: str
        tool_payload: Dict[str, Any] = {}
        if "movies_now" in planner_tools:
            reply, tool_called, tool_finish_reason, tool_titles, tool_payload = self._run_movies_tool_flow(
                prompt_pkg, msg_strip, sampler
            )
        else:
            reply = self._gen(
                prompt_pkg,
                sampler_overrides=sampler,
                stream_callback=stream_callback,
            )
            tool_finish_reason = self._last_gen_meta.get("finish_reason")

        # mirror punctuation/case tics lightly
        reply = apply_postprocessing(reply, style_prof)
        if not reply or len(reply) < 12:
            reply = "I’m here. Tell me more about what’s hitting the hardest right now."

        # Optional follow-up (Love Loop)
        try:
            next_q = maybe_ask_question()
            if next_q:
                reply += f"\n\n{next_q}"
        except Exception:
            pass

        # Memory write-back (session + persisted summaries)
        summary = ""
        try:
            summary = summarize_pair(self.loader.generate, msg_strip, reply)
            self.session_memory.add_pair(msg_strip, reply, summary, token_counter=self.count_tokens)
        except Exception as e:
            logger.warning("[WARN] summarize/add_pair failed: %s", e)

        if not summary and msg_strip:
            summary = msg_strip[:180].strip()

        facts = self.memory.extract_facts(msg_strip)
        try:
            self.memory.update_session_summary(
                summary,
                session_id=self._session_id,
                facts=facts,
            )
        except Exception as e:
            logger.warning("[WARN] session summary update failed: %s", e)

        finish_reason = tool_finish_reason or self._last_gen_meta.get("finish_reason")
        self._log_tool_trace(
            turn_id=turn_id,
            freshness_required=tool_required,
            planned_tools=planner_tools,
            tool_called=tool_called,
            finish_reason=finish_reason,
            reply=reply,
            allowed_titles=tool_titles,
            planner_reason=planner_reason,
        )

        return reply

    def reset_session(self):
        self.session_memory.reset()

        axes = get_persona_axes()
        default_vec = get_mode_axis_vector("default") or [0.8, 0.1, 0.85, 0.7, 0.85, 0.5]
        for i, a in enumerate(axes):
            cur = self.current_axis.get(a, default_vec[i])
            self.current_axis[a] = 0.7 * cur + 0.3 * default_vec[i]

        self._lead_mode = "default"
        self._lead_mode_lock_until = -1
        self.mode_activation.clear()
        self.mode_activation["default"] = 0.8
        self._normalize_activations()
        self.system_injections = []

    def generate_reply(self, user_input: str) -> str:
        try:
            emotional_weights = self._axis_to_emotional_weights()
            prompt_pkg = PromptTemplates.build_prompt(
                memories_json=[],  # No memory context for one-off replies
                user_message=user_input or "",
                emotional_weights=emotional_weights,
                active_persona=self._lead_mode,
            )
            self.log_chat("prompt_user_turn", prompt_pkg["messages"][-1]["content"])
            self._last_gen_meta = {"finish_reason": None, "usage": None, "params": {}}
            raw = self.loader.generate(prompt_pkg)
            if isinstance(raw, dict):
                self._last_gen_meta.update(
                    finish_reason=raw.get("finish_reason"),
                    usage=raw.get("usage"),
                )
            reply = self._clean_reply(raw)
            return reply or "[no reply]"
        except Exception as e:
            logger.error("[ERROR] generate_reply failed: %s", e)
            return "[error]"

    def _load_traits_state(self) -> bool:
        try:
            return bool(self.traits)
        except Exception:
            return False


lexi_persona = LexiPersona()
lex_persona = lexi_persona  # backward compatibility alias
__all__ = ["LexiPersona", "lexi_persona", "lex_persona"]
