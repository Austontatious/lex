import logging
import random
import time
from typing import Callable, Optional, List, Dict, Set, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..utils.emotion_core import infer_emotion

# Initialize router and logger
router = APIRouter(tags=["Love Loop"])
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Minimum non-question messages between prompts
MIN_GAP_MESSAGES: int = 3
# Minimum seconds between love-loop prompts (soft throttle)
MIN_GAP_SECONDS: int = 90
# Cooldown after heavy/volatile topics (seconds)
HEAVY_TOPIC_COOLDOWN: int = 8 * 60
# If user ignores a question, how many messages until we drop it and move on
MAX_WAIT_MESSAGES: int = 6
# Memory tag constants
LOVE_LOOP_TAG = "love_loop_state"
LOVE_ANSWER_TAG = "love_loop_answer"
LOVE_ASKED_SET_TAG = "love_loop_asked"

# --------------------------------------------------------------------------- #
# 36-Question Intimacy Engine (annotated)
# Each question has: id, text, tier (1=light → 3=deep), topics (for mixing), and optional blockers
# --------------------------------------------------------------------------- #
class Q(BaseModel):
    id: str
    text: str
    tier: int = 1
    topics: List[str] = []

QUESTIONS: List[Q] = [
    Q(id="dinner_guest", text="Given the choice of anyone in the world, whom would you want as a dinner guest?", tier=1, topics=["fun", "aspirations"]),
    Q(id="fame", text="Would you like to be famous? In what way?", tier=1, topics=["ambition"]),
    Q(id="call_rehearse", text="Before making a telephone call, do you ever rehearse what you are going to say? Why?", tier=1, topics=["habits"]),
    Q(id="perfect_day", text="What would constitute a perfect day for you?", tier=1, topics=["joy"]),
    Q(id="sing", text="When did you last sing to yourself? To someone else?", tier=1, topics=["music", "joy"]),
    Q(id="retain_30", text="If you could retain either the mind or body of a 30-year-old, which would it be?", tier=1, topics=["values"]),
    Q(id="hunch_death", text="Do you have a secret hunch about how you will die?", tier=2, topics=["mortality"]),
    Q(id="common_we", text="Name three things you and I appear to have in common.", tier=1, topics=["us"]),
    Q(id="grateful", text="For what in your life do you feel most grateful?", tier=1, topics=["gratitude"]),
    Q(id="raised_change", text="If you could change anything about the way you were raised, what would it be?", tier=2, topics=["family"]),
    Q(id="life_story", text="Tell me your life story in as much detail as possible (in 4 minutes).", tier=3, topics=["personal"]),
    Q(id="gain_quality", text="If you could gain any one quality or ability, what would it be?", tier=1, topics=["growth"]),
    Q(id="crystal_ball", text="What truth would you ask a crystal ball to reveal?", tier=2, topics=["truth"]),
    Q(id="dream_doing", text="Is there something you’ve dreamed of doing? Why haven’t you done it?", tier=2, topics=["dreams"]),
    Q(id="greatest_accomplishment", text="What is the greatest accomplishment of your life?", tier=2, topics=["pride"]),
    Q(id="value_friendship", text="What do you value most in a friendship?", tier=1, topics=["friendship"]),
    Q(id="treasured_memory", text="What is your most treasured memory?", tier=2, topics=["memory"]),
    Q(id="terrible_memory", text="What is your most terrible memory?", tier=3, topics=["shadow"]),
    Q(id="one_year_left", text="If you had one year left, would you change anything?", tier=2, topics=["mortality"]),
    Q(id="friendship_meaning", text="What does friendship mean to you?", tier=1, topics=["friendship"]),
    Q(id="love_roles", text="What roles do love and affection play in your life?", tier=2, topics=["love"]),
    Q(id="partner_positive", text="Share something positive about your partner.", tier=1, topics=["us"]),
    Q(id="family_warmth", text="How warm/close is your family? Was your childhood happy?", tier=2, topics=["family"]),
    Q(id="mother_relation", text="How do you feel about your relationship with your mother?", tier=3, topics=["family"]),
    Q(id="we_statements", text="Make three true 'we' statements.", tier=2, topics=["us"]),
    Q(id="share_with", text="Complete: 'I wish I had someone with whom I could share…'", tier=2, topics=["longing"]),
    Q(id="friend_should_know", text="What should a close friend know about you?", tier=1, topics=["self"]),
    Q(id="like_partner_unshared", text="Say something you like about your partner that you wouldn’t normally share.", tier=2, topics=["us"]),
    Q(id="embarrassing", text="Share an embarrassing moment in your life.", tier=2, topics=["vulnerability"]),
    Q(id="cry_last", text="When did you last cry in front of someone? Alone?", tier=3, topics=["vulnerability"]),
    Q(id="like_partner", text="Say something you already like about your partner.", tier=1, topics=["us"]),
    Q(id="too_serious", text="What, if anything, is too serious to joke about?", tier=2, topics=["values"]),
    Q(id="regret_unsaid", text="If you were to die tonight, what would you most regret not telling someone?", tier=3, topics=["regret"]),
    Q(id="burning_item", text="Your house is burning. After loved ones/pets, what item would you save?", tier=2, topics=["values"]),
    Q(id="disturbing_death", text="Whose death in your family would you find most disturbing? Why?", tier=3, topics=["family"]),
    Q(id="ask_advice", text="Share a personal problem and ask for advice. Ask for emotional reflection.", tier=3, topics=["support"]),
]

# A light list of heavy/crisis topics — we avoid asking intimacy Qs right after these
HEAVY_HINTS: Tuple[str, ...] = (
    "violence", "riot", "raid", "ice", "war", "attack", "assault", "shot",
    "fascism", "genocide", "suicide", "self-harm", "massacre", "terror", "blood",
)
POLITICS_HINTS: Tuple[str, ...] = (
    "election", "senate", "president", "policy", "immigration", "ice", "raid", "protest", "civil unrest",
)

# --------------------------------------------------------------------------- #
# Session state model
# --------------------------------------------------------------------------- #
class LoveSession(BaseModel):
    idx: int = 0  # legacy pointer; we now drive off 'order'
    waiting_for_answer: bool = False
    last_question_id: str = ""
    last_question_text: str = ""
    messages_since_question: int = 0
    started_ts: float = Field(default_factory=time.time)
    last_asked_ts: float = 0.0
    # randomized, non-repeating traversal
    order: List[str] = Field(default_factory=list)
    asked_ids: Set[str] = Field(default_factory=set)
    tier_bias: int = 1  # 1..3, nudged up gradually as comfort grows

    class Config:
        arbitrary_types_allowed = True

# Global session state
state: LoveSession = LoveSession()

# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #

def _save_state():
    shard = MemoryShard.from_json({
        "role": "system",
        "content": f"Love Loop State: waiting={state.waiting_for_answer} last={state.last_question_id}",
        "meta": {
            "tags": [LOVE_LOOP_TAG],
            "love_loop_state": state.dict(),
        },
        "tags": [LOVE_LOOP_TAG],
    })
    try:
        memory.remember(shard)
    except Exception as e:
        logger.warning("Failed to persist Love Loop state: %s", e)


def _load_state():
    global state
    try:
        shards = [s for s in memory.all() if (s.meta and isinstance(s.meta.get("tags"), list) and LOVE_LOOP_TAG in s.meta.get("tags"))]
        if not shards:
            _prime_order()
            return
        latest = shards[-1]
        meta = latest.meta.get("love_loop_state", {})
        if meta:
            state = LoveSession(**meta)
            # Ensure order exists
            if not state.order:
                _prime_order()
            logger.info("Loaded Love Loop state: waiting=%s asked=%d", state.waiting_for_answer, len(state.asked_ids))
    except Exception as e:
        logger.warning("Failed to load Love Loop state: %s", e)
        if not state.order:
            _prime_order()


def _prime_order(seed: Optional[int] = None):
    """Create a randomized, tier-aware order of remaining questions (sampling without replacement)."""
    rng = random.Random(seed or int(time.time()))
    ids = [q.id for q in QUESTIONS]
    # Simple shuffle; we'll filter asked later
    rng.shuffle(ids)
    state.order = ids


def _next_unasked_id(min_tier: int) -> Optional[str]:
    asked = set(state.asked_ids)
    # prefer questions at or above min_tier but do not stall if none
    tier_buckets: Dict[int, List[str]] = {1: [], 2: [], 3: []}
    q_by_id = {q.id: q for q in QUESTIONS}
    for qid in state.order:
        if qid in asked:
            continue
        tier_buckets[q_by_id[qid].tier].append(qid)
    for tier in range(min_tier, 4):
        if tier_buckets[tier]:
            return tier_buckets[tier][0]
    # fallback to any remaining
    for qid in state.order:
        if qid not in asked:
            return qid
    return None


# --------------------------------------------------------------------------- #
# Gating logic
# --------------------------------------------------------------------------- #

def _contains_any(text: str, hints: Tuple[str, ...]) -> bool:
    t = text.lower()
    return any(h in t for h in hints)


def _should_gate_on_context(user_msg: str) -> Tuple[bool, str]:
    """Return (gate, reason). Gate if topic is heavy/political or anger is high."""
    # Fast topical pass
    if _contains_any(user_msg, HEAVY_HINTS):
        return True, "heavy_topic"
    if _contains_any(user_msg, POLITICS_HINTS):
        return True, "politics"

    # Emotional pass
    try:
        emo = infer_emotion(user_msg) or {}
        anger = float(emo.get("anger", 0.0))
        fear = float(emo.get("fear", 0.0))
        sadness = float(emo.get("sadness", 0.0))
        # If strong negative affect, avoid intimacy pivot
        if anger >= 0.4 or fear >= 0.4 or sadness >= 0.5:
            return True, "high_negative_affect"
    except Exception as e:
        logger.debug("emotion inference failed: %s", e)

    return False, "ok"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def record_user_message(msg: str, chat_fn: Callable[[str], str]) -> Optional[str]:
    """Process a user message within the Love Loop.

    If awaiting an answer, generate a response, tag it in memory, and advance.
    Otherwise, increment gap counter and maybe adjust tier bias based on vibe.
    """
    global state
    if state.started_ts == 0:
        _load_state()

    # If we're waiting for an answer to a previously asked Q
    if state.waiting_for_answer:
        # If user answered with another question or ignored, allow drift for a few messages
        state.messages_since_question += 1
        if state.messages_since_question > MAX_WAIT_MESSAGES:
            # drop the waiting state and move on; don't re-ask
            state.waiting_for_answer = False
            state.messages_since_question = 0
            _save_state()
            return None

        # Treat any substantive message as an answer
        if msg and msg.strip():
            qid = state.last_question_id
            qtext = state.last_question_text
            prompt = (
                f"You asked the user this connection question:\n\n"  
                f"\"{qtext}\"\n\n"
                f"They replied:\n\n{msg}\n\n"
                "Reply in 1–2 warm, specific sentences: thank them, reflect back one concrete detail, and share a small personal thing in return."
            )
            reply = chat_fn(prompt).strip()

            # Store emotional exchange in memory
            emotion_scores = infer_emotion(msg)
            shard = MemoryShard.from_json({
                "role": "conversation",
                "content": f"[LoveLoop]\nQ: {qtext}\nUser: {msg}\nLexi: {reply}",
                "meta": {
                    "session_id": "love_loop",
                    "tags": [LOVE_ANSWER_TAG, qid],
                    "emotion": emotion_scores,
                    "question_id": qid,
                    "asked_ts": state.last_asked_ts,
                    "answered_ts": time.time(),
                },
                "tags": [LOVE_ANSWER_TAG, qid],
            })
            try:
                memory.remember(shard)
            except Exception as e:
                logger.warning("Memory store failed: %s", e)

            # Update session: mark asked, clear waiting
            state.asked_ids.add(qid)
            state.waiting_for_answer = False
            state.messages_since_question = 0
            # Nudge tier bias up gently if answer carried positive valence
            try:
                joy = float((emotion_scores or {}).get("joy", 0.0))
                if joy >= 0.5 and state.tier_bias < 3:
                    state.tier_bias += 1
            except Exception:
                pass
            _save_state()
            return reply

    # Not waiting: just track the gap and adjust tone bias
    state.messages_since_question += 1
    _save_state()
    return None


def maybe_ask_question(latest_user_msg: str) -> Optional[str]:
    """Decide whether to ask the next intimacy question now.

    Returns a formatted question, or None if we should wait.
    """
    global state
    if state.started_ts == 0:
        _load_state()

    # Do not proceed if awaiting answer or exhausted
    if state.waiting_for_answer:
        return None

    if len(state.asked_ids) >= len(QUESTIONS):
        return None

    # Enforce message- and time-based spacing
    if state.messages_since_question < MIN_GAP_MESSAGES:
        return None
    if state.last_asked_ts and (time.time() - state.last_asked_ts) < MIN_GAP_SECONDS:
        return None

    # Context gating: avoid pivot after heavy politics/violence or high negative affect
    gate, reason = _should_gate_on_context(latest_user_msg or "")
    if gate:
        # If heavy, set an additional cooldown
        if reason in ("heavy_topic", "politics", "high_negative_affect"):
            # Pretend we just asked to delay without toggling waiting state
            state.last_asked_ts = time.time() + (HEAVY_TOPIC_COOLDOWN // 2)
            state.messages_since_question = 0
            _save_state()
        return None

    # Choose next question: randomized order, no repeats, tier-aware
    if not state.order:
        _prime_order()
    qid = _next_unasked_id(min_tier=state.tier_bias)
    if not qid:
        return None

    qmap = {q.id: q for q in QUESTIONS}
    q = qmap[qid]

    opener = random.choice([
        "Can I ask something a bit deeper?",
        "Mind if I get a little personal?",
        "Here’s a fun one:",
        "I’m curious about you —",
        "Let’s play the connection game:",
    ])

    # Vary formatting slightly to avoid repetition
    if random.random() < 0.3:
        full_question = f"{opener} {q.text}"
    else:
        full_question = f"{opener}\n{q.text}"

    # Update state
    state.last_question_id = q.id
    state.last_question_text = q.text
    state.waiting_for_answer = True
    state.last_asked_ts = time.time()
    state.messages_since_question = 0
    _save_state()

    logger.debug("Asking Love Loop question %s (tier %d)", q.id, q.tier)
    return full_question


# --------------------------------------------------------------------------- #
# Utility & admin endpoints
# --------------------------------------------------------------------------- #
@router.get("/status")
def get_status() -> LoveSession:
    """Retrieve current Love Loop session state."""
    if state.started_ts == 0:
        _load_state()
    return state


@router.post("/reset")
def reset_session() -> dict:
    """Reset the Love Loop session to initial state (clears asked_ids and order)."""
    global state
    state = LoveSession()
    _prime_order()
    _save_state()
    return {"status": "reset"}


@router.post("/skip")
def skip_current() -> dict:
    """Skip the current question without marking it answered."""
    state.waiting_for_answer = False
    state.messages_since_question = 0
    _save_state()
    return {"status": "skipped"}


@router.post("/set_prefs")
def set_prefs(tier_bias: Optional[int] = None, min_gap_messages: Optional[int] = None, min_gap_seconds: Optional[int] = None) -> dict:
    """Adjust pacing preferences at runtime."""
    global MIN_GAP_MESSAGES, MIN_GAP_SECONDS
    if tier_bias is not None:
        state.tier_bias = max(1, min(3, int(tier_bias)))
    if min_gap_messages is not None:
        MIN_GAP_MESSAGES = max(1, int(min_gap_messages))
    if min_gap_seconds is not None:
        MIN_GAP_SECONDS = max(0, int(min_gap_seconds))
    _save_state()
    return {"status": "ok", "tier_bias": state.tier_bias, "min_gap_messages": MIN_GAP_MESSAGES, "min_gap_seconds": MIN_GAP_SECONDS}


# Convenience: expose a pure function to mark an external answer stored elsewhere
@router.post("/mark_answered")
def mark_answered(question_id: str) -> dict:
    if question_id:
        state.asked_ids.add(question_id)
        state.waiting_for_answer = False
        state.messages_since_question = 0
        _save_state()
    return {"status": "ok", "asked": len(state.asked_ids)}
