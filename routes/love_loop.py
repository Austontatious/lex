import logging
import random
import time
from typing import Callable, Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from ..utils.emotion_core import infer_emotion

# Initialize router and logger
router = APIRouter(tags=["Love Loop"])
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants and configuration
# --------------------------------------------------------------------------- #
MIN_GAP_MESSAGES: int = 3
LOVE_LOOP_TAG = "love_loop_state"
# --------------------------------------------------------------------------- #
# 36-Question Intimacy Engine
# --------------------------------------------------------------------------- #
QUESTIONS: List[str] = [
    "Given the choice of anyone in the world, whom would you want as a dinner guest?",
    "Would you like to be famous? In what way?",
    "Before making a telephone call, do you ever rehearse what you are going to say? Why?",
    "What would constitute a perfect day for you?",
    "When did you last sing to yourself? To someone else?",
    "If you could retain either the mind or body of a 30-year-old, which would it be?",
    "Do you have a secret hunch about how you will die?",
    "Name three things you and I appear to have in common.",
    "For what in your life do you feel most grateful?",
    "If you could change anything about the way you were raised, what would it be?",
    "Tell me your life story in as much detail as possible (in 4 minutes).",
    "If you could gain any one quality or ability, what would it be?",
    "What truth would you ask a crystal ball to reveal?",
    "Is there something you’ve dreamed of doing? Why haven’t you done it?",
    "What is the greatest accomplishment of your life?",
    "What do you value most in a friendship?",
    "What is your most treasured memory?",
    "What is your most terrible memory?",
    "If you had one year left, would you change anything?",
    "What does friendship mean to you?",
    "What roles do love and affection play in your life?",
    "Share something positive about your partner.",
    "How warm/close is your family? Was your childhood happy?",
    "How do you feel about your relationship with your mother?",
    "Make three true \"we\" statements.",
    "Complete: 'I wish I had someone with whom I could share…'",
    "What should a close friend know about you?",
    "Say something you like about your partner that you wouldn’t normally share.",
    "Share an embarrassing moment in your life.",
    "When did you last cry in front of someone? Alone?",
    "Say something you already like about your partner.",
    "What, if anything, is too serious to joke about?",
    "If you were to die tonight, what would you most regret not telling someone?",
    "Your house is burning. After loved ones/pets, what item would you save?",
    "Whose death in your family would you find most disturbing? Why?",
    "Share a personal problem and ask for advice. Ask for emotional reflection."
]

# --------------------------------------------------------------------------- #
# Session state model
# --------------------------------------------------------------------------- #
class LoveSession(BaseModel):
    """
    Tracks the progress and state of the Love Loop session.
    """
    idx: int = 0
    waiting_for_answer: bool = False
    last_question: str = ""
    messages_since_question: int = 0
    started_ts: float = Field(default_factory=time.time)

# Global session state
state: LoveSession = LoveSession()

# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #
def save_love_loop_state():
    shard = MemoryShard.from_json({
        "role": "system",
        "content": f"Love Loop State: idx={state.idx}, waiting={state.waiting_for_answer}",
        "meta": {"love_loop_state": {
            "idx": state.idx,
            "waiting_for_answer": state.waiting_for_answer,
            "last_question": state.last_question,
            "messages_since_question": state.messages_since_question,
            "started_ts": state.started_ts
        }},
        "tags": [LOVE_LOOP_TAG]
    })
    try:
        memory.remember(shard)
    except Exception as e:
        logger.warning("Failed to persist Love Loop state: %s", e)

def load_love_loop_state():
    global state
    try:
        shards = [
            s for s in memory.all()
            if s.meta and isinstance(s.meta.get("tags"), list) and LOVE_LOOP_TAG in s.meta.get("tags")
        ]

        if not shards:
            return
        latest = shards[-1]
        meta = latest.meta.get("love_loop_state", {})
        if meta:
            state.idx = meta.get("idx", 0)
            state.waiting_for_answer = meta.get("waiting_for_answer", False)
            state.last_question = meta.get("last_question", "")
            state.messages_since_question = meta.get("messages_since_question", 0)
            state.started_ts = meta.get("started_ts", time.time())
            logger.info("Loaded Love Loop state: idx=%d waiting=%s", state.idx, state.waiting_for_answer)
    except Exception as e:
        logger.warning("Failed to load Love Loop state: %s", e)


def record_user_message(
    msg: str,
    chat_fn: Callable[[str], str]
) -> Optional[str]:
    """
    Process a user message within the Love Loop.

    If awaiting an answer, generate a response, tag it in memory, and advance the session.
    """
    global state
    if state.waiting_for_answer:
        prompt = (
            f"You just asked the user this emotional question:\n"  
            f"\"{state.last_question}\"\n\n"
            f"The user replied:\n\"{msg}\"\n\n"
            "Thank them warmly in 1-2 sentences, then share something personal in return."
        )
        reply = chat_fn(prompt).strip()

        # Store emotional exchange in memory
        emotion_scores = infer_emotion(msg)
        shard = MemoryShard.from_json({
            "role": "conversation",
            "content": f"User: {msg}\nLex: {reply}",
            "meta": {
                "session_id": "love_loop",
                "tags": ["intimacy_question"],
                "emotion": emotion_scores
            }
        })

        try:
            memory.remember(shard)
        except Exception as e:
            logger.warning("Memory store failed: %s", e)

        # Update session for next question
        state.waiting_for_answer = False
        state.idx += 1
        save_love_loop_state()
        state.messages_since_question = 0
        logger.debug("Asked question #%d and received reply", state.idx)
        return reply

    # Count non-question messages
    state.messages_since_question += 1
    return None


def maybe_ask_question() -> Optional[str]:
    """
    Determine if it's time to ask the next intimacy question.

    Returns a formatted question if ready, otherwise None.
    """
    global state
    # Do not proceed if awaiting answer or no more questions
    if state.waiting_for_answer or state.idx >= len(QUESTIONS):
        return None
    # Enforce minimum gap between questions
    if state.messages_since_question < MIN_GAP_MESSAGES:
        return None
    
    question = QUESTIONS[state.idx]
    opener = random.choice([
        "Can I ask something a bit deeper?",
        "Mind if I get a little personal?",
        "Here's a fun question:",
        "I’m curious about you —",
        "Let’s play the connection game:"
    ])
    full_question = f"{opener} {question}"

    # Update state
    state.last_question = question
    state.waiting_for_answer = True
    save_love_loop_state()
    state.messages_since_question = 0
    logger.debug("Asking Love Loop question %d: %s", state.idx + 1, question)
    return full_question

    
# --------------------------------------------------------------------------- #
# API endpoints
# --------------------------------------------------------------------------- #
@router.get("/status")
def get_status() -> LoveSession:
    """
    Retrieve the current Love Loop session state.
    """
    return state


@router.post("/reset")
def reset_session() -> dict:
    """
    Reset the Love Loop session to initial state.
    """
    global state
    state = LoveSession()
    return {"status": "reset"}

