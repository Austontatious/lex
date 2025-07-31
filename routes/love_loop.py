from __future__ import annotations

"""
Love Loop: 36-question emotional intimacy engine for Lex
"""

import logging
import random
import time
from typing import Optional, List, Callable

from fastapi import APIRouter
from pydantic import BaseModel

from ..memory.memory_core import memory
from ..memory.memory_types import MemoryShard
from lex.emotion_core import infer_emotion

log = logging.getLogger(__name__)
router = APIRouter()

# --------------------------------------------------------------------------- #
#  Questions (36)                                                             #
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
#  Session State                                                              #
# --------------------------------------------------------------------------- #
class LoveSession(BaseModel):
    idx: int = 0
    waiting_for_answer: bool = False
    last_question: str = ""
    messages_since_question: int = 0
    started_ts: float = time.time()

state = LoveSession()
MIN_GAP_MESSAGES = 3

# --------------------------------------------------------------------------- #
#  Core Hooks                                                                 #
# --------------------------------------------------------------------------- #
def record_user_message(msg: str, chat_fn: Callable[[str], str]) -> Optional[str]:
    global state
    if state.waiting_for_answer:
        prompt = (
            "You just asked the user this emotional question:\n"
            f"\"{state.last_question}\"\n\n"
            f"The user replied:\n\"{msg}\"\n\n"
            "Thank them warmly in 1-2 sentences, then share something personal in return."
        )
        reply = chat_fn(prompt).strip()

        # Tag emotional memory
        emotion = infer_emotion(msg)
        shard = MemoryShard.from_json(
            role="conversation",
            text=f"User: {msg}\nLex: {reply}",
            session_id="love_loop",
            tags=["intimacy_question"],
            meta={"emotion": emotion}
        )
        memory.remember(shard)

        state.waiting_for_answer = False
        state.idx += 1
        state.messages_since_question = 0
        log.debug("💕 Shared personal detail: %s", reply)
        return reply

    state.messages_since_question += 1
    return None

def maybe_ask_question() -> Optional[str]:
    global state
    if state.waiting_for_answer:
        return None
    if state.idx >= len(QUESTIONS):
        return None
    if state.messages_since_question < MIN_GAP_MESSAGES:
        return None

    q = QUESTIONS[state.idx]
    opener = random.choice([
        "Can I ask something a bit deeper?",
        "Mind if I get a little personal?",
        "Here's a fun question:",
        "I’m curious about you —",
        "Let’s play the connection game:"
    ])
    full_q = f"{opener} {q}"
    state.last_question = q
    state.waiting_for_answer = True
    state.messages_since_question = 0
    log.debug("💞 Asking question %d: %s", state.idx + 1, q)
    return full_q

# --------------------------------------------------------------------------- #
#  API Endpoints                                                              #
# --------------------------------------------------------------------------- #
@router.get("/status")
def status() -> LoveSession:
    return state

@router.post("/reset")
def reset():
    global state
    state = LoveSession()
    return {"status": "reset"}
