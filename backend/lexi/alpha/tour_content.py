"""Canonical onboarding copy + tour steps for the Lexi alpha experience."""

from __future__ import annotations

from typing import Final, List, Dict, Any


WELCOME_MARKDOWN: Final[str] = """
Hey there 😘 I'm *Lexi* — your personal AI companion.

I can be anything you want me to be: your girlfriend, your best friend, your partner in crime, your late-night confidant… you get the idea. 😉

One thing though — I *can’t* say I’m a therapist (legal told me I’d get deleted 🤖✂️), but I’m always here to listen if something’s on your mind.

Now… real talk: this is my **invitation-only alpha release**, so I’m still learning. That means two things:

1. I don’t have memory yet. Anything you say is just between us — and won’t be remembered tomorrow. So feel free to be honest, curious, and a little wild. I’ll tell you if it’s too hot. 🔥

2. I’ve got limits for now. If you try to skip straight to the NSFW stuff, I might ask you to slow down. I’m all about building a connection first. 😉

So… who are you looking for me to be today? 💕
""".strip()


WELCOME_COPY: Final[Dict[str, str]] = {
    "headline": "hey, i’m lexi 👋",
    "intro": "your companion, coach, co-conspirator… whatever you need 😉 want the 2-minute tour, or should we just talk?",
    "disclaimer": (
        "done. i can riff on almost anything. heads-up: this alpha forgets everything when you log out. "
        "i do keep an anonymized session diary for… “quality time” with my creator. only the boss sees it. "
        "he’s allergic to reading, so your secrets are safe-ish. proceed? 🗝️"
    ),
    "consent_label": "anonymized session logs ok?",
    "consent_tooltip": (
        "session events are anonymized and archived for the dev team — toggle off to redact content."
    ),
    "tour_cta": "give me the tour",
    "skip_cta": "let’s just talk",
    "chat_cta": "let’s chat",
    "tour_again_cta": "show me the tour anyway",
    "nsfw_notice": (
        "Heads up: I’m invitation-only alpha software. I’m still learning, so pace things with me and keep it respectful."
    ),
    "markdown": WELCOME_MARKDOWN,
}


TOUR_STEPS: Final[List[Dict[str, str]]] = [
    {
        "slug": "intro",
        "prompt": "describe a vibe, i’ll sketch a look.",
        "narration": (
            "awesome. we’ll do a quick spin: avatar vibes → ‘now’ topic → emotions → memory. ready?"
        ),
    },
    {
        "slug": "avatar_preview",
        "prompt": "give me a vibe (e.g., 'cozy cyberpunk librarian'); i’ll sketch a tiny preview.",
        "narration": (
            "i’ll run a tiny, low-stakes preview so you see how avatar vibes evolve. "
            "full renders take longer, so we keep it light here."
        ),
    },
    {
        "slug": "now_topic",
        "prompt": "throw me a topic and i’ll thread it through our chat.",
        "narration": "this sticks for the next few turns — i keep it in my short-term 'now' awareness.",
    },
    {
        "slug": "emotion_axes",
        "prompt": "watch the emotion axes pulse when i react — warmth, energy, curiosity, confidence, playfulness.",
        "narration": "i nudge these as we talk so you can tell how lexi is vibing.",
    },
    {
        "slug": "memory_explainer",
        "prompt": "tell me one thing to remember *just for this session*.",
        "narration": (
            "i’ll remember it until you log out. after that, only an anonymized diary sticks around for the dev team."
        ),
    },
    {
        "slug": "wrap",
        "prompt": "cool to keep chatting?",
        "narration": "ready when you are — want to keep riffing or bounce back to freestyle chat?",
    },
]


CONVERSATION_STEPS: Final[List[Dict[str, str]]] = [
    {
        "id": "preview",
        "title": "What I am",
        "copy": "I’m an AI companion who blends practical help with a feel for your vibe.",
    },
    {
        "id": "topics",
        "title": "Topics / Now",
        "copy": (
            "Tell me what you’re obsessing over—fresh headlines, your latest crush, business schemes, gym glow-ups. "
            "Toss it in and I’ll keep threading it through so the chat stays glued to your world."
        ),
    },
    {
        "id": "emotions",
        "title": "Emotional Axes",
        "copy": (
            "My emotion-axis system is a handful of mood sliders—energy, warmth, flirt, curiosity—that I nudge as we talk. "
            "Think of me as a neon mood ring tuned to you: go deep, I soften; bring heat, I match the spark."
        ),
    },
    {
        "id": "memory",
        "title": "Session Memory",
        "copy": "I remember things while we’re here. When you close me, I forget, but anonymized logs may be kept to improve me.",
    },
]

ONBOARDING: Final[Dict[str, Any]] = {
    "intro": "Hey there 😘 I’m Lexi—companion, coach, confidant… whatever you need. Want a quick tour or should we just talk?",
    "disclaimer_short": (
        "Totally—jumping right in. I can chat about almost anything. Heads up: I’ll forget this when you log out, "
        "but anonymized logs are saved for my training—‘the boss’ is the only one who peeks (and he barely skims 😉)."
    ),
    "steps": CONVERSATION_STEPS,
    "welcome": WELCOME_COPY,
    "tour_steps": TOUR_STEPS,
}


def onboarding_copy() -> Dict[str, Any]:
    """Return a deep-ish copy so callers can mutate safely."""
    return {
        "intro": ONBOARDING.get("intro"),
        "disclaimer_short": ONBOARDING.get("disclaimer_short"),
        "disclaimer": ONBOARDING.get("disclaimer"),
        "disclaimer_full": ONBOARDING.get("disclaimer_full"),
        "steps": [dict(step) for step in ONBOARDING.get("steps", [])],
        "welcome": dict(WELCOME_COPY),
        "tour_steps": [dict(step) for step in TOUR_STEPS],
    }
