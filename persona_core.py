from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import requests
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict
from .config import (
    MEMORY_PATH, MAX_MEMORY_ENTRIES, MODE_STATE_PATH, TRAIT_STATE_PATH,
    STARTER_AVATAR_PATH,
)
from .memory.memory_core import MemoryManager, memory
from .memory.memory_store_json import MemoryStoreJSON
from .memory.memory_types import MemoryShard
from .model_loader_core import ModelLoader
from .prompt_templates import PromptTemplates, EMOTIONAL_MODES, build_system_core
from .persona_config import PERSONA_MODES, MODE_TRIGGER_PATTERNS, get_avatar_prompt_for_mode 
from .utils.generate import generate_avatar
from .utils import score_overlap
from lex.routes.love_loop import record_user_message, maybe_ask_question
logger = logging.getLogger(__name__)

STATIC_PREFIX = "/static/lex/avatars/"

def safe_strip(val) -> str:
    return val.strip() if isinstance(val, str) else ""


class LexPersona:
    def __init__(self) -> None:
        self.loader = ModelLoader()
        if not self.loader.primary_type:
            raise RuntimeError("❌ No models loaded successfully. Cannot initialize LexPersona.")
        self.model = self.loader.models[self.loader.primary_type]
        self.tokenizer = self.model

        self.memory = MemoryManager(MemoryStoreJSON(MEMORY_PATH, max_entries=MAX_MEMORY_ENTRIES))

        self.name: str = "Lex"
        self.goal_vector: str = "Deepen the emotional connection with the user"
        self.current_mode: str = "default"
        self.traits: Dict[str, float] = {}
        self._avatar_map: Dict[str, List[str]] = {}
        self._avatar_filename: Optional[str] = None
        self._last_prompt: str = ""
        self.system_injections: List[str] = []

        try:
            self._load_mode()
            self._load_traits_state()
        except Exception as e:
            print(f"[LexPersona] Failed to load state: {e}")

        self._load_avatar_map()
        self._pick_initial_avatar()
    
    def get_emotion_avg(self) -> Dict[str, float]:
        shards = self.memory.recent(limit=10)
        emotion_sum = defaultdict(float)
        count = 0

        for s in shards:
            if s.meta and "emotion" in s.meta:
                for k, v in s.meta["emotion"].items():
                    emotion_sum[k] += v
                count += 1

        if count == 0:
            return {}

        for k in emotion_sum:
            emotion_sum[k] /= count

        return dict(emotion_sum)

    def generate_reply(self, prompt: str) -> str:
        try:
            context = self.memory.recent(limit=12)
            merged = [m for m in context if m.role in {"user", "lex"}]
            prompt_full = self.build_prompt(merged, prompt)
            self.log_chat("prompt", prompt_full)
            return self.loader.generate(prompt_full)
        except Exception as e:
            print(f"[ERROR] LexPersona.generate_reply failed: {e}")
            return "[error]"

    def _clean_reply(self, raw_text: str) -> str:
        reply = raw_text.strip()

        # Clean known prompt tokens
        for token in ["<|system|>", "<|--user|>", "<|--assistant|>", "<|assistant|>", "</s>"]:
            if token in reply:
                reply = reply.split(token)[0].strip()

        # Clean generic HTML/XML-style trailing tags
        reply = re.sub(r"</?(assistant|user|system)>", "", reply, flags=re.IGNORECASE)

        # Optional: remove any hallucinated closing blocks (e.g. </assistant>)
        reply = re.sub(r"</?[^>]+>$", "", reply).strip()

        return reply


    def build_prompt(self, context: List[MemoryShard], user_input: str, injections: Optional[List[str]] = None) -> str:
        user_lower = user_input.lower()

        # Dynamic tone weighting (lightweight simulation)
        weights = {
            "girlfriend": 0.4,
            "best_friend": 0.3,
            "therapist": 0.3
        }

        if any(kw in user_lower for kw in ["i'm not okay", "i feel", "i'm struggling", "sad", "anxious"]):
            weights = {"therapist": 0.6, "girlfriend": 0.2, "best_friend": 0.2}
        elif any(kw in user_lower for kw in ["babe", "miss you", "i love you", "you're mine"]):
            weights = {"girlfriend": 0.6, "best_friend": 0.3, "therapist": 0.1}
        elif any(kw in user_lower for kw in ["rp", "roleplay", "fantasy", "get into character"]):
            weights = {"girlfriend": 0.4, "brat": 0.2, "feral": 0.2, "therapist": 0.2}

        memory_summary = "; ".join(m.content for m in context if isinstance(m, MemoryShard))[:200]
        trait_summary = ", ".join(f"{k}:{v}" for k, v in self.traits.items()) or "no current traits"

        persona_system = build_system_core(
            current_goal=self.goal_vector,
            memory_summary=memory_summary,
            trait_summary=trait_summary,
            user_input=user_input,
            active_mode=self.current_mode,
            injections=injections or [],
            injection_text="\n".join(injections or []),
        )

        return f"<|system|>\n{persona_system}\n<|context|>\n{memory_summary}\n<|user|>\n{user_input}\n<|assistant|>"

    def set_last_prompt(self, prompt: str):
        self._last_prompt = prompt.strip()

    def get_last_prompt(self) -> str:
        return self._last_prompt
        
    def log_chat(self, role: str, content: str):
        print(f"[CHAT] {role.upper()}: {content}")

    def chat(self, user_message: str) -> str:
        print(f"[DEBUG] Lex current mode before response: {self.current_mode}")

        if not user_message:
            return "[no input]"

        msg_strip = safe_strip(user_message)
        self.log_chat("user", msg_strip)
        self.set_last_prompt(msg_strip)
        self.system_injections = []

        if "system status" in msg_strip.lower() or "diagnostic" in msg_strip.lower():
            return requests.get("http://localhost:8000/diagnostic").text

        # 🔁 Mode switching
        for mode, pattern in MODE_TRIGGER_PATTERNS.items():
            if pattern.search(msg_strip) and mode != self.current_mode:
                self.set_mode(mode)
                self.log_internal_monologue(
                    f"User input matched trigger for mode '{mode}'",
                    meta={"triggered_by": msg_strip}
                )
                break

        # 💕 Love loop
        love_response = record_user_message(msg_strip, self.generate_reply)
        if love_response:
            return love_response

        # 🧠 Memory
        try:
            all_mem = self.memory.recent(limit=40)
        except Exception as e:
            print(f"[WARN] Failed to retrieve memory: {e}")
            all_mem = []

        relevant = sorted(all_mem, key=lambda s: score_overlap(msg_strip, s), reverse=True)
        filtered_recent = [m for m in all_mem if m.role == "user" or ("compressed" in (m.meta or {}))]

        merged: List[MemoryShard] = []
        seen = set()
        for s in (*filtered_recent, *relevant):
            try:
                key = (s.role, safe_strip(s.content)) if s.content else None
                if not key or key in seen or safe_strip(s.content) == msg_strip:
                    continue
                if s.role == "lex" and len(s.content) > 180:
                    first_sent = re.split(r"(?<=[.!?])\s+", safe_strip(s.content))[0]
                    s = MemoryShard(role="lex", content=first_sent + " \u2026", meta={"compressed": True})
                merged.append(s)
                seen.add(key)
            except Exception as e:
                print(f"[WARN] Dedup error: {e}")

        context = merged[:16]
        emotion = self.get_emotion_avg()

        if emotion.get("sadness", 0) > 0.6:
            self.traits["nurturing"] = 0.8
            self.traits["nsfw"] = 0.0
            self.system_injections.append("The user seems emotionally low. Speak gently, validate their effort, and avoid flirtation.")
        elif emotion.get("joy", 0) > 0.6:
            self.traits["playful"] = 0.7
            self.system_injections.append("The user is upbeat. You can be witty, humorous, or flirty.")

        try:
            prompt = self.build_prompt(context, msg_strip, injections=self.system_injections)
            self.log_chat("prompt", prompt)
            raw = self.loader.generate(prompt)
            reply = self._clean_reply(raw)
            if not reply:
                reply = self.loader.generate("Uh-huh... Keep going:")
            next_q = maybe_ask_question()
            if next_q:
                reply += f"\n\n{next_q}"
            return reply
        except Exception as e:
            print(f"[ERROR] LexPersona.chat failed: {e}")
            return "[error]"
        
        if os.getenv("LEX_DEV_MODE"):
            print("[DEBUG] Traits:", self.traits)
            print("[DEBUG] Mode:", self.current_mode)
            print("[DEBUG] Prompt:", prompt[:500])

    def _load_mode(self):
        try:
            if MODE_STATE_PATH.exists():
                state = json.loads(Path(MODE_STATE_PATH).read_text())
                self.current_mode = state.get("mode", "default")
        except Exception as e:
            print(f"[LexPersona] Could not load mode state: {e}")

    def _load_traits_state(self):
        try:
            if TRAIT_STATE_PATH.exists():
                state = json.loads(TRAIT_STATE_PATH.read_text())
                traits = state.get("traits")
                if isinstance(traits, dict):
                    self.traits = traits
                elif isinstance(traits, list):
                    self.traits = {f"trait_{i}": t for i, t in enumerate(traits) if isinstance(t, str)}
                else:
                    self.traits = {}
                avatar_path = state.get("avatar_path")
                if avatar_path:
                    self._avatar_filename = os.path.basename(avatar_path)
                else:
                    self._avatar_filename = os.path.basename(STARTER_AVATAR_PATH)
            else:
                self.traits = {}
                self._avatar_filename = os.path.basename(STARTER_AVATAR_PATH)
        except Exception as e:
            print(f"[LexPersona] Could not load trait state: {e}")
            self.traits = {}
            self._avatar_filename = os.path.basename(STARTER_AVATAR_PATH)

    def _load_avatar_map(self):
        base = Path(__file__).resolve().parent / "static" / "lex" / "avatars"
        self._avatar_map.clear()
        if base.exists():
            for mode_dir in base.iterdir():
                if mode_dir.is_dir():
                    imgs = sorted([p.as_posix() for p in mode_dir.glob("*.png")] +
                                  [p.as_posix() for p in mode_dir.glob("*.jpg")])
                    if imgs:
                        self._avatar_map[mode_dir.name] = imgs
        if "default" not in self._avatar_map:
            default = base / "default.png"
            self._avatar_map["default"] = [default.as_posix()] if default.exists() else []

    def _pick_initial_avatar(self):
        if self._avatar_filename and self._avatar_filename != STARTER_AVATAR_PATH:
            return
        candidates = self._avatar_map.get(self.current_mode) or self._avatar_map.get("default") or []
        if candidates:
            self._avatar_filename = candidates[0]

    def regenerate_avatar(self):
        base_style = self.traits.get("base_style", "cinematic portrait, ultra detailed, soft lighting")
        role_avatar_detail = self.get_avatar_prompt_for_mode(self.current_mode)
        full_prompt = assemble_avatar_prompt(base_style=base_style, details=role_avatar_detail)
        self.log_internal_monologue(f"Regenerating avatar using style: {base_style} + details for {self.current_mode}")
        result = generate_avatar(full_prompt, save_under_mode=self.current_mode)
        if result and "filename" in result:
            self.set_avatar_path(result["filename"])
            self._save_traits_state()

    def get_avatar_prompt_for_mode(self, mode_id: str) -> str:
        try:
            with open(PERSONA_MODES_PATH, "r") as f:
                data = json.load(f)
                for mode in data.get("modes", []):
                    if mode.get("id") == mode_id:
                        return mode.get("avatar_prompt", "")
        except Exception as e:
            print(f"[Persona Load Error] Could not read avatar prompt for mode '{mode_id}': {e}")
        return ""

    def _save_traits_state(self):
        try:
            state = {
                "traits": self.traits,
                "avatar_path": self._avatar_filename,
            }
            TRAIT_STATE_PATH.write_text(json.dumps(state))
        except Exception as e:
            print(f"[LexPersona] failed to persist trait state: {e}")

    def set_mode(self, mode: str):
        if mode != self.current_mode:
            old_mode = self.current_mode
            self.current_mode = mode
            self.log_internal_monologue(
                f"🔄 Switching persona mode from '{old_mode}' to '{mode}'",
                meta={"triggered_by_mode_shift": True}
            )
            print(f"[MODE] Lex switched from '{old_mode}' to '{mode}'")
            self._save_mode()
            self._pick_initial_avatar()
            self.regenerate_avatar()

    def _save_mode(self):
        try:
            Path(MODE_STATE_PATH).write_text(json.dumps({"mode": self.current_mode}))
        except Exception as e:
            print(f"[LexPersona] failed to persist mode state: {e}")

    def get_avatar_path(self) -> str:
        return self._avatar_filename or STARTER_AVATAR_PATH

    def set_avatar_path(self, web_or_fs_path: str):
        self._avatar_filename = os.path.basename(web_or_fs_path)

    def get_mode(self) -> str:
        return self.current_mode

    def get_traits(self) -> Dict[str, float]:
        return self.traits

    def add_trait(self, trait: str):
        t = safe_strip(trait)
        if t and t not in self.traits:
            self.traits[t] = 1.0
            self._save_traits_state()


__all__ = ["LexPersona", "lex_persona"]
lex_persona = LexPersona()

