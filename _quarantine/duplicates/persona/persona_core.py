from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import requests
import tiktoken
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict

from ..config.config import (
    MEMORY_PATH, MAX_MEMORY_ENTRIES, MODE_STATE_PATH, TRAIT_STATE_PATH,
    STARTER_AVATAR_PATH,
)
from ..memory.memory_core import MemoryManager, memory
from ..memory.memory_store_json import MemoryStoreJSON
from ..memory.memory_types import MemoryShard
from ..memory.session_memory import SessionMemoryManager
from ..core.model_loader_core import ModelLoader
from .prompt_templates import PromptTemplates, EMOTIONAL_MODES, build_system_core
from .persona_config import (
    PERSONA_MODE_REGISTRY,
    assemble_avatar_prompt,
    get_persona_axes,            # <— needed
    get_mode_axis_vector,        # <— needed
)
from .persona_manipulation import get_persona_nudge_vector
from ..sd.generate import generate_avatar
from ..utils import score_overlap
from ..utils.emotion_core import infer_emotion
from ..utils.summarize import summarize_pair
from ..routes.love_loop import record_user_message, maybe_ask_question, load_love_loop_state

logger = logging.getLogger(__name__)
STATIC_PREFIX = "/static/lex/avatars/"
SMALL = 1e-9

def safe_strip(val) -> str:
    return val.strip() if isinstance(val, str) else ""

class LexPersona:
    MAX_TOKENS = 4096
    SAFETY_MARGIN = 200
    
    def __init__(self) -> None:
        self.loader = ModelLoader()
        if not self.loader.primary_type:
            raise RuntimeError("❌ No models loaded successfully. Cannot initialize LexPersona.")
        self.model = self.loader.models[self.loader.primary_type]
        self.tokenizer = self.model
        self.memory = MemoryManager(MemoryStoreJSON(MEMORY_PATH, max_entries=MAX_MEMORY_ENTRIES))
        load_love_loop_state()  # <-- Load Love Loop persistent state here

        self.memory = MemoryManager(MemoryStoreJSON(MEMORY_PATH, max_entries=MAX_MEMORY_ENTRIES))
        self.session_memory = SessionMemoryManager(max_pairs=20)
        self.current_emotion_state: Dict[str, float] = {"joy": 0.5, "sadness": 0.0, "arousal": 0.0}
        self.name: str = "Lex"
        self.goal_vector: str = "Deepen the emotional connection with the user"
        self.current_mode: str = "default"
        self.traits: Dict[str, float] = {}
        self._avatar_map: Dict[str, List[str]] = {}
        self._avatar_filename: Optional[str] = None
        self._last_prompt: str = ""
        self.system_injections: List[str] = []
        self.current_mode: str = "default"

        # NEW: step counter & hysteresis
        self._step = 0
        self._lead_mode_lock_until = -1  # turn index
        self._lead_mode: str = "default"
        self._seriousness_last: float = 0.0

        # NEW: mode activation blend (softmax-ish, but we’ll store raw activations and normalize)
        self.mode_activation: Dict[str, float] = defaultdict(float)
        self.mode_activation["default"] = 0.8

        # NEW: axis state (EMA tweening)
        axes = get_persona_axes()
        default_vec = get_mode_axis_vector("default") or [0.8,0.1,0.85,0.7,0.85,0.5]
        self.axis_names = axes
        self.current_axis = {a: default_vec[i] for i, a in enumerate(axes)}
        self.target_axis  = dict(self.current_axis)

        # tuneables
        self.activation_nudge = 0.35      # how much a strong regex hit bumps activation
        self.activation_decay = 0.08       # per turn decay
        self.lead_threshold   = 0.70
        self.lead_lock_turns  = 3          # refractory period after a decisive shift
        self.axis_alpha       = 0.25       # EMA tween rate per turn

    # --- helpers
    # Drop these anywhere inside the LexPersona class (near other small helpers)

    def log_chat(self, role: str, content: str) -> None:
        """Lightweight chat logger used throughout the class."""
        try:
            logger.debug("[CHAT][%s] %s", role, content)
        except Exception:
            pass  # never let logging kill the turn

    def set_last_prompt(self, prompt: str) -> None:
        """Remember the last compiled prompt for debugging."""
        self._last_prompt = prompt or ""

    def _clean_reply(self, raw) -> str:
        import re
        if raw is None:
            return ""

        text = raw["text"].strip() if isinstance(raw, dict) and "text" in raw else (
            raw.strip() if isinstance(raw, str) else str(raw)
        )

        # Strip xml-ish artifacts
        text = re.sub(r"</?(assistant|system|user)>", "", text, flags=re.I)

        # Collapse whitespace and line spam
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # De-dupe near-identical paragraphs (exact)
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        exact_keys = set()
        first_pass = []
        for p in paras:
            key = re.sub(r"[^\w\s]", "", p.lower())
            if key not in exact_keys:
                first_pass.append(p)
                exact_keys.add(key)

        # De-dupe near-duplicates (fuzzy: token Jaccard)
        def _tokset(s):
            return set(re.findall(r"[a-z0-9']{2,}", s.lower()))
        filtered = []
        for p in first_pass:
            toks_p = _tokset(p)
            too_similar = False
            for q in filtered:
                inter = len(toks_p & _tokset(q))
                uni = max(1, len(toks_p | _tokset(q)))
                if inter / uni >= 0.85:   # similarity threshold
                    too_similar = True
                    break
            if not too_similar:
                filtered.append(p)
        text = "\n\n".join(filtered)

        # If banter off, scrub flirt/innuendo-y lines
        if hasattr(self, "_banter_allowed") and not self._banter_allowed:
            # kill over-the-top innuendo & snark that show up often
            BLOCK = [
                r"\bbuckle up\b",
                r"\bsecret rendezvous\b",
                r"\blet's get( a little)? closer\b",
                r"\bgive .* a big middle finger\b",
                r"\banything goes\b",
            ]
            for pat in BLOCK:
                text = re.sub(pat, "", text, flags=re.I)
            # remove trailing orphan lines after stripping
            text = re.sub(r"\n{2,}", "\n\n", text).strip()
            # remove most emojis to keep it grounded
            text = re.sub(r"[\U0001F300-\U0001FAFF]", "", text)

        # Sentence clamp (skip when serious or user asked for detail)
        wants_detail = bool(re.search(r"\b(more detail|explain|why|how|tell me more|elaborate)\b", text, flags=re.I))
        is_serious = getattr(self, "_seriousness_last", 0.0) >= 0.5
        if not wants_detail and not is_serious:
            # Split into sentences conservatively; keep first 2–3
            sents = re.split(r"(?<=[.!?…])\s+", text)
            kept = sents[:3]  # hard cap
            text = " ".join(kept).strip()

        # Ensure it ends cleanly (period or question mark)
        if not re.search(r"[.!?…]$", text):
            text += "."

        return text.strip()



    def _normalize_activations(self):
        total = sum(self.mode_activation.values()) or 1.0
        for k in list(self.mode_activation.keys()):
            self.mode_activation[k] /= total

    def _decay_activations(self):
        for k in list(self.mode_activation.keys()):
            self.mode_activation[k] = max(0.0, self.mode_activation[k] * (1.0 - self.activation_decay))
        # keep default from collapsing
        self.mode_activation["default"] = max(self.mode_activation.get("default", 0.0), 0.2)

    def _score_triggers(self, text: str) -> Dict[str, float]:
        scores = {}
        imperative_boost = 1.0
        if any(v in text for v in [" be ", " act ", " play ", " pretend ", " be a ", " be my "]):
            imperative_boost = 1.25

        third_party_guard = any(ref in text for ref in [' my daughter', ' she ', ' her ', ' they ', ' their ', ' them '])

        for mode_id, info in PERSONA_MODE_REGISTRY.items():
            pat = info.get("trigger")
            if not pat:
                continue
            if pat.search(text):
                base = 1.0
                if info.get("imperative_required", False) and imperative_boost <= 1.0:
                    base *= 0.4  # asked but not imperative enough
                if third_party_guard:
                    base *= 0.2  # likely not about Lex
                scores[mode_id] = base * imperative_boost
        return scores

    def _gen(self, prompt: str) -> str:
        """Centralize generation knobs + safe stop sequences."""
        gen_opts = dict(
            max_new_tokens=80,          # ~2–3 sentences
            temperature=0.55,
            top_p=0.9,
            presence_penalty=0.2,
            repetition_penalty=1.12,    # reduce repeats
            stop=[
                "</assistant>", "<|end|>",
                "\n\nUser:", "\n\n<|user|>",  # existing
                "<|user|>", "User:", "\nUser:",  # new guards
                "<|system|>", "\n<|system|>"     # don't spill back into system
            ]
        )
        try:
            raw = self.loader.generate(prompt, **gen_opts)
        except TypeError:
            # If ModelLoader doesn't accept kwargs, degrade gracefully
            raw = self.loader.generate(prompt)
        return self._clean_reply(raw)

    def _update_mode_activation(self, text: str):
        self._decay_activations()
        hits = self._score_triggers(text)
        for mode_id, strength in hits.items():
            self.mode_activation[mode_id] = min(1.25, self.mode_activation.get(mode_id, 0.0) + strength * self.activation_nudge)
        self._normalize_activations()

        # hysteresis: pick lead if something clearly wins
        top_mode = max(self.mode_activation.items(), key=lambda kv: kv[1])[0]
        if self.mode_activation[top_mode] >= self.lead_threshold and self._step >= self._lead_mode_lock_until:
            self._lead_mode = top_mode
            self._lead_mode_lock_until = self._step + self.lead_lock_turns

    def _blend_axis_target_from_modes(self) -> Dict[str, float]:
        # weighted average of axis vectors
        blended = {a: 0.0 for a in self.axis_names}
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

        bf = max(0.0, 0.5*joy + 0.5*energy)
        gf = max(0.0, 0.6*affection + 0.4*warmth)
        th = max(0.0, 0.6*(1.0-anger) + 0.4*warmth)

        s = bf + gf + th + SMALL
        return {"best_friend": bf/s, "girlfriend": gf/s, "therapist": th/s}

    def _assess_seriousness(self, text: str) -> float:
        """Return 0..1 seriousness score based on keywords + current emotions."""
        t = text.lower()
        # add whatever else you want here
        hard = [
            "hospital", "er", "suicide", "self harm", "assault", "relapse",
            "alcoholism", "withdrawal", "rehab", "detox", "domestic", "panic attack",
            "grief", "funeral", "custody", "court", "abuse", "overdose"
        ]
        soft = ["lonely", "alone", "blame", "my fault", "scared", "worried", "anxious", "depressed"]

        score = 0.0
        if any(w in t for w in hard): score += 0.7
        if any(w in t for w in soft): score += 0.25

        # blend with current emotion state (if you track it)
        score += 0.2 * max(0.0, self.current_emotion_state.get("sadness", 0.0))
        return min(1.0, score)

    def _apply_seriousness_policy(self, seriousness: float):
        """Throttle flirt, boost Therapist, extend lock when serious."""
        self._banter_allowed = seriousness < 0.35
        if seriousness >= 0.5:
            # push Therapist up and lock it in briefly
            self.mode_activation["therapist"] = max(self.mode_activation.get("therapist", 0.0), 0.95)
            self._normalize_activations()
            # extend the hysteresis lock a bit longer when serious
            self._lead_mode = "therapist"
            self._lead_mode_lock_until = max(self._lead_mode_lock_until, self._step + 5)

    def count_tokens(self, text: str) -> int:
        if hasattr(self.model, "tokenize"):
            return len(self.model.tokenize(text.encode("utf-8")))
        if hasattr(self.model, "encode"):
            return len(self.model.encode(text))
        return max(1, len(text) // 4)

    def build_prompt(self, context: List[MemoryShard], user_input: str) -> str:
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

        return PromptTemplates.build_prompt(
            memories_json=memories_json,
            user_message=user_input,
            emotional_weights=self._axis_to_emotional_weights(),
            active_persona=self._lead_mode
        )


        
    def set_avatar_path(self, path: str):
        self._avatar_filename = path

    def get_avatar_path(self) -> Optional[str]:
        return self._avatar_filename or STARTER_AVATAR_PATH

    def get_mode(self) -> str:
        return self.current_mode       
    
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
                ltm_shards = [s for s in relevant if score_overlap(msg_strip, s) >= 0.15][:ltm_limit]
                # simple: inject at most ltm_limit non-weak shards
                ltm_memories = [{"role": s.role, "content": s.content} for s in ltm_shards]
            except Exception as e:
                logger.warning("[WARN] Failed to retrieve memory: %s", e)
                ltm_memories = []

        total_token_count = session_token_count + sum(self.count_tokens(m["content"]) for m in ltm_memories)
        percent_used = total_token_count / self.MAX_TOKENS

        return prompt_memories + ltm_memories, percent_used

    def chat(self, user_message: str) -> str:
        self._step += 1
        if not user_message:
            return "[no input]"
        msg_strip = safe_strip(user_message)

        # emotion update → gently blend into axes via your existing infer_emotion, then persona nudge
        serious = self._assess_seriousness(msg_strip)
        self._seriousness_last = serious
        self._apply_seriousness_policy(serious)

        emotion_update = infer_emotion(msg_strip)
        for key, value in emotion_update.items():
            base = self.current_emotion_state.get(key, 0.0)
            self.current_emotion_state[key] = 0.7*base + 0.3*value

        # update mode activations from triggers (soft + hysteresis)
        self._update_mode_activation(f" {msg_strip.lower()} ")

        # fuse mode-blended axis target + nudges toward healthier state
        mode_axis_target = self._blend_axis_target_from_modes()
        nudged = get_persona_nudge_vector(mode_axis_target, user_baseline=None)
        self._tween_axis(nudged)

        self.log_chat("user", msg_strip)
        self.set_last_prompt(msg_strip)

        if msg_strip.startswith("diagnostic"):
            return requests.get("http://localhost:8000/diagnostic").text

        love_response = record_user_message(msg_strip, self.generate_reply)
        if love_response:
            return love_response

        # --- Build prompt context and budget ---
        
        # compact session if token budget is getting tight
        try:
            self.session_memory.compact_oldest(
                max_keep_tokens=self.MAX_TOKENS - self.SAFETY_MARGIN
            )
        except Exception as e:
            logger.warning("[WARN] session compaction skipped: %s", e)

        # Quick LTM strength estimation (no model call)
        from ..utils import score_overlap
        all_mem = self.memory.all()
        top_score = 0.0
        for s in all_mem:
            try:
                top_score = max(top_score, score_overlap(msg_strip, s))
            except Exception:
                pass

        # Adaptive session vs LTM balance
        session_keep = 10 if top_score >= 0.25 else 18
        session_entries = self.session_memory.buffer[-session_keep:]

        prompt_memories = []
        for entry in session_entries:
            prompt_memories.append({"role": "user", "content": entry["user"]})
            prompt_memories.append({"role": "assistant", "content": entry["ai"]})

        session_token_count = sum(self.count_tokens(m["content"]) for m in prompt_memories)
        # Build a fast lookup of recent content to avoid LTM echo
        recent_blobs = [m["content"] for m in prompt_memories]
        def _too_similar_to_recent(txt: str) -> bool:
            low = txt.lower()
            # substring check (cheap)
            if any(low in r.lower() or r.lower() in low for r in recent_blobs):
                return True
            # token jaccard vs last few turns (tighter)
            def _tokset(s): return set(re.findall(r"[a-z0-9']{2,}", s.lower()))
            T = _tokset(txt)
            for r in recent_blobs[-6:]:
                R = _tokset(r)
                inter = len(T & R)
                uni = max(1, len(T | R))
                if inter / uni >= 0.80:
                    return True
            return False
        
        ltm_budget = self.MAX_TOKENS - session_token_count - self.SAFETY_MARGIN
        ltm_memories = []
        if ltm_budget > 120:
            try:
                all_mem = self.memory.all()
                relevant = sorted(all_mem, key=lambda s: score_overlap(msg_strip, s), reverse=True)
                existing_contents = set(m["content"] for m in prompt_memories)
                # threshold to skip weak matches
                def _score(s): return score_overlap(msg_strip, s)
                picked = []
                for s in relevant:
                    if _score(s) < 0.15:   # ignore weak matches
                        continue
                    if s.content in existing_contents:
                        continue
                    if _too_similar_to_recent(s.content):
                        continue
                    picked.append(s)
                    if len(picked) >= 3:
                        break
                ltm_memories = [{"role": m.role, "content": m.content} for m in picked]
            except Exception as e:
                logger.warning("[WARN] Failed to retrieve memory: %s", e)
                ltm_memories = []

        total_token_count = session_token_count + sum(self.count_tokens(m["content"]) for m in ltm_memories)
        
        if total_token_count / self.MAX_TOKENS > 0.9:
            # Instead of bailing, drop oldest LTM memories until safe
            while ltm_memories and (total_token_count / self.MAX_TOKENS) > 0.85:
                dropped = ltm_memories.pop(0)
                total_token_count -= self.count_tokens(dropped["content"])
            # If still too big, drop oldest session memories (last resort)
            while prompt_memories and (total_token_count / self.MAX_TOKENS) > 0.85:
                prompt_memories.pop(0)
                total_token_count = sum(self.count_tokens(m["content"]) for m in prompt_memories + ltm_memories)

            # If *still* too big after trimming, just continue with what’s left
            logger.warning("[WARN] Token budget still tight after trimming — proceeding anyway")

        # micro-transition style injection based on axis deltas
        style_bits = []
        if serious >= 0.5:
            style_bits += [
                "This is serious. Be grounded, validating, and specific.",
                "Avoid flirtation, innuendo, or emojis.",
                "Acknowledge responsibility confusion without assigning blame.",
                "Offer one practical next step and ask permission before advice."
            ]
        else:
            if not self._banter_allowed:
                style_bits.append("Keep it friendly but avoid flirtation or innuendo.")
        if self.current_axis.get("warmth", 0.5) > 0.85:
            style_bits.append("Your tone is very warm and tactile — but keep it concise.")
        if self.current_axis.get("chaos", 0.5) > 0.8:
            style_bits.append("Be playful but concise; avoid metaphor chains.")
        if self.current_axis.get("energy", 0.5) < 0.4:
            style_bits.append("Slow your cadence; keep it soft and cozy — and succinct.")
        if self.current_axis.get("affection", 0.5) > 0.9:
            style_bits.append("Subtle flirtation is welcome; keep it to 2–3 clean sentences.")

        injections = self.system_injections + style_bits

        try:
            emotional_weights = self._axis_to_emotional_weights()
            # Build prompt with the *lead* persona (hysteresis) but blended style inside
            prompt = PromptTemplates.build_prompt(
                memories_json=prompt_memories + ltm_memories,
                user_message=msg_strip,
                emotional_weights=emotional_weights,
                active_persona=self._lead_mode
            )
            # Lightweight injection: prepend an extra system line
            if not self._banter_allowed:
                injections.append("Do not use flirtation or romantic innuendo in this reply.")
            # Global length guard unless serious
            if serious < 0.5:
                injections.append("Keep this reply to 2–3 sentences, maximum.")
            if injections:
                prompt = f"<|system|>{' '.join(injections)}\n" + prompt

            self.log_chat("prompt", prompt)
            reply = self._gen(prompt)
            if not reply or len(reply) < 12:
                reply = "I’m here. Tell me more about what’s hitting the hardest right now."


            next_q = maybe_ask_question()
            if next_q:
                reply += f"\n\n{next_q}"

            summary = summarize_pair(self.loader.generate, msg_strip, reply)
            self.session_memory.add_pair(msg_strip, reply, summary, token_counter=self.count_tokens)

            return reply
        except Exception as e:
            logger.error("[ERROR] LexPersona.chat failed: %s", e)
            return "[error]"


    def reset_session(self):
        # clear short-term chat history
        self.session_memory.reset()

        # soften back toward default vibe (don’t hard-reset; keep it natural)
        axes = get_persona_axes()
        default_vec = get_mode_axis_vector("default") or [0.8, 0.1, 0.85, 0.7, 0.85, 0.5]
        for i, a in enumerate(axes):
            cur = self.current_axis.get(a, default_vec[i])
            self.current_axis[a] = 0.7 * cur + 0.3 * default_vec[i]

        # unlock hysteresis and bias toward default without whiplash
        self._lead_mode = "default"
        self._lead_mode_lock_until = -1
        self.mode_activation.clear()
        self.mode_activation["default"] = 0.8
        self._normalize_activations()

        # clear transient style nudges
        self.system_injections = []

            
    def generate_reply(self, user_input: str) -> str:
        try:
            emotional_weights = self._axis_to_emotional_weights() 
            prompt = PromptTemplates.build_prompt(
                memories_json=[],  # No memory context for one-off replies
                user_message=user_input,
                emotional_weights=emotional_weights
            )
            self.log_chat("prompt", prompt)
            raw = self.loader.generate(prompt)
            reply = self._clean_reply(raw)
            return reply or "[no reply]"
        except Exception as e:
            logger.error("[ERROR] generate_reply failed: %s", e)
            return "[error]"

    def _load_traits_state(self) -> bool:
        # minimal: return True if traits exist on disk or in memory
        try:
            return bool(self.traits)  # or actually read TRAIT_STATE_PATH here if you want
        except Exception:
            return False


__all__ = ["LexPersona", "lex_persona"]
lex_persona = LexPersona()

