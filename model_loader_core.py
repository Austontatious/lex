# ── lex/model_loader_core.py (merged & refactored) ─────────────────────
"""
Unified model loader combining prior multi‑model (friday/model_loader_core.py)
capabilities with the refined heuristic, anti‑ramble streaming logic from
lex/model_loader.py.

Key Features:
  * Supports **multiple named models** via a declarative `model_config` (if present).
  * Provides a **primary conversational model** (shortcut: `.primary`) for legacy
    single‑model callers (so existing `from .model_loader_core import ModelLoader; ml = ModelLoader(); ml.generate(...)` still works).
  * Exposes **generate()** and **generate_stream()** with intelligent early‑finish
    heuristics (micro‑mirror detection, paragraph duplicate, metaphor overrun,
    question inflation, sentence cap) that only trigger after a coherent answer
    forms – preserving "aliveness" without arbitrary truncations.
  * Passes frequency & presence penalties on both blocking and streaming paths.
  * Optional visualization of token embeddings (guarded by env var).
  * Environment tunables (prefixed `LEX_` / `FRIDAY_`) let you adjust heuristics
    without editing code.

Backward Compatibility:
  * If no external `model_config` (multi‑model registry) is available, we fall
    back to the single‑model config constants in `config.py` (MODEL_PATH, etc.).
  * Old imports like `from .model_loader import ModelLoader` can be changed to
    `from .model_loader_core import ModelLoader`.
  * Existing code that used `ModelLoader().llama` still works.

Usage Patterns:
    ml = ModelLoader()                 # loads (multi or single)
    text = ml.generate(prompt)
    for chunk in ml.generate_stream(prompt):
        ... append to UI ...

    # multi-model access
    other = ml.get(ModelType.EMBED)    # if defined in model_config

Heuristic tuning at top of file.
"""
from __future__ import annotations

import os
import re
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, List, Any

import numpy as np
from llama_cpp import Llama

# --------------------------------------------------------------------- #
#  Logging                                                              #
# --------------------------------------------------------------------- #
logger = logging.getLogger("lex.model_loader")
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------- #
#  Attempt multi‑model registry import (optional)                       #
# --------------------------------------------------------------------- #
# Expected interface if present:
#   model_config.configs: Dict[ModelType, ModelConfig]
#   ModelConfig(path, context_length, gpu_layers, temperature, top_p,
#               repetition_penalty, stop, frequency_penalty?, presence_penalty?)
try:  # pragma: no cover - optional dependency
    from .model_config import model_config, ModelType, ModelConfig  # type: ignore
    HAVE_MULTI = True
except Exception:  # Fallback to single‑model constants
    HAVE_MULTI = False
    from .config import (
        MODEL_PATH,
        CONTEXT_LENGTH,
        GPU_LAYERS,
        TEMPERATURE,
        TOP_P,
        REPEAT_PENALTY,
        FREQUENCY_PENALTY,
        PRESENCE_PENALTY,
        STOP,
        MAX_TOKENS,
    )

# --------------------------------------------------------------------- #
#  Heuristic Tunables                                                   #
# --------------------------------------------------------------------- #
FLUSH_LEN               = int(os.getenv("LEX_STREAM_FLUSH_LEN", "8"))
FLUSH_INTERVAL_SEC      = float(os.getenv("LEX_STREAM_FLUSH_INTERVAL", "0.035"))
MICRO_MIRROR_MIN_TAIL   = int(os.getenv("LEX_MICRO_TAIL", "60"))
MICRO_MIRROR_MIN_HALF   = int(os.getenv("LEX_MICRO_HALF", "18"))
PARA_DUP_MIN_ACC_LEN    = int(os.getenv("LEX_PARA_ACC_MIN", "160"))
PARA_DUP_MIN_BLOCK_LEN  = int(os.getenv("LEX_PARA_BLOCK_MIN", "70"))
SENTENCE_EARLY_STOP_MAX = int(os.getenv("LEX_SENTENCE_CAP", "5"))
QUESTION_OVERFLOW_LIMIT = int(os.getenv("LEX_QUESTION_LIMIT", "2"))
EARLY_METAPHOR_LIMIT    = int(os.getenv("LEX_METAPHOR_LIMIT", "1"))
ENABLE_VIS              = os.getenv("LEX_VIZ_DISABLE", "0") != "1"
DEBUG_STREAM            = os.getenv("LEX_DEBUG_STREAM", "0") == "1"
DEBUG_BLOCK             = os.getenv("LEX_DEBUG_BLOCK", "0") == "1"
DEFAULT_MAX_TOKENS      = int(os.getenv("LEX_MAX_TOKENS", str(2048 if not HAVE_MULTI else 512)))
PHRASE_REPEAT_MIN_LEN   = int(os.getenv("LEX_PHRASE_REPEAT_MIN", "15"))  # New: Phrase repetition detection

METAPHOR_WORDS = {
    "library", "archive", "galaxy", "ocean", "sea", "forest", "hive", "lab",
    "workshop", "garden", "train station", "jazz", "orchestra", "theater",
}
_LIKE_AS_RE = re.compile(r"\b(like|as)\s+a[n]?\s+([a-z ]{2,30})", re.I)
_SENT_END_RE = re.compile(r"(?<=[.!?])\s+")

# --------------------------------------------------------------------- #
#  Embedding helper                                                     #
# --------------------------------------------------------------------- #

def _static_token_embedding(llama: Llama, tok: str):
    try:
        ids = llama.tokenize(tok.encode("utf-8"), add_bos=False)
    except Exception:
        return None
    if not ids:
        return None
    tid = ids[0]
    for getter in ("get_embeddings", "get_embedding"):
        if hasattr(llama, getter):
            try:
                vec = getattr(llama, getter)(tid)
                return np.asarray(vec, dtype=np.float32)
            except Exception:
                pass
    for path in ("_model.token_embeddings", "_model._token_embeddings"):
        try:
            obj = llama
            for part in path.split("."):
                obj = getattr(obj, part)
            return np.asarray(obj[tid], dtype=np.float32)
        except Exception:
            continue
    return None

# --------------------------------------------------------------------- #
#  Data structures (fallback single model config)                       #
# --------------------------------------------------------------------- #
if not HAVE_MULTI:
    @dataclass
    class ModelConfig:  # minimal shim
        path: str
        context_length: int
        gpu_layers: int
        temperature: float
        top_p: float
        repetition_penalty: float
        stop: list[str]
        frequency_penalty: float
        presence_penalty: float
        max_tokens: int

    class ModelType(str):  # simple shim behaving like an Enum
        VALUE = "primary"
        def __new__(cls, value):  # pragma: no cover
            return str.__new__(cls, value)
        @property
        def value(self):  # pragma: no cover
            return self

    class _SingleConfigContainer:  # shim to mimic multi interface
        def __init__(self):
            self.configs = {ModelType("primary"): ModelConfig(
                path=MODEL_PATH,
                context_length=CONTEXT_LENGTH,
                gpu_layers=GPU_LAYERS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPEAT_PENALTY,
                stop=STOP,
                frequency_penalty=FREQUENCY_PENALTY,
                presence_penalty=PRESENCE_PENALTY,
                max_tokens=MAX_TOKENS,
            )}
    model_config = _SingleConfigContainer()

# GPU heuristic (works for single & multi)                               
USE_GPU_DEFAULT = any(cfg.gpu_layers > 32 for cfg in model_config.configs.values())

# --------------------------------------------------------------------- #
#  Core Loader                                                          #
# --------------------------------------------------------------------- #
class ModelLoader:
    """Multi / single hybrid loader with heuristic generation methods."""

    def __init__(self):
        self.models: Dict[ModelType, Llama] = {}
        self.failed_models: Dict[ModelType, str] = {}
        self.cfgs: Dict[ModelType, ModelConfig] = model_config.configs

        logger.info("🔍 Initializing ModelLoader (%d configs)...", len(self.cfgs))
        for mtype, cfg in self.cfgs.items():
            try:
                self.models[mtype] = self._load_llama_model(cfg)
                logger.info("✅ Loaded model: %s", getattr(mtype, 'value', mtype))
            except Exception as e:
                self.failed_models[mtype] = str(e)
                logger.error("❌ Failed to load model %s: %s", getattr(mtype, 'value', mtype), e)

        # Select primary (first successfully loaded)
        self.primary_type: Optional[ModelType] = next(iter(self.models.keys()), None)
        self._runtime_cfg = {
            "max_tokens":      MAX_TOKENS,          # or whatever default
            "temperature":     TEMPERATURE,
            "top_p":           TOP_P,
            "repeat_penalty":  REPEAT_PENALTY,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty":  PRESENCE_PENALTY,
            "stop":            STOP,
        }

        self.llama: Optional[Llama] = self.models.get(self.primary_type) if self.primary_type else None
        self.active_cfg: Optional[ModelConfig] = self.cfgs.get(self.primary_type) if self.primary_type else None

        if self.llama is None:
            logger.warning("⚠️ No primary model loaded; generate calls will noop.")
        else:
            try:
                eos_id = self.llama.token_eos()
                eos_str = self.llama.detokenize([eos_id]).decode()
                logger.info("EOS token id=%s str=%r", eos_id, eos_str)
            except Exception:
                pass

        # mutable runtime overrides for the primary (max_tokens adjusted per request)
        self.runtime_overrides: Dict[str, Any] = {}

    # Back‑compat: expose a mutable cfg dict like the old loader
    @property
    def cfg(self):
        """
        Backwards compatibility shim.
        Returns a dict with the active (primary) model's runtime sampling params.
        Mutations to temperature/top_p/repeat_penalty/frequency_penalty/presence_penalty/stop/max_tokens
        will affect subsequent generate / generate_stream calls.
        """
        return self._runtime_cfg

    def update_cfg(self, **kwargs):
        self._runtime_cfg.update({k: v for k, v in kwargs.items() if k in self._runtime_cfg})

    # -- internal loader ------------------------------------------------
    def _load_llama_model(self, cfg: ModelConfig) -> Llama:
        if not os.path.exists(cfg.path):
            raise FileNotFoundError(f"Model file not found: {cfg.path}")
        use_gpu = (os.environ.get("USE_CUDA", "false").lower() == "true") or (cfg.gpu_layers > 32 and USE_GPU_DEFAULT)
        return Llama(
            model_path=cfg.path,
            n_ctx=cfg.context_length,
            n_batch=512,
            n_gpu_layers=cfg.gpu_layers if use_gpu else 0,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repetition_penalty,
            stop=cfg.stop,
            verbose=False,
        )

    # -- public access --------------------------------------------------
    def get(self, model_type: ModelType) -> Optional[Llama]:
        if model_type in self.models:
            return self.models[model_type]
        if model_type in self.failed_models:
            logger.warning("Attempt to access failed model %s: %s", getattr(model_type, 'value', model_type), self.failed_models[model_type])
        return None

    def list_models(self) -> List[str]:
        return [getattr(mt, 'value', str(mt)) for mt in self.models.keys()]

    def get_failures(self) -> Dict[str, str]:
        return {getattr(mt, 'value', str(mt)): err for mt, err in self.failed_models.items()}

    # -- runtime cfg helpers --------------------------------------------
    def set_max_tokens(self, n: int):
        self.runtime_overrides['max_tokens'] = int(n)

    def _effective_max_tokens(self) -> int:
        if 'max_tokens' in self.runtime_overrides:
            return int(self.runtime_overrides['max_tokens'])
        if self.active_cfg and getattr(self.active_cfg, 'max_tokens', None):
            return int(self.active_cfg.max_tokens)
        return DEFAULT_MAX_TOKENS

    # ------------------------------------------------------------------ #
    #  Blocking generation (primary model)                              #
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str) -> str:
        if not self.llama:
            return ""
        if DEBUG_BLOCK:
            logger.info("[BLOCK] prompt chars=%d", len(prompt))
        max_toks = self._effective_max_tokens()
        cfg = self.active_cfg
        try:
            out = self.llama(
                prompt=prompt,
                max_tokens=max_toks,
                temperature=cfg.temperature if cfg else 0.7,
                top_p=cfg.top_p if cfg else 0.9,
                repeat_penalty=cfg.repetition_penalty if cfg else 1.1,
                frequency_penalty=getattr(cfg, 'frequency_penalty', 0.0),
                presence_penalty=getattr(cfg, 'presence_penalty', 0.0),
                stop=cfg.stop if cfg else [],
            )
        except Exception as e:
            logger.exception("[LEX] llama blocking generate failed: %s", e)
            return ""

        text = out["choices"][0].get("text", "")

        # Role marker trim
        for marker in ("<|user|>", "<|assistant|>", "<|system|>"):
            if marker in text:
                text = text.split(marker, 1)[0].rstrip()

        # Whole-block half duplicate
        half = len(text) // 2
        if half > 80 and text[:half] == text[half:]:
            text = text[:half].rstrip()

        # Tail mirror
        tail = text[-140:]
        h = len(tail) // 2
        if h >= 40 and tail[:h] == tail[h:]:
            text = text[:-h].rstrip()

        # Paragraph duplicate removal
        seen = set()
        kept = []
        for ln in text.splitlines():
            key = ln.strip().lower()
            if key in seen and len(ln.strip()) > 24:
                break
            seen.add(key)
            kept.append(ln)
        text = "\n".join(kept)

        # NEW: Phrase repetition detection
        if len(text) > PHRASE_REPEAT_MIN_LEN * 2:
            last_phrase = text[-PHRASE_REPEAT_MIN_LEN:]
            if last_phrase in text[:-PHRASE_REPEAT_MIN_LEN]:
                text = text[:-PHRASE_REPEAT_MIN_LEN].rstrip()

        first_line = text.splitlines()[0] if text else ""
        if re.search(r"\bviews\b|\bsubscribe\b|#\w+", first_line.lower()):
            text = "\n".join(text.splitlines()[1:])

        text = re.sub(r"(?<!\n) {2,}", " ", text)
        return text.strip()

    # ------------------------------------------------------------------ #
    #  Streaming generation (primary model)                            #
    # ------------------------------------------------------------------ #
    def generate_stream(self, prompt: str):
        if not self.llama:
            yield ""
            return
        if DEBUG_STREAM:
            logger.info("[STREAM] prompt chars=%d", len(prompt))

        cfg = self.active_cfg
        max_toks = self._effective_max_tokens()
        try:
            acc = ""
            flush_buffer = ""
            micro_tail = ""
            step = 0
            last_yield = time.time()
            sentence_count = 0
            question_count = 0
            metaphors_found = set()
            finished = False

            user_requested_expansion = bool(
                re.search(r"\b(explain|detail|long|list|why|how|story|compare|elaborate|examples?)\b", prompt.lower())
            )

            llama_kwargs = dict(
                prompt=prompt,
                stream=True,
                max_tokens=max_toks,
                temperature=cfg.temperature if cfg else 0.7,
                top_p=cfg.top_p if cfg else 0.9,
                repeat_penalty=cfg.repetition_penalty if cfg else 1.1,
                frequency_penalty=getattr(cfg, 'frequency_penalty', 0.0),
                presence_penalty=getattr(cfg, 'presence_penalty', 0.0),
                stop=cfg.stop if cfg else [],
            )

            for chunk in self.llama(**llama_kwargs):
                choice = chunk["choices"][0]
                tok = choice.get("text") or choice.get("content") or ""
                if not tok:
                    continue

                acc += tok
                flush_buffer += tok
                if DEBUG_STREAM:
                    print(f"[STREAM TOK]{repr(tok)}")

                # Sentence count (cheap)
                sentence_count = len([s for s in _SENT_END_RE.split(acc.strip()) if s])
                if "?" in tok:
                    question_count = acc.count("?")

                # Metaphor detection early
                if len(metaphors_found) <= EARLY_METAPHOR_LIMIT:
                    lower_acc = acc.lower()
                    for mw in list(METAPHOR_WORDS):
                        if f" {mw}" in lower_acc:
                            metaphors_found.add(mw)
                    for mm in _LIKE_AS_RE.findall(acc):
                        metaphors_found.add(mm[1].strip().lower())

                # Visualization
                if ENABLE_VIS:
                    emb = choice.get("embedding")
                    if emb is None:
                        emb = _static_token_embedding(self.llama, tok)
                    if emb is not None:
                        try:
                            vec = np.asarray(emb, dtype=np.float32)
                            prob = float(choice.get("probability", 1.0))
                            send_to_viz(vec, z=step * 0.05, intensity=prob)
                        except Exception:
                            pass
                step += 1

                # Flush timing/size
                if len(flush_buffer) >= FLUSH_LEN or (time.time() - last_yield) > FLUSH_INTERVAL_SEC:
                    yield flush_buffer
                    flush_buffer = ""
                    last_yield = time.time()

                # Micro mirror repetition
                micro_tail = (micro_tail + tok)[-MICRO_MIRROR_MIN_TAIL:]
                half = len(micro_tail) // 2
                if half >= MICRO_MIRROR_MIN_HALF and micro_tail[:half] == micro_tail[half:]:
                    if DEBUG_STREAM: print("[STREAM GUARD] micro mirror -> stop")
                    finished = True
                    break

                # NEW: Phrase repetition detection
                if len(acc) > PHRASE_REPEAT_MIN_LEN * 2:
                    last_phrase = acc[-PHRASE_REPEAT_MIN_LEN:]
                    if last_phrase in acc[:-PHRASE_REPEAT_MIN_LEN]:
                        if DEBUG_STREAM: print("[STREAM GUARD] phrase repeat -> stop")
                        finished = True
                        break

                # Paragraph duplicate
                if len(acc) >= PARA_DUP_MIN_ACC_LEN:
                    paragraphs = [p for p in re.split(r"\n{2,}", acc) if p.strip()]
                    if len(paragraphs) >= 2:
                        last_para = paragraphs[-1].strip()
                        if len(last_para) >= PARA_DUP_MIN_BLOCK_LEN:
                            norm_last = re.sub(r"\s+", " ", last_para.lower())
                            earlier = " ".join(re.sub(r"\s+", " ", p.lower()) for p in paragraphs[:-1])
                            if norm_last in earlier:
                                if DEBUG_STREAM: print("[STREAM GUARD] duplicate paragraph -> stop")
                                finished = True
                                break

                # Question inflation
                if not user_requested_expansion and question_count > QUESTION_OVERFLOW_LIMIT and sentence_count >= 2:
                    if DEBUG_STREAM: print("[STREAM GUARD] question overflow -> stop")
                    finished = True
                    break

                # Metaphor overrun
                if not user_requested_expansion and len(metaphors_found) > EARLY_METAPHOR_LIMIT and sentence_count >= 2:
                    if DEBUG_STREAM: print("[STREAM GUARD] metaphor overrun -> stop")
                    finished = True
                    break

                # Soft sentence cap
                if not user_requested_expansion and sentence_count >= SENTENCE_EARLY_STOP_MAX:
                    if DEBUG_STREAM: print("[STREAM GUARD] sentence cap -> stop")
                    finished = True
                    break

            if flush_buffer:
                yield flush_buffer
            if DEBUG_STREAM and finished:
                print("[STREAM] natural early finish")

        except Exception as e:  # fallback
            logger.exception("[LEX] generate_stream fatal: %s", e)
            yield self.generate(prompt)

# --------------------------------------------------------------------- #
#  Singleton Helper (optional)                                         #
# --------------------------------------------------------------------- #
_model_loader_singleton: Optional[ModelLoader] = None

def initialize_model_loader() -> ModelLoader:
    global _model_loader_singleton
    if _model_loader_singleton is None:
        _model_loader_singleton = ModelLoader()
    return _model_loader_singleton

__all__ = ["ModelLoader", "initialize_model_loader"]
