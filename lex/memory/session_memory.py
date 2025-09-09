# lex/memory/session_memory.py

import os
import re
import json
import time
from typing import Callable, List, Dict, Optional
from ..utils.emotion_core import infer_emotion  # import at top if not already

STOPWORDS = set("""
a an the and or but if then else for of on in to from with without by as at is are was were be been being
i you he she it we they me him her us them my your his her its our their mine yours ours theirs
this that these those here there now today yesterday tomorrow
""".split())

def _sentences(text: str) -> List[str]:
    # super cheap splitter
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def _keywords(text: str, k: int = 8) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
    freq: Dict[str,int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

def _emotion_label(avg: Dict[str, float]) -> str:
    # crude but fast: pick top 1–2 axes that are high
    if not avg: return "neutral"
    top = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)[:2]
    tags = [f"{k}:{v:.2f}" for k,v in top]
    return ", ".join(tags)

def cheap_compact_summary(oldest_chunk: List[Dict]) -> str:
    """
    Deterministic compaction of many turns => short continuity blurb.
    No model usage. Runs on CPU fast.
    """
    # stitch summaries (prefer) else user lines
    texts: List[str] = []
    for e in oldest_chunk:
        s = (e.get("summary") or "").strip()
        if not s:
            s = (e.get("user") or "").strip()
        if s:
            texts.append(s)

    joined = " ".join(texts)
    keys = _keywords(joined, k=7)
    sents = _sentences(joined)

    # keep first and last substantive sentences
    picks = []
    if sents:
        picks.append(sents[0])
    if len(sents) > 1:
        picks.append(sents[-1])

    # add 1–2 middle sentences that contain frequent keywords
    middle = sents[1:-1] if len(sents) > 2 else []
    scored = []
    keyset = set(keys)
    for i, s in enumerate(middle):
        score = sum(1 for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", s.lower()) if w in keyset)
        if score:
            scored.append((score, i, s))
    scored.sort(reverse=True)
    for _, _, s in scored[:2]:
        picks.append(s)

    # build final compact block
    parts = []
    if keys:
        parts.append("Earlier topics: " + ", ".join(keys) + ".")
    if picks:
        parts += picks

    compact = "\n".join(parts).strip()
    # clamp length (very short)
    if len(compact) > 900:
        compact = compact[:900].rsplit(".", 1)[0] + "."

    return compact or "[earlier conversation summary]"
    
class SessionMemoryManager:
    """
    Handles session-based working memory as summarized turn pairs,
    with session reset and buffer pruning.
    """
    def __init__(self, session_path: str = None, max_pairs: int = 20):
        folder = os.path.dirname(os.path.abspath(__file__))
        self.session_path = session_path or os.path.join(folder, "session_memory.json")
        self.max_pairs = max_pairs
        self.session_id = self._new_session_id()
        self.buffer: List[Dict] = []
        self._load()

    def _new_session_id(self) -> str:
        # Use timestamp for now; could use uuid.uuid4() if you prefer
        return str(int(time.time()))

    def reset(self):
        """Clears current session memory and assigns new session_id."""
        self.buffer = []
        self.session_id = self._new_session_id()
        if os.path.exists(self.session_path):
            os.remove(self.session_path)

    def add_pair(self, user_msg: str, ai_msg: str, summary: str, token_counter=None):
        emotion = infer_emotion(user_msg)  # NEW
        entry = {
            "session_id": self.session_id,
            "user": user_msg,
            "ai": ai_msg,
            "summary": summary.strip(),
            "emotion": emotion               # NEW
        }
        if token_counter:
            entry["_tok_user"] = token_counter(user_msg)
            entry["_tok_ai"] = token_counter(ai_msg)
            entry["_tok_sum"] = token_counter(summary.strip())
        self.buffer.append(entry)

        if len(self.buffer) > self.max_pairs:
            self.buffer = self.buffer[-self.max_pairs:]
        self._save()
    
    def total_tokens(self) -> int:
        """Fast token count using cached values if present."""
        total = 0
        for entry in self.buffer:
            if "_tok_user" in entry:
                total += entry["_tok_user"] + entry["_tok_ai"] + entry["_tok_sum"]
            else:
                # fallback if not cached
                total += len(entry["user"])//4 + len(entry["ai"])//4 + len(entry["summary"])//4
        return total

    def compact_oldest(self, max_keep_tokens: int):
        """
        Compact oldest entries into a single summary when token budget is exceeded.
        Uses cheap_compact_summary (no model).
        """
        def _total_tokens_entry(e: Dict) -> int:
            if "_tok_user" in e:
                return e.get("_tok_user",0)+e.get("_tok_ai",0)+e.get("_tok_sum",0)
            # fallback estimate
            return len(e.get("user",""))//4 + len(e.get("ai",""))//4 + len(e.get("summary",""))//4

        def _total_tokens_all(buf: List[Dict]) -> int:
            return sum(_total_tokens_entry(e) for e in buf)

        # only compact if we’re actually over budget
        if _total_tokens_all(self.buffer) <= max_keep_tokens or len(self.buffer) < 2:
            return

        # choose an oldest slice to compact; keep last (max_pairs-1) items
        oldest_chunk = self.buffer[:- (self.max_pairs - 1)] if self.max_pairs > 1 else self.buffer[:-1]
        if not oldest_chunk:
            return

        # summarize text cheaply
        combined = cheap_compact_summary(oldest_chunk)

        # merge emotions by averaging
        merged_emotion: Dict[str, float] = {}
        cnt = 0
        for e in oldest_chunk:
            emo = e.get("emotion", {})
            if not isinstance(emo, dict): 
                continue
            for k, v in emo.items():
                merged_emotion[k] = merged_emotion.get(k, 0.0) + float(v)
            cnt += 1
        if cnt:
            for k in list(merged_emotion.keys()):
                merged_emotion[k] /= cnt

        # include a tiny tag of mood in the summary text so the model sees it
        mood_tag = _emotion_label(merged_emotion)
        if mood_tag and "[mood:" not in combined.lower():
            combined = f"[mood: {mood_tag}]\n{combined}"

        # replace oldest chunk with one compact entry
        compact_entry = {
            "session_id": self.session_id,
            "user": "[earlier conversation]",
            "ai": "",
            "summary": combined.strip(),
            "emotion": merged_emotion,
            "_tok_user": 5,  # tiny placeholders; avoids recounting later
            "_tok_ai": 0,
            "_tok_sum": max(1, len(combined)//4)
        }
        self.buffer = [compact_entry] + self.buffer[-(self.max_pairs - 1):]
        self._save()


    def _save(self):
        data = {
            "session_id": self.session_id,
            "buffer": self.buffer
        }
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if os.path.exists(self.session_path):
            try:
                with open(self.session_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.session_id = data.get("session_id", self._new_session_id())
                    self.buffer = data.get("buffer", [])
            except Exception:
                self.buffer = []
        else:
            self.buffer = []

    def get_recent_summaries(self, max_pairs: Optional[int] = None) -> List[str]:
        """Returns up to max_pairs summaries from this session."""
        n = max_pairs if max_pairs is not None else self.max_pairs
        return [entry["summary"] for entry in self.buffer[-n:]]

    def percent_full(self, token_counter, max_tokens: int) -> float:
        """Returns percent of max_tokens used by session summaries."""
        total_tokens = sum(token_counter(s) for s in self.get_recent_summaries())
        return total_tokens / max_tokens if max_tokens else 0.0

    def clear(self):
        """Alias for reset (for external calls)."""
        self.reset()

