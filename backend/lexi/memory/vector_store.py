"""
Chroma-backed vector stubs (opt-in, off by default).

This module is intentionally lazy: no heavy imports or model loads unless
LEXI_VECTOR_ENABLED=1. It keeps the interface ready for future rollout while
remaining a no-op today.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from ..utils.user_identity import normalize_user_id

log = logging.getLogger("lexi.vector")

DEFAULT_CHROMA_PATH = Path(
    os.getenv("LEXI_VECTOR_CHROMA_PATH", "/workspace/ai-lab/Lex/vector_store")
)
DEFAULT_COLLECTION = os.getenv("LEXI_VECTOR_COLLECTION", "lex_memory")
EMBED_MODEL_NAME = os.getenv("LEXI_VECTOR_EMBED_MODEL", "all-MiniLM-L6-v2")
_ENABLED = os.getenv("LEXI_VECTOR_ENABLED", "0").lower() in {"1", "true", "yes", "on"}

# Lazy singletons
_client = None
_collection = None
_embed_model = None


def vector_feature_enabled() -> bool:
    return _ENABLED


def _lazy_imports():
    import chromadb  # type: ignore

    return chromadb


def _embedding_model():
    global _embed_model
    if _embed_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            log.warning("Vector embeddings unavailable (sentence_transformers import failed): %s", exc)
            return None
        try:
            _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as exc:  # pragma: no cover - model load guard
            log.warning("Vector embeddings unavailable (model load failed): %s", exc)
            _embed_model = None
    return _embed_model


def _get_collection() -> Tuple[Optional[object], Optional[object]]:
    """Return (collection, client) or (None, None) if disabled/unavailable."""
    global _client, _collection
    if not vector_feature_enabled():
        return None, None
    if _collection is not None and _client is not None:
        return _collection, _client

    try:
        chromadb = _lazy_imports()
    except Exception as exc:  # pragma: no cover - import guard
        log.warning("Chroma unavailable: %s", exc)
        return None, None

    try:
        DEFAULT_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.Client(
            chromadb.config.Settings(persist_directory=str(DEFAULT_CHROMA_PATH))
        )
        _collection = _client.get_or_create_collection(
            name=DEFAULT_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        log.warning("Chroma client unavailable: %s", exc)
        _client = None
        _collection = None
        return None, None
    return _collection, _client


def _namespace(user_id: Optional[str]) -> str:
    return normalize_user_id(user_id) or "shared"


def archive_context_to_chroma(
    entries: Iterable[Dict], session_id: str, *, user_id: Optional[str] = None
) -> Dict[str, object]:
    """
    Best-effort append of contextual chunks to Chroma. No-ops when disabled.
    """
    collection, _ = _get_collection()
    if not collection:
        return {"ok": False, "stored": 0, "enabled": False}

    items = [e for e in entries if isinstance(e, dict) and e.get("text")]
    if not items:
        return {"ok": False, "stored": 0, "enabled": True}

    docs = [str(e["text"]) for e in items]
    ids = [str(e.get("id") or f"{session_id}-{i}") for i, _ in enumerate(items)]
    metas = []
    for e in items:
        md = dict(e.get("metadata") or {})
        md.setdefault("session_id", session_id)
        md.setdefault("user_id", _namespace(user_id))
        metas.append(md)

    model = _embedding_model()
    if model is None:
        return {"ok": False, "stored": 0, "enabled": True, "reason": "embed_model_unavailable"}

    try:
        embeddings = model.encode(docs, normalize_embeddings=True).tolist()
        collection.add(documents=docs, embeddings=embeddings, metadatas=metas, ids=ids)
        return {"ok": True, "stored": len(docs), "enabled": True}
    except Exception as exc:  # pragma: no cover - runtime guard
        log.warning("Chroma append failed: %s", exc)
        return {"ok": False, "stored": 0, "enabled": True, "reason": str(exc)}


def semantic_search(
    query: str, k: int = 5, *, user_id: Optional[str] = None
) -> List[Dict[str, object]]:
    """
    Vector search with optional user scoping. Returns [] when disabled/unavailable.
    """
    collection, _ = _get_collection()
    if not collection:
        return []

    model = _embedding_model()
    if model is None:
        return []

    try:
        q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, k),
            where={"user_id": _namespace(user_id)},
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        log.warning("Chroma query failed: %s", exc)
        return []

    matches: List[Dict[str, object]] = []
    ids = results.get("ids") or [[]]
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    for id_, doc, meta in zip(ids[0], docs[0], metas[0]):
        matches.append(
            {
                "id": id_,
                "text": doc,
                "metadata": dict(meta or {}),
            }
        )
    return matches


__all__ = [
    "archive_context_to_chroma",
    "semantic_search",
    "vector_feature_enabled",
    "DEFAULT_CHROMA_PATH",
    "DEFAULT_COLLECTION",
    "EMBED_MODEL_NAME",
]
