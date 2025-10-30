import chromadb
from typing import List, Dict
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/workspace/ai-lab/friday/friday_memory/"
COLLECTION_NAME = "friday_memory"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
_client = chromadb.Client(chromadb.config.Settings(
    persist_directory=CHROMA_PATH
))
_collection = _client.get_or_create_collection(name=COLLECTION_NAME)

def archive_context_to_chroma(entries: List[Dict], session_id: str):
    if not entries:
        return

    documents = [e["text"] for e in entries]
    embeddings = _embed_model.encode(documents).tolist()
    metadatas = [{"metadata": e["metadata"]} for e in entries]
    ids = [e["id"] for e in entries]

    _collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def semantic_search(query: str, k: int = 5) -> List[Dict]:
    query_embedding = _embed_model.encode([query])[0].tolist()
    results = _collection.query(query_embeddings=[query_embedding], n_results=k)

    matches: List[Dict] = []
    for id_, doc, meta in zip(results["ids"][0], results["documents"][0], results["metadatas"][0]):
        matches.append({
            "id": id_,
            "text": doc,
            "metadata": meta.get("metadata", {})
        })
    return matches

