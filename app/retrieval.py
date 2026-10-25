import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from app.config import CORPUS_PATH, EMBEDDING_MODEL_NAME

print("Loading embedding model (one-time)...")
_embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# ---- Module-level cache ----
_corpus = []
_bm25 = None
_faiss_index = None
_row_to_chunk_id = []


def _tokenize(text):
    return text.lower().split()


def rebuild_indexes():
    """Reload corpus.json and rebuild BM25 + FAISS indexes from scratch.
    Handles a missing file AND an empty/corrupt file without crashing."""
    global _corpus, _bm25, _faiss_index, _row_to_chunk_id

    try:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            content = f.read().strip()
            _corpus = json.loads(content) if content else []
    except (FileNotFoundError, json.JSONDecodeError):
        _corpus = []

    if not _corpus:
        _bm25 = None
        _faiss_index = None
        _row_to_chunk_id = []
        print("Indexes cleared — corpus is empty.")
        return

    print(f"Building BM25 index over {len(_corpus)} chunks...")
    tokenized_corpus = [_tokenize(c["text"]) for c in _corpus]
    _bm25 = BM25Okapi(tokenized_corpus)

    print("Building FAISS index...")
    chunk_texts = [c["text"] for c in _corpus]
    embeddings = _embedding_model.encode(chunk_texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    _faiss_index = faiss.IndexFlatIP(dimension)
    _faiss_index.add(embeddings)

    _row_to_chunk_id = [c["chunk_id"] for c in _corpus]
    print("Indexes ready.")


def get_corpus_lookup():
    """chunk_id -> full chunk dict, for pulling text/source when generating answers."""
    return {c["chunk_id"]: c for c in _corpus}


def bm25_search(query, top_k=5):
    if _bm25 is None:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(_corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(c["chunk_id"], float(score)) for c, score in ranked]


def dense_search(query, top_k=5):
    if _faiss_index is None:
        return []
    query_vec = _embedding_model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k, len(_corpus))
    scores, indices = _faiss_index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((_row_to_chunk_id[idx], float(score)))
    return results


def hybrid_search(query, top_k=5, k_constant=60, candidate_pool=20):
    """Reciprocal Rank Fusion of BM25 + dense rankings."""
    bm25_results = bm25_search(query, top_k=candidate_pool)
    dense_results = dense_search(query, top_k=candidate_pool)

    rrf_scores = {}
    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank)
    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


# Build indexes once at import time — handles missing/empty corpus.json internally
rebuild_indexes()