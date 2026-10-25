from bm25 import bm25_search
from dense import dense_search


def hybrid_search(query, top_k=5, k_constant=60, candidate_pool=20):
    """
    Combines BM25 and dense rankings using Reciprocal Rank Fusion (RRF).
    Uses rank position (not raw scores) since BM25 and cosine-similarity
    scores live on incomparable scales.
    """
    bm25_results = bm25_search(query, top_k=candidate_pool)
    dense_results = dense_search(query, top_k=candidate_pool)

    rrf_scores = {}
    for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank)
    for rank, (chunk_id, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k_constant + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked


if __name__ == "__main__":
    import json
    from config import QUERIES_PATH

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    test_q = queries[0]
    print(f"\nTest question: {test_q['question']}")
    print(f"Gold chunk_id: {test_q['gold_chunk_id']}")

    print("\nHybrid (RRF) top 5 results:")
    for chunk_id, score in hybrid_search(test_q["question"]):
        print(f"  {chunk_id}  (RRF score: {score:.4f})")