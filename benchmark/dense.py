import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from config import CORPUS_PATH, QUERIES_PATH, EMBEDDING_MODEL_NAME

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def dense_search(query, top_k=5):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    if not corpus:
        return []

    chunk_texts = [chunk["text"] for chunk in corpus]
    chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)

    row_to_chunk_id = [chunk["chunk_id"] for chunk in corpus]

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    k = min(top_k, len(corpus))
    scores, indices = index.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((row_to_chunk_id[idx], float(score)))
    return results


if __name__ == "__main__":
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    test_q = queries[0]
    print(f"\nTest question: {test_q['question']}")
    print(f"Gold chunk_id: {test_q['gold_chunk_id']}")

    print("\nDense top 5 results:")
    for chunk_id, score in dense_search(test_q["question"]):
        print(f"  {chunk_id} (score: {score:.4f})")