import json
from rank_bm25 import BM25Okapi
from config import CORPUS_PATH, QUERIES_PATH


def tokenize(text):
    return text.lower().split()


def bm25_search(query, top_k=5):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = json.load(f)

    tokenized_corpus = [tokenize(chunk["text"]) for chunk in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [(chunk["chunk_id"], score) for chunk, score in ranked]


if __name__ == "__main__":
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    test_q = queries[0]
    print(f"Question: {test_q['question']}")
    print(f"Gold chunk_id: {test_q['gold_chunk_id']}")

    for chunk_id, score in bm25_search(test_q["question"]):
        print(chunk_id, score)