import json
import random
from datasets import load_dataset
from config import CORPUS_PATH, QUERIES_PATH

random.seed(42)  # reproducibility — same subset every run

print("Loading SQuAD validation split...")
squad = load_dataset("rajpurkar/squad", split="validation")

# Step 1: Deduplicate contexts -> build corpus
print("Building deduplicated corpus...")
context_to_id = {}
corpus = []

for row in squad:
    ctx = row["context"]
    if ctx not in context_to_id:
        chunk_id = f"chunk_{len(corpus)}"
        context_to_id[ctx] = chunk_id
        corpus.append({"chunk_id": chunk_id, "text": ctx})

print(f"Total unique chunks in full corpus: {len(corpus)}")

# Step 2: Build query set with ground-truth chunk_id
print("Building query set...")
queries = []
for row in squad:
    chunk_id = context_to_id[row["context"]]
    queries.append({
        "question": row["question"],
        "gold_chunk_id": chunk_id
    })

# Step 3: Subsample to a manageable size
sampled_corpus = random.sample(corpus, min(300, len(corpus)))
sampled_ids = {c["chunk_id"] for c in sampled_corpus}

sampled_queries = [q for q in queries if q["gold_chunk_id"] in sampled_ids]
sampled_queries = random.sample(sampled_queries, min(150, len(sampled_queries)))

print(f"Final corpus size: {len(sampled_corpus)} chunks")
print(f"Final query set size: {len(sampled_queries)} questions")

with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled_corpus, f, indent=2)

with open(QUERIES_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled_queries, f, indent=2)

print(f"Saved {CORPUS_PATH} and {QUERIES_PATH}")