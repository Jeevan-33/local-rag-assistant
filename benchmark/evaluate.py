import json
import pandas as pd
import matplotlib.pyplot as plt
from bm25 import bm25_search
from dense import dense_search
from hybrid import hybrid_search
from config import QUERIES_PATH, RESULTS_CSV_PATH, RECALL_CHART_PATH, MRR_CHART_PATH

with open(QUERIES_PATH, encoding="utf-8") as f:
    queries = json.load(f)

K_VALUES = [1, 3, 5, 10]
MAX_K = max(K_VALUES)


def get_gold_rank(results, gold_chunk_id):
    for rank, (chunk_id, _) in enumerate(results, start=1):
        if chunk_id == gold_chunk_id:
            return rank
    return None


methods = {
    "BM25": lambda q: bm25_search(q, top_k=MAX_K),
    "Dense": lambda q: dense_search(q, top_k=MAX_K),
    "Hybrid": lambda q: hybrid_search(q, top_k=MAX_K),
}

print(f"Running {len(queries)} queries through {len(methods)} retrieval methods...\n")

all_ranks = {name: [] for name in methods}

for i, q in enumerate(queries):
    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/{len(queries)} queries...")
    for name, search_fn in methods.items():
        results = search_fn(q["question"])
        rank = get_gold_rank(results, q["gold_chunk_id"])
        all_ranks[name].append(rank)

print("\nDone retrieving. Computing metrics...\n")


def compute_metrics(ranks):
    n = len(ranks)
    metrics = {}
    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        metrics[f"Recall@{k}"] = hits / n
        metrics[f"Precision@{k}"] = (hits / n) / k
    reciprocal_ranks = [1 / r if r is not None else 0 for r in ranks]
    metrics["MRR"] = sum(reciprocal_ranks) / n
    return metrics


results_table = {name: compute_metrics(ranks) for name, ranks in all_ranks.items()}
df = pd.DataFrame(results_table).T
df = df.round(4)

print(df.to_string())
df.to_csv(RESULTS_CSV_PATH)
print(f"\nSaved {RESULTS_CSV_PATH}")

plt.figure(figsize=(8, 5))
for name in methods:
    recalls = [results_table[name][f"Recall@{k}"] for k in K_VALUES]
    plt.plot(K_VALUES, recalls, marker="o", label=name)
plt.xlabel("k")
plt.ylabel("Recall@k (Hit Rate)")
plt.title("Retrieval Method Comparison: Recall@k")
plt.xticks(K_VALUES)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(RECALL_CHART_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {RECALL_CHART_PATH}")

plt.figure(figsize=(6, 5))
names = list(methods.keys())
mrr_values = [results_table[name]["MRR"] for name in names]
plt.bar(names, mrr_values, color=["#4C72B0", "#55A868", "#C44E52"])
plt.ylabel("MRR")
plt.title("Mean Reciprocal Rank by Method")
plt.ylim(0, 1)
for i, v in enumerate(mrr_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.savefig(MRR_CHART_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {MRR_CHART_PATH}")