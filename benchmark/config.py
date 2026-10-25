import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BENCHMARK_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.json")
RESULTS_CSV_PATH = os.path.join(BENCHMARK_DIR, "results.csv")
RECALL_CHART_PATH = os.path.join(BENCHMARK_DIR, "recall_comparison.png")
MRR_CHART_PATH = os.path.join(BENCHMARK_DIR, "mrr_comparison.png")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)