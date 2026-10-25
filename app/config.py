import os

# Absolute path to the app/ folder itself — the anchor for everything else
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.json")

# Ollama (local LLM) settings
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_NUM_CTX = 8192       # bigger context window so long chunk sets don't get silently truncated
OLLAMA_NUM_PREDICT = 1200   # allow longer generated answers (default cuts off too early for exam-depth explanations)

# Embedding model for dense retrieval
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# File indexing settings
CHUNK_SIZE_WORDS = 300
CHUNK_OVERLAP_WORDS = 50
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

os.makedirs(DATA_DIR, exist_ok=True)

USER_HOME = os.path.expanduser("~")
SCAN_ROOT = USER_HOME  # practical "whole laptop" = the user's own profile tree

EXCLUDED_DIR_NAMES = {
    "node_modules", "venv", ".venv", "__pycache__", ".git", "AppData",
    "Windows", "Program Files", "Program Files (x86)", "$Recycle.Bin",
    "System Volume Information", ".cache", "site-packages", "Anaconda3",
}

MAX_FILE_SIZE_MB = 20          # skip huge files, keeps scanning fast
RESCAN_INTERVAL_SECONDS = 300  # background rescan every 5 minutes

MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.json")