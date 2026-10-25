<div align="center">

# 🧠 Local RAG Study Assistant

### Chat with your own documents — privately, offline, and for free.

*Point it at a PDF or folder. Ask anything. Get exam-ready explanations — grounded in your own notes, generated entirely on your machine.*

</div>

---

## 📌 Why this exists

Most RAG tutorials just pick a retrieval method and hope for the best. This project didn't.

Before writing a single line of the actual app, we ran a **controlled experiment** comparing three retrieval strategies — keyword search, semantic search, and a hybrid of both — on 150 labeled questions, measured the results, and *then* built the app on the winner. No guessing, no hand-waving — just evidence.

> 💡 **TL;DR**: Hybrid retrieval won. By a lot. See the numbers below.

---

## 📊 The Research: Which Retrieval Method Actually Wins?

| Method | Recall@1 | Recall@5 | MRR |
|:--|:--:|:--:|:--:|
| 🔤 BM25 (keyword) | 78.67% | 92.00% | 0.8446 |
| 🧬 Dense (semantic) | 82.67% | 94.67% | 0.8870 |
| 🏆 **Hybrid (RRF)** | **86.67%** | **98.67%** | **0.9112** |

**Hybrid retrieval beat both individual methods on every single metric** — an 8-point jump in top-1 accuracy over keyword search alone. That's the retrieval engine powering this app, not a guess.

📁 Full methodology, charts, and raw results live in [`/benchmark`](./benchmark).

---

## ✨ What It Does

- 📂 **Index anything** — PDF, DOCX, TXT, or Markdown. Point it at a file or a whole folder.
- 🔍 **Hybrid retrieval** — BM25 + dense embeddings, fused with Reciprocal Rank Fusion, so it finds the right passage whether you use exact keywords or paraphrase.
- 🤖 **Local LLM** — answers generated entirely on-device via Ollama. No API keys. No per-query cost. No data ever leaves your laptop.
- 💬 **Real conversation memory** — multi-chat sidebar, follow-up questions that actually understand "it" and "that."
- 🎓 **Built for studying** — explains concepts in depth rather than spitting out terse snippets, and never just says "I don't know" when there's material to work with.

---

## 🏗️ Architecture

```
┌──────────────┐      ┌─────────────┐      ┌──────────────────────┐      ┌───────────┐
│  Chat UI      │ ───▶ │  FastAPI     │ ───▶ │  Hybrid Retriever      │ ───▶ │  Ollama    │
│  (HTML/JS)    │ ◀─── │  Backend     │ ◀─── │  (BM25 + FAISS + RRF)  │ ◀─── │  (local LLM)│
└──────────────┘      └──────┬──────┘      └──────────────────────┘      └───────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │  SQLite      │
                        │  (chat memory)│
                        └─────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Backend | FastAPI, Python |
| Retrieval | `rank_bm25`, `sentence-transformers` (`all-MiniLM-L6-v2`), FAISS |
| Generation | Ollama (`llama3.2:3b`) |
| Memory | SQLite |
| File parsing | `pypdf`, `python-docx` |
| Frontend | Vanilla HTML/CSS/JS — zero build step |

---

## 📁 Project Structure

```
rag/
├── app/                   # The live application
│   ├── config.py           # Paths, model names, chunk settings
│   ├── indexing.py         # File scanning + extraction + chunking
│   ├── retrieval.py        # BM25 + dense + hybrid search (cached)
│   ├── generation.py       # Prompt building + Ollama calls
│   ├── memory.py           # SQLite conversation storage
│   └── main.py              # FastAPI routes
│
├── benchmark/               # The research behind the retrieval choice
│   ├── prepare_data.py
│   ├── bm25.py / dense.py / hybrid.py
│   ├── evaluate.py
│   └── results.csv, *.png
│
└── frontend/
    └── index.html            # Chat UI with multi-conversation sidebar
```

---

## 🚀 Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Install Ollama + pull a model**

Download from [ollama.com](https://ollama.com), then:
```bash
ollama pull llama3.2:3b
```

**3. Start the backend**
```bash
uvicorn app.main:app --reload --port 8000
```

**4. Open the frontend**

Just open `frontend/index.html` in your browser. No server, no build step.

**5. Index a document and start chatting**

Paste a file or folder path into the index bar → click **Index** → ask away.

---

## 🔬 Reproduce the Benchmark

```bash
cd benchmark
python prepare_data.py
python evaluate.py
```

Outputs `results.csv`, `recall_comparison.png`, and `mrr_comparison.png`.

---

## 📄 License

MIT — use it, fork it, learn from it.

<div align="center">

**Built for studying smarter, not harder. 📚**

</div>
