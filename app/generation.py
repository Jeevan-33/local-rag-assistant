import json
import requests
from app.config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT
from app.retrieval import hybrid_search, get_corpus_lookup

OVERVIEW_PHRASES = [
    "explain this pdf", "explain this document", "explain this file",
    "explain the pdf", "explain the notes", "explain this notes",
    "summarize this", "summary of this", "about this pdf", "about this document",
    "what is this pdf about", "what is this document about", "overview",
    "explain everything", "explain all of this", "explain in detail",
]


def is_overview_request(question):
    q = question.lower()
    return any(phrase in q for phrase in OVERVIEW_PHRASES)


def get_all_chunks_ordered(corpus_lookup, limit=25):
    """For 'explain this whole pdf' style requests: return chunks in document
    order (not relevance-ranked, since a vague question won't rank well),
    capped so we don't blow past the model's context window."""
    chunks = list(corpus_lookup.values())
    chunks.sort(key=lambda c: (c.get("source_file", ""), c.get("chunk_index_in_file", 0)))
    return [(c["chunk_id"], 1.0) for c in chunks[:limit]]


def build_prompt(question, retrieved_chunks, corpus_lookup, conversation_history=None, overview_mode=False):
    context_blocks = [corpus_lookup[cid]["text"] for cid, _ in retrieved_chunks if cid in corpus_lookup]
    context_text = "\n\n".join(context_blocks)

    history_text = ""
    if conversation_history:
        turns = []
        for role, content in conversation_history:
            speaker = "User" if role == "user" else "Assistant"
            turns.append(f"{speaker}: {content}")
        history_text = "Previous conversation:\n" + "\n".join(turns) + "\n\n"

    instructions = """You are a study assistant helping a student prepare for exams.

Rules:
- Explain the material in detail and in simple, clear language — the student needs to understand it well enough to write exam answers.
- Use the context below as your source material, but elaborate on it: explain concepts, give examples, break down anything technical step by step, the way a teacher would.
- Do not just copy sentences from the context — teach the concept.
- Never say "I don't have enough information to answer that." Work with what is in the context, explain it as thoroughly as possible, and elaborate using your own understanding of the subject where it helps clarify the material.
- Structure longer answers with short paragraphs or bullet points so they're easy to study from."""

    return f"""{instructions}

{history_text}Context from the student's notes/PDF:
{context_text}

Question: {question}

Answer:"""


def generate_answer(question, top_k=5, min_score_threshold=0.01, conversation_history=None):
    corpus_lookup = get_corpus_lookup()

    if not corpus_lookup:
        return {
            "answer": "No documents are indexed yet — index a file or folder first.",
            "sources": [],
            "retrieved_chunk_ids": []
        }

    overview_mode = is_overview_request(question)

    if overview_mode:
        retrieved = get_all_chunks_ordered(corpus_lookup, limit=25)
    else:
        if conversation_history:
            recent_text = " ".join(content for role, content in conversation_history[-2:])
            retrieval_query = f"{recent_text} {question}"
        else:
            retrieval_query = question

        retrieved = hybrid_search(retrieval_query, top_k=top_k)
        # If normal retrieval comes back weak, fall back to a broad context
        # instead of refusing — this is a study tool, not a strict QA system.
        if not retrieved or retrieved[0][1] < min_score_threshold:
            retrieved = get_all_chunks_ordered(corpus_lookup, limit=15)

    prompt = build_prompt(question, retrieved, corpus_lookup, conversation_history, overview_mode)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": OLLAMA_NUM_PREDICT,
            }
        }
    )
    response.raise_for_status()
    answer = response.json()["response"]

    sources = list({
        corpus_lookup[cid]["source_file"]
        for cid, _ in retrieved if cid in corpus_lookup
    })

    return {
        "answer": answer.strip(),
        "sources": sources,
        "retrieved_chunk_ids": [cid for cid, _ in retrieved]
    }