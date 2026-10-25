import os
import re
from pypdf import PdfReader
from docx import Document as DocxDocument
from app.config import SUPPORTED_EXTENSIONS, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS


def extract_text_from_file(filepath):
    """Pull raw text out of a file, based on its extension."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".pdf":
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        elif ext == ".docx":
            doc = DocxDocument(filepath)
            return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        print(f"  WARNING: failed to read {filepath}: {e}")
        return ""
    return ""


def clean_text(text):
    """Collapse whitespace, strip PDF extraction noise."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=CHUNK_SIZE_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """
    Split into overlapping word chunks. Overlap means the tail of chunk N
    reappears at the start of chunk N+1, so info sitting right at a chunk
    boundary doesn't get lost.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def index_path(path):
    """
    Accepts either a single file or a folder. Returns a list of chunk dicts:
    {chunk_id, text, source_file, chunk_index_in_file}
    """
    corpus = []
    chunk_counter = 0

    def process_file(filepath):
        nonlocal chunk_counter
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return
        raw_text = extract_text_from_file(filepath)
        cleaned = clean_text(raw_text)
        if not cleaned:
            print(f"  (skipped — no extractable text): {filepath}")
            return
        for i, chunk in enumerate(chunk_text(cleaned)):
            corpus.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": chunk,
                "source_file": filepath,
                "chunk_index_in_file": i
            })
            chunk_counter += 1

    if os.path.isfile(path):
        process_file(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for filename in files:
                process_file(os.path.join(root, filename))
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

    return corpus