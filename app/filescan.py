import os
import json
import threading
import time

from app.config import (
    SCAN_ROOT, EXCLUDED_DIR_NAMES, SUPPORTED_EXTENSIONS, CORPUS_PATH,
    MANIFEST_PATH, MAX_FILE_SIZE_MB, RESCAN_INTERVAL_SECONDS
)
from app.indexing import extract_text_from_file, clean_text, chunk_text
from app.retrieval import rebuild_indexes

_scan_lock = threading.Lock()
_scan_status = {
    "status": "idle",   # idle | scanning | done | error
    "files_scanned": 0,
    "files_indexed": 0,
    "chunk_count": 0,
    "last_scan_time": None,
    "error": None,
}


def get_scan_status():
    return dict(_scan_status)


def _load_manifest():
    try:
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def _load_corpus():
    try:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_corpus(corpus):
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2)


def _should_skip_dir(dirname):
    return dirname in EXCLUDED_DIR_NAMES or dirname.startswith(".")


def discover_files(root=SCAN_ROOT):
    """Walk root, yielding (filepath, mtime) for supported, size-ok files —
    pruning excluded directories in-place so os.walk never even descends
    into them (this is what keeps a profile-wide scan fast)."""
    for current_root, dirs, files in os.walk(root, topdown=True):
        dirs[:] = [d for d in dirs if not _should_skip_dir(d)]
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            filepath = os.path.join(current_root, filename)
            try:
                stat = os.stat(filepath)
                if stat.st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    continue
                yield filepath, stat.st_mtime
            except OSError:
                continue  # permission denied / broken link — skip silently


def run_scan(root=SCAN_ROOT):
    """
    Incremental scan: skips files unchanged since the last scan (tracked by
    mtime in manifest.json), so only new/edited files get re-processed.
    This is what makes background rescans fast after the first full pass.
    """
    with _scan_lock:
        _scan_status["status"] = "scanning"
        _scan_status["error"] = None

        manifest = _load_manifest()
        corpus = _load_corpus()
        corpus_by_file = {}
        for chunk in corpus:
            corpus_by_file.setdefault(chunk["source_file"], []).append(chunk)

        new_manifest = {}
        files_scanned = 0
        next_chunk_counter = max(
            (int(c["chunk_id"].split("_")[1]) for c in corpus), default=-1
        ) + 1

        try:
            for filepath, mtime in discover_files(root):
                files_scanned += 1
                new_manifest[filepath] = mtime

                if manifest.get(filepath) == mtime and filepath in corpus_by_file:
                    continue  # unchanged — skip re-extraction

                raw_text = extract_text_from_file(filepath)
                cleaned = clean_text(raw_text)
                if not cleaned:
                    continue

                new_chunks = []
                for i, chunk_text_content in enumerate(chunk_text(cleaned)):
                    new_chunks.append({
                        "chunk_id": f"chunk_{next_chunk_counter}",
                        "text": chunk_text_content,
                        "source_file": filepath,
                        "chunk_index_in_file": i
                    })
                    next_chunk_counter += 1

                corpus_by_file[filepath] = new_chunks

                if files_scanned % 200 == 0:
                    _scan_status["files_scanned"] = files_scanned

            # Drop files that no longer exist on disk
            corpus_by_file = {fp: c for fp, c in corpus_by_file.items() if fp in new_manifest}

            final_corpus = [c for chunks in corpus_by_file.values() for c in chunks]
            _save_corpus(final_corpus)
            _save_manifest(new_manifest)
            rebuild_indexes()

            _scan_status.update({
                "status": "done",
                "files_scanned": files_scanned,
                "files_indexed": len(corpus_by_file),
                "chunk_count": len(final_corpus),
                "last_scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
        except Exception as e:
            _scan_status["status"] = "error"
            _scan_status["error"] = str(e)


def start_background_scanning():
    """Initial scan on startup, then periodic rescans in a daemon thread —
    the user never has to manually trigger indexing again."""
    def loop():
        while True:
            run_scan()
            time.sleep(RESCAN_INTERVAL_SECONDS)
    threading.Thread(target=loop, daemon=True).start()