import sqlite3
import os
from datetime import datetime
from app.config import DATA_DIR

DB_PATH = os.path.join(DATA_DIR, "memory.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()


def create_conversation(title="New chat"):
    conn = _conn()
    cur = conn.execute(
        "INSERT INTO conversations (title, created_at) VALUES (?, ?)",
        (title, datetime.utcnow().isoformat())
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def list_conversations():
    conn = _conn()
    rows = conn.execute("SELECT id, title, created_at FROM conversations ORDER BY id DESC").fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def rename_conversation(conv_id, title):
    conn = _conn()
    conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))
    conn.commit()
    conn.close()


def delete_conversation(conv_id):
    conn = _conn()
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()


def save_message(conversation_id, role, content, sources=None):
    conn = _conn()
    sources_str = ",".join(sources) if sources else None
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, sources_str, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def get_recent_history(conversation_id, limit=6):
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, limit)
    ).fetchall()
    conn.close()
    return list(reversed(rows))


def get_conversation_messages(conversation_id):
    conn = _conn()
    rows = conn.execute(
        "SELECT role, content, sources, timestamp FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (conversation_id,)
    ).fetchall()
    conn.close()
    return [{"role": r, "content": c, "sources": s.split(",") if s else [], "timestamp": t} for r, c, s, t in rows]


init_db()