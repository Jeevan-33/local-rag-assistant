from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json
import requests

from app.config import CORPUS_PATH
from app.indexing import index_path
from app.retrieval import rebuild_indexes
from app.generation import generate_answer
from app.memory import (
    create_conversation, list_conversations, rename_conversation,
    delete_conversation, save_message, get_recent_history, get_conversation_messages
)

app = FastAPI(title="Local RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    question: str
    conversation_id: int


class NewConversationRequest(BaseModel):
    title: Optional[str] = None


class RenameRequest(BaseModel):
    title: str


@app.get("/")
def root():
    return {"message": "RAG Assistant API is running. See /docs for endpoints."}


@app.get("/status")
def status():
    try:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            content = f.read().strip()
            corpus = json.loads(content) if content else []
        return {"indexed": len(corpus) > 0, "chunk_count": len(corpus)}
    except FileNotFoundError:
        return {"indexed": False, "chunk_count": 0}


@app.post("/index")
def index_endpoint(req: IndexRequest):
    clean_path = req.path.strip().strip('"').strip("'")
    try:
        corpus = index_path(clean_path)
        with open(CORPUS_PATH, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2)
        rebuild_indexes()
        return {"success": True, "indexed": True, "chunk_count": len(corpus)}
    except FileNotFoundError:
        return {"success": False, "error": f"Path not found: {clean_path}"}
    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {str(e)}"}


@app.get("/conversations")
def get_conversations():
    return {"conversations": list_conversations()}


@app.post("/conversations")
def new_conversation(req: NewConversationRequest):
    return {"id": create_conversation(req.title or "New chat")}


@app.get("/conversations/{conv_id}/messages")
def conversation_messages(conv_id: int):
    return {"messages": get_conversation_messages(conv_id)}


@app.put("/conversations/{conv_id}")
def rename_conv(conv_id: int, req: RenameRequest):
    rename_conversation(conv_id, req.title)
    return {"renamed": True}


@app.delete("/conversations/{conv_id}")
def delete_conv(conv_id: int):
    delete_conversation(conv_id)
    return {"deleted": True}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        history = get_recent_history(req.conversation_id, limit=6)
        is_first_message = len(history) == 0

        result = generate_answer(req.question, conversation_history=history)

        save_message(req.conversation_id, "user", req.question)
        save_message(req.conversation_id, "assistant", result["answer"], sources=result["sources"])

        if is_first_message:
            title = req.question[:40] + ("..." if len(req.question) > 40 else "")
            rename_conversation(req.conversation_id, title)

        result["success"] = True
        return result
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Could not reach Ollama. Run: ollama run llama3.2:3b"}
    except Exception as e:
        return {"success": False, "error": f"Chat failed: {str(e)}"}