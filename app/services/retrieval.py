"""Retrieval pipeline: ingest documents and answer questions (RAG)."""
import json
from pathlib import Path

from app.core.config import settings
from app.services import embeddings, llm
from app.services.vector_store import store


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    """Naive paragraph-aware chunker. Good enough for Week 1; can be
    upgraded to token-based splitting later."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks or [text]


def ingest_knowledge_base(path: str = "data/knowledge_base.json") -> int:
    """Load documents, chunk, embed, and upsert into Qdrant.

    Each document is a dict: {"title": ..., "text": ...}
    Returns the number of chunks ingested.
    """
    store.ensure_collection()
    docs = json.loads(Path(path).read_text(encoding="utf-8"))

    texts: list[str] = []
    payloads: list[dict] = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            texts.append(chunk)
            payloads.append({"title": doc["title"], "text": chunk})

    vectors = embeddings.embed(texts)
    return store.upsert(vectors, payloads)


def answer(question: str) -> dict:
    """Full RAG: embed query -> retrieve -> generate."""
    query_vec = embeddings.embed_one(question)
    contexts = store.search(query_vec, top_k=settings.top_k)
    response = llm.generate(question, contexts)
    return {
        "answer": response,
        "sources": [
            {"title": c.get("title"), "score": round(c.get("score", 0), 3)}
            for c in contexts
        ],
    }
