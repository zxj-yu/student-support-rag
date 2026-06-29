"""Tests for the /chat and /health API endpoints.

External services (embeddings, vector store, LLM) are stubbed in conftest.py
so these tests run without Qdrant, sentence-transformers, anthropic, or a key.
"""
from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)


def test_health_returns_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_chat_rejects_empty_question():
    res = client.post("/chat", json={"question": "   "})
    assert res.status_code == 400


def test_chat_returns_answer_and_sources(monkeypatch):
    """A normal question should return the structure the frontend expects."""
    fake = {
        "answer": "You can drop before the DNE deadline.",
        "sources": [{"title": "Withdrawal Policy", "score": 0.91}],
    }
    monkeypatch.setattr(main.retrieval, "answer", lambda q: fake)
    res = client.post("/chat", json={"question": "How do I drop a course?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == fake["answer"]
    assert body["sources"][0]["title"] == "Withdrawal Policy"