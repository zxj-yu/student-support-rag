"""Tests for error handling and edge cases."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.core.config import settings
from app.services import retrieval

client = TestClient(main.app)


# --- /chat input validation ---

def test_chat_rejects_overlong_question(monkeypatch):
    """A pasted novel should be rejected by schema validation (422), not embedded."""
    monkeypatch.setattr(main.store, "count", lambda: 5)
    too_long = "a" * (settings.max_question_chars + 1)
    res = client.post("/chat", json={"question": too_long})
    assert res.status_code == 422


def test_chat_accepts_question_at_limit(monkeypatch):
    monkeypatch.setattr(main.store, "count", lambda: 5)
    monkeypatch.setattr(
        main.retrieval, "answer", lambda q: {"answer": "ok", "sources": []}
    )
    at_limit = "a" * settings.max_question_chars
    res = client.post("/chat", json={"question": at_limit})
    assert res.status_code == 200


# --- empty knowledge base ---

def test_chat_reports_empty_knowledge_base(monkeypatch):
    """Asking before ingesting should say so, not answer from nothing."""
    monkeypatch.setattr(main.store, "count", lambda: 0)
    res = client.post("/chat", json={"question": "How do I drop a course?"})
    assert res.status_code == 409
    assert "empty" in res.json()["detail"].lower()


# --- vector store unreachable ---

def _boom():
    raise ConnectionError("qdrant is down")


def test_chat_reports_vector_store_down(monkeypatch):
    monkeypatch.setattr(main.store, "count", _boom)
    res = client.post("/chat", json={"question": "anything"})
    assert res.status_code == 503
    assert "unavailable" in res.json()["detail"].lower()


def test_health_reports_degraded_when_store_down(monkeypatch):
    """Health must tell the truth when a dependency is down."""
    monkeypatch.setattr(main.store, "count", _boom)
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "degraded"
    assert body["vector_store"] == "unreachable"


def test_health_reports_ok_and_document_count(monkeypatch):
    monkeypatch.setattr(main.store, "count", lambda: 12)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["documents"] == 12
    assert body["knowledge_base"] == "loaded"


def test_health_flags_empty_knowledge_base(monkeypatch):
    monkeypatch.setattr(main.store, "count", lambda: 0)
    body = client.get("/health").json()
    assert body["knowledge_base"] == "empty"


# --- knowledge base file validation ---

def test_ingest_rejects_malformed_json(tmp_path, monkeypatch):
    bad = tmp_path / "kb.json"
    bad.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(retrieval.store, "ensure_collection", lambda: None)
    with pytest.raises(ValueError, match="not valid JSON"):
        retrieval.ingest_knowledge_base(str(bad))


def test_ingest_rejects_empty_array(tmp_path, monkeypatch):
    empty = tmp_path / "kb.json"
    empty.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(retrieval.store, "ensure_collection", lambda: None)
    with pytest.raises(ValueError, match="non-empty"):
        retrieval.ingest_knowledge_base(str(empty))


def test_ingest_rejects_document_missing_fields(tmp_path, monkeypatch):
    bad = tmp_path / "kb.json"
    bad.write_text(json.dumps([{"title": "no text field"}]), encoding="utf-8")
    monkeypatch.setattr(retrieval.store, "ensure_collection", lambda: None)
    with pytest.raises(ValueError, match="'title' and 'text'"):
        retrieval.ingest_knowledge_base(str(bad))


def test_ingest_rejects_empty_document_text(tmp_path, monkeypatch):
    bad = tmp_path / "kb.json"
    bad.write_text(
        json.dumps([{"title": "Blank", "text": "   "}]), encoding="utf-8"
    )
    monkeypatch.setattr(retrieval.store, "ensure_collection", lambda: None)
    with pytest.raises(ValueError, match="empty text"):
        retrieval.ingest_knowledge_base(str(bad))


def test_ingest_missing_file_raises_filenotfound(monkeypatch):
    monkeypatch.setattr(retrieval.store, "ensure_collection", lambda: None)
    with pytest.raises(FileNotFoundError):
        retrieval.ingest_knowledge_base("data/does_not_exist.json")
