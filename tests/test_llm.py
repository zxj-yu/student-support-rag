"""Tests for the LLM service's no-key fallback behavior."""
from app.services import llm


def test_fallback_used_when_no_key(monkeypatch):
    """With no API key, generate() returns retrieved context, not an API call."""
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "")
    contexts = [{"title": "Withdrawal Policy", "text": "Drop before DNE.", "score": 0.9}]
    result = llm.generate("How do I drop a course?", contexts)
    assert "No LLM key set" in result
    assert "Withdrawal Policy" in result
    assert "Drop before DNE." in result


def test_fallback_handles_empty_contexts(monkeypatch):
    """No retrieved documents should produce a clear message, not a crash."""
    monkeypatch.setattr(llm.settings, "anthropic_api_key", "")
    result = llm.generate("anything", [])
    assert "No relevant documents" in result


def test_build_prompt_includes_all_sources():
    """The prompt builder should include every provided context block."""
    contexts = [
        {"title": "A", "text": "alpha"},
        {"title": "B", "text": "beta"},
    ]
    prompt = llm.build_prompt("q", contexts)
    assert "alpha" in prompt and "beta" in prompt
    assert "Question: q" in prompt