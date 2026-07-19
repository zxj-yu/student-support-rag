"""Tests for the agent layer: no-key fallback, tool dispatch, and the loop.

The Anthropic client is stubbed (see conftest.py), so these tests exercise
the loop and dispatch logic without a real API call.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app import main
from app.services import agent

client = TestClient(main.app)


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def _tool_block(name, tool_input, block_id="tu_1"):
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def _response(content, stop_reason):
    return SimpleNamespace(content=content, stop_reason=stop_reason)


def test_no_key_falls_back_to_plain_rag(monkeypatch):
    """Without an API key the agent must degrade to the RAG pipeline."""
    monkeypatch.setattr(agent.settings, "anthropic_api_key", "")
    fake = {"answer": "fallback answer", "sources": []}
    monkeypatch.setattr(agent.retrieval, "answer", lambda q: fake)

    result = agent.run_agent("How do I drop a course?")

    assert result["answer"] == "fallback answer"
    assert result["steps"][0]["tool"] == "search_course_documents"


def test_agent_loop_dispatches_tool_and_returns_answer(monkeypatch):
    """One tool-use round followed by a final text answer."""
    monkeypatch.setattr(agent.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(
        agent, "search_course_documents", lambda q: '[{"title": "Policy"}]'
    )

    fake_client = MagicMock()
    fake_client.messages.create.side_effect = [
        _response(
            [_tool_block("search_course_documents", {"query": "withdrawal"})],
            stop_reason="tool_use",
        ),
        _response([_text_block("Final grounded answer.")], stop_reason="end_turn"),
    ]
    monkeypatch.setattr(agent, "Anthropic", lambda api_key: fake_client)

    result = agent.run_agent("How do I withdraw?")

    assert result["answer"] == "Final grounded answer."
    assert result["steps"] == [
        {"tool": "search_course_documents", "input": {"query": "withdrawal"}}
    ]
    assert fake_client.messages.create.call_count == 2


def test_agent_loop_stops_at_round_limit(monkeypatch):
    """A model that keeps requesting tools must hit the round cap, not loop."""
    monkeypatch.setattr(agent.settings, "anthropic_api_key", "test-key")
    monkeypatch.setattr(agent, "search_course_documents", lambda q: "[]")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = _response(
        [_tool_block("search_course_documents", {"query": "loop"})],
        stop_reason="tool_use",
    )
    monkeypatch.setattr(agent, "Anthropic", lambda api_key: fake_client)

    result = agent.run_agent("loop forever")

    assert fake_client.messages.create.call_count == agent.MAX_ROUNDS
    assert "steps" in result and len(result["steps"]) == agent.MAX_ROUNDS


def test_unknown_tool_returns_error_payload():
    result = agent._dispatch("nonexistent_tool", {})
    assert "unknown tool" in result


def test_agent_endpoint_returns_answer_and_steps(monkeypatch):
    monkeypatch.setattr(main.store, "count", lambda: 5)
    fake = {"answer": "ok", "steps": [{"tool": "search_course_documents", "input": {}}]}
    monkeypatch.setattr(main.agent, "run_agent", lambda q: fake)

    res = client.post("/agent", json={"question": "How do I drop a course?"})

    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "ok"
    assert body["steps"][0]["tool"] == "search_course_documents"


def test_agent_endpoint_rejects_empty_knowledge_base(monkeypatch):
    monkeypatch.setattr(main.store, "count", lambda: 0)
    res = client.post("/agent", json={"question": "anything"})
    assert res.status_code == 409
