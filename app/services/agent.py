"""Agent layer: a tool-use loop in front of the RAG pipeline.

Instead of always running the fixed "retrieve -> generate" pipeline, the
model decides per question how to proceed: search the knowledge base, call
another tool, chain several calls, or answer directly. Falls back to plain
RAG when no API key is configured, so the endpoint works in every mode.
"""
import json
import logging
from typing import Any

from anthropic import Anthropic

from app.core.config import settings
from app.services import embeddings, retrieval
from app.services.vector_store import store

logger = logging.getLogger(__name__)

MAX_ROUNDS = 5

SYSTEM_PROMPT = (
    "You are a student support assistant. Use the available tools to ground "
    "your answers: search the knowledge base for questions about policies, "
    "deadlines, or academic rules, and check assignment status when asked "
    "about submissions. If the tools do not return the information needed, "
    "say so and suggest contacting the relevant office. Be concise."
)

TOOLS = [
    {
        "name": "search_course_documents",
        "description": (
            "Semantic search over the student support knowledge base. Use for "
            "questions about academic policies, enrollment, withdrawal, GPA, "
            "tuition, or deadlines."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or keywords to search for",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_assignment_status",
        "description": (
            "Look up a student's assignment submission status for a course. "
            "Use for questions like 'how many assignments do I still need to "
            "submit?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "student_id": {"type": "string"},
                "course_id": {"type": "string"},
            },
            "required": ["student_id", "course_id"],
        },
    },
]


def search_course_documents(query: str) -> str:
    """Tool: semantic search over the vector store (same path as /chat)."""
    query_vec = embeddings.embed_one(query)
    contexts = store.search(query_vec, top_k=settings.top_k)
    return json.dumps(contexts, ensure_ascii=False)


def check_assignment_status(student_id: str, course_id: str) -> str:
    """Tool: assignment status lookup.

    Demo implementation returning fixed data; in production this would query
    the LMS (Canvas/Moodle) API or an internal database.
    """
    return json.dumps(
        {
            "student_id": student_id,
            "course_id": course_id,
            "pending": 2,
            "submitted": 8,
            "total": 10,
            "note": "demo data",
        }
    )


def _dispatch(name: str, tool_input: dict) -> str:
    if name == "search_course_documents":
        return search_course_documents(tool_input["query"])
    if name == "check_assignment_status":
        return check_assignment_status(
            tool_input.get("student_id", "unknown"),
            tool_input.get("course_id", "unknown"),
        )
    return json.dumps({"error": f"unknown tool: {name}"})


def run_agent(question: str) -> dict:
    """Run the agent loop and return the answer plus a trace of tool calls.

    Returns {"answer": str, "steps": [{"tool": ..., "input": ...}, ...]}.
    Without an API key, degrades to the plain RAG pipeline so the endpoint
    stays usable in no-key mode.
    """
    if not settings.anthropic_api_key:
        result = retrieval.answer(question)
        return {
            "answer": result["answer"],
            "steps": [{"tool": "search_course_documents", "input": {"query": question}}],
        }

    client = Anthropic(api_key=settings.anthropic_api_key)
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[dict] = []

    for _ in range(MAX_ROUNDS):
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": answer, "steps": steps}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            steps.append({"tool": block.name, "input": dict(block.input)})
            try:
                result = _dispatch(block.name, block.input)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool %s failed: %s", block.name, exc)
                result = json.dumps({"error": str(exc)})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": (
            "I couldn't resolve this within the allowed number of steps. "
            "Please rephrase or contact the relevant office."
        ),
        "steps": steps,
    }
