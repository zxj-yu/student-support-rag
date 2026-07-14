"""
agent_layer.py

Adds a lightweight agentic decision layer in front of the original
"retrieve -> generate" RAG pipeline: the model first decides how to
handle a given question — look it up in the docs directly? call a
specific tool? or does it need multiple steps?

This is a minimal but legitimate way to upgrade a "RAG system" into
an "agent system." You don't need a heavy framework like LangChain/
LangGraph to demonstrate the core idea of agentic behavior: letting
the model decide what to do next, instead of following a hardcoded
pipeline.
"""

import json
from anthropic import Anthropic
from typing import Any

client = Anthropic()

# ---- Step 1: define the tools ----
# Two example tools here: one checks assignment status (mocked),
# the other runs your existing RAG retrieval.
# In a real project, swap these for actual data sources
# (a database query, the Canvas/Moodle API, etc.)

TOOLS = [
    {
        "name": "search_course_documents",
        "description": "Semantic search over the course document store. Good for questions about course content, syllabus, or policies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question or keywords to search for"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "check_assignment_status",
        "description": "Checks a student's assignment submission status for a given course. Good for questions like 'how many assignments do I still have left to submit?'",
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


def search_course_documents(query: str, retriever) -> str:
    """Calls your existing Qdrant + SentenceTransformer retrieval logic"""
    results = retriever.search(query, top_k=3)
    return json.dumps([r.payload for r in results], ensure_ascii=False)


def check_assignment_status(student_id: str, course_id: str) -> str:
    """Example only — replace with a real query against your assignment system"""
    # TODO: replace with a real lookup against your assignment tracking system
    mock_data = {"pending": 2, "submitted": 8, "total": 10}
    return json.dumps(mock_data, ensure_ascii=False)


def run_agent(user_question: str, retriever, student_id: str = "demo", course_id: str = "demo") -> str:
    """
    Core agent loop:
    1. Pass the question along with the tool list to the model
    2. The model decides: answer directly, or call one of the tools
    3. If a tool is called, feed the result back so the model can keep reasoning
       (this may take several rounds)
    4. Repeat until the model produces a final answer
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_question}]

    for _ in range(5):  # cap at 5 rounds of tool calls to avoid infinite loops
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )

        # Model returned a plain text answer without calling a tool -> done
        if response.stop_reason != "tool_use":
            return "".join(block.text for block in response.content if block.type == "text")

        # Model requested a tool call
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "search_course_documents":
                result = search_course_documents(block.input["query"], retriever)
            elif block.name == "check_assignment_status":
                result = check_assignment_status(
                    block.input.get("student_id", student_id),
                    block.input.get("course_id", course_id),
                )
            else:
                result = json.dumps({"error": "unknown tool"})

            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )

        messages.append({"role": "user", "content": tool_results})

    return "Sorry, this question required more steps than I'm allowed to take — I couldn't produce an answer within the round limit."
