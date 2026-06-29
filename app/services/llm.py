"""LLM service: generates a grounded answer from retrieved context.

Kept deliberately thin and pluggable so the provider can be swapped without
touching the retrieval code.
"""
from anthropic import Anthropic

from app.core.config import settings

SYSTEM_PROMPT = (
    "You are a helpful student support assistant. Answer using ONLY the "
    "provided context. If the context does not contain the answer, say you "
    "don't have that information and suggest contacting the relevant office. "
    "Be concise and cite which source you used."
)


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title", f"source {i}")
        text = ctx.get("text", "")
        blocks.append(f"[Source {i}: {title}]\n{text}")
    joined = "\n\n".join(blocks)
    return f"Context:\n{joined}\n\nQuestion: {question}"


def _fallback_answer(question: str, contexts: list[dict]) -> str:
    """Used when no API key is set: return retrieved context directly."""
    if not contexts:
        return "(No relevant documents found in the knowledge base.)"
    lines = [
        "[No LLM key set - showing retrieved context instead of a generated "
        "answer.]\n",
        f"Most relevant sources for: {question!r}\n",
    ]
    for i, ctx in enumerate(contexts, 1):
        title = ctx.get("title", f"source {i}")
        score = ctx.get("score", 0)
        text = ctx.get("text", "")
        lines.append(f"--- Source {i}: {title} (score {score:.3f}) ---\n{text}\n")
    return "\n".join(lines)


def generate(question: str, contexts: list[dict]) -> str:
    """Non-streaming generation. Falls back to raw context if no API key."""
    if not settings.anthropic_api_key:
        return _fallback_answer(question, contexts)

    client = Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    )
