"""FastAPI application entrypoint."""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services import retrieval
from app.services.vector_store import store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Student Support RAG Assistant", version="0.2.0")

# Allow the local HTML frontend to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    # Enforce a sane length at the schema level: FastAPI returns a clear 422
    # instead of us wasting an embedding call on a pasted novel.
    question: str = Field(..., min_length=1, max_length=settings.max_question_chars)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    """Real health check: reports whether the vector store is actually reachable.

    A health endpoint that always returns "ok" is worse than none at all - it
    tells you nothing. This one reports degraded status when Qdrant is down.
    """
    try:
        count = store.count()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check: vector store unreachable: %s", exc)
        return {"status": "degraded", "vector_store": "unreachable", "documents": 0}
    return {
        "status": "ok",
        "vector_store": "reachable",
        "documents": count,
        "knowledge_base": "empty" if count == 0 else "loaded",
    }


@app.post("/ingest")
def ingest() -> dict:
    """(Re)build the vector index from the knowledge base file."""
    try:
        count = retrieval.ingest_knowledge_base()
    except FileNotFoundError:
        raise HTTPException(404, "Knowledge base file not found.") from None
    except ValueError as exc:
        raise HTTPException(422, f"Knowledge base is invalid: {exc}") from None
    except ConnectionError:
        raise HTTPException(
            503, "Vector store is unavailable. Is Qdrant running?"
        ) from None

    logger.info("Ingested %d chunks", count)
    return {"ingested_chunks": count, "total_in_store": store.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Question must not be empty.")

    # Fail clearly if nobody has ingested the knowledge base yet, rather than
    # silently returning an answer grounded in nothing.
    try:
        if store.count() == 0:
            raise HTTPException(
                409, "The knowledge base is empty. Run POST /ingest first."
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Vector store unreachable during chat: %s", exc)
        raise HTTPException(
            503, "Vector store is unavailable. Is Qdrant running?"
        ) from None

    try:
        result = retrieval.answer(question)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure answering question")
        raise HTTPException(500, f"Failed to answer question: {exc}") from None

    return ChatResponse(**result)
