"""FastAPI application entrypoint."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.services import retrieval
from app.services.vector_store import store

app = FastAPI(title="Student Support RAG Assistant", version="0.1.0")
# Allow the local HTML frontend to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; restrict to specific domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict:
    """(Re)build the vector index from the knowledge base file."""
    try:
        count = retrieval.ingest_knowledge_base()
    except FileNotFoundError:
        raise HTTPException(404, "Knowledge base file not found.")
    return {"ingested_chunks": count, "total_in_store": store.count()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(400, "Question must not be empty.")
    result = retrieval.answer(req.question)
    return ChatResponse(**result)
