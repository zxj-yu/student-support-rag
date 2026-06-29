# Student Support RAG Assistant

A retrieval-augmented (RAG) chatbot that answers student questions (course FAQs,
enrollment rules, academic policies) by grounding LLM responses in a curated
knowledge base. Built as a production-style project: containerized, tested, and
deployable.

## Architecture

```
                +-------------------+
   user query   |   HTML/JS client  |
   ───────────▶ |   (chat UI, SSE)  |
                +---------+---------+
                          │  HTTP
                          ▼
                +-------------------+
                |  FastAPI backend  |
                |  /chat  /ingest   |
                +----+---------+----+
                     │         │
        embed query  │         │  retrieved context + prompt
                     ▼         ▼
          +----------------+  +----------------+
          | SentenceTransf |  |      LLM       |
          | (embeddings)   |  | (answer gen)   |
          +-------+--------+  +----------------+
                  │  vectors
                  ▼
          +----------------+
          |    Qdrant      |
          | (vector store) |
          +----------------+
```

## Tech stack

- **Backend:** Python 3.11 + FastAPI (auto OpenAPI docs at `/docs`)
- **Embeddings:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Vector store:** Qdrant (runs as a service via docker-compose)
- **Generation:** pluggable LLM client (Anthropic API by default)
- **Infra:** Docker Compose, GitHub Actions CI, pytest

## Quick start

```bash
# 1. start everything (backend + Qdrant)
docker-compose up --build

# 2. ingest the knowledge base (one time)
curl -X POST http://localhost:8000/ingest

# 3. ask a question
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I drop a course after the deadline?"}'
```

Interactive API docs: http://localhost:8000/docs

## Project layout

```
app/
  api/        route handlers (chat, ingest, health)
  core/       config, settings
  services/   embedding, vector store, retrieval, llm
data/         knowledge base documents (JSON)
tests/        pytest suite
```

## Roadmap

- [x] **Week 1 — Core RAG loop:** FastAPI backend, SentenceTransformer
  embeddings, Qdrant vector store, document ingestion, grounded answer
  generation, containerized with docker-compose, CI on every push.
- [ ] **Week 2 — UX & engineering:** HTML/JS chat frontend, streaming
  responses (SSE), expanded test coverage (retrieval + API), ruff/lint gates.
- [ ] **Week 3 — Ship it:** cloud deployment with a public demo link, error
  handling and edge cases, logging, polished documentation.

## License

MIT — see [LICENSE](LICENSE).
