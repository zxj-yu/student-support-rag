# Student Support RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers student questions
— course FAQs, enrollment rules, academic policies — by grounding responses in a
curated knowledge base instead of relying on a model's parametric memory. Built
end to end as a production-style project: containerized, tested, CI-gated, and
deployment-ready for Cloud Run.

**🔗 Live demo (frontend):** https://zxj-yu.github.io/student-support-rag/frontend/

> The chat **interface** is deployed and public. Full conversational answers
> require the backend (FastAPI + Qdrant) running locally via `docker-compose`
> — see Quick start below. The deployed page demonstrates the UI; the backend
> is run on demand.

## Demo

The assistant uses semantic search, not keyword matching. Asking *"How do I
drop a course after the deadline?"* correctly surfaces the **Course Withdrawal
Policy** documents (even though the question says "drop" and the policy says
"withdrawal"), ranked by relevance score.

## Architecture

```
                +-------------------+
   user query   |   HTML/JS client  |
   ───────────▶ |     (chat UI)     |
                +---------+---------+
                          │  HTTP (CORS-enabled)
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
          | (embeddings)   |  | (answer gen,   |
          |                |  |  no-key fallbk)|
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
- **Generation:** pluggable LLM client; falls back to returning retrieved
  context when no API key is configured, so the pipeline runs end to end for free
- **Frontend:** vanilla HTML/CSS/JS chat UI (no framework)
- **Infra:** Docker Compose, GitHub Actions CI, pytest, ruff

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

# or open the chat UI
open frontend/index.html
```

Interactive API docs: http://localhost:8000/docs

## Agent mode

Beyond the fixed retrieve→generate pipeline, `POST /agent` runs an agentic
loop: the model decides per question whether to search the knowledge base,
call another tool (e.g. an assignment-status lookup), chain several calls, or
answer directly. The response includes a `steps` trace of every tool call, so
the decision process is observable:

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"question": "How many assignments do I still need to submit?"}'
```

Implemented directly on the Anthropic tool-use API (no agent framework), with
a round cap and graceful degradation to plain RAG when no API key is set.

## Evaluation

The `evaluation/` harness measures quality on two layers, using test cases
grounded in the actual knowledge base (`evaluation/cases.json`):

- **Retrieval hit rate** — did the expected source document appear in the
  top-k results? Runs against any live backend, no API key needed.
- **Answer quality (LLM-as-judge)** — a judge model scores generated answers
  against reference answers on a 1-5 scale. Enabled when
  `ANTHROPIC_API_KEY` is set.

```bash
# with the stack running and ingested:
python -m evaluation.evaluate                        # local backend
python -m evaluation.evaluate --base-url https://<your-app>.run.app
```

## Running the tests

```bash
pytest tests/ -v
```

External services (Qdrant, SentenceTransformer, the LLM client) are stubbed in
`tests/conftest.py`, so the suite runs fast and in isolation — no database, no
model download, no API key. The same suite runs in CI on every push.

## Project layout

```
app/
  core/        config / settings
  services/    embeddings, vector store, retrieval, llm, agent
  main.py      FastAPI app (routes, CORS)
data/          knowledge base documents (JSON)
evaluation/    eval harness: retrieval hit rate + LLM-as-judge
frontend/      static chat UI (deployed to GitHub Pages)
tests/         pytest suite (API + unit, with mocks)
```

## Roadmap

- [x] **Core RAG loop:** FastAPI backend, SentenceTransformer embeddings,
  Qdrant vector store, document ingestion, grounded retrieval, containerized
  with docker-compose, CI on every push.
- [x] **UX & engineering:** HTML/JS chat frontend, CORS, no-key fallback mode,
  expanded test coverage (API + unit with mocks), ruff lint gate.
- [x] **Ship it:** frontend deployed to GitHub Pages.
- [x] **Agent layer:** tool-use loop (`/agent`) where the model decides which
  tools to call, built directly on the Anthropic API with an observable
  step trace.
- [x] **Evaluation:** two-layer eval harness (retrieval hit rate +
  LLM-as-judge) with test cases grounded in the knowledge base.
- [ ] **Next:** full-stack cloud deployment (Cloud Run + Qdrant Cloud),
  streaming responses (SSE).

## License

MIT — see [LICENSE](LICENSE).
