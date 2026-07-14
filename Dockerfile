# ---- Build stage: install dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies (needed if you're using sentence-transformers, which requires compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Runtime stage: copy only what's needed, keeps the image small ----
FROM python:3.11-slim

WORKDIR /app

# Run as non-root user — a security best practice worth mentioning in interviews
RUN useradd -m appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY . .

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Cloud Run requires listening on the $PORT env var, defaults to 8080
ENV PORT=8080
EXPOSE 8080

USER appuser

# Use shell form so $PORT gets expanded
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
