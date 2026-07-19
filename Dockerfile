# ---- Build stage: install dependencies ----
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies (needed for packages that compile native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- Runtime stage: copy only what's needed, keeps the image small ----
FROM python:3.11-slim

WORKDIR /app

# Run as non-root user - a security best practice
RUN useradd -m appuser
COPY --from=builder /root/.local /home/appuser/.local
COPY app ./app
COPY data ./data

ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT (defaults to 8080); local docker-compose sets PORT=8000
ENV PORT=8080
EXPOSE 8080

USER appuser

# Shell form so $PORT gets expanded. NOTE: the app module is app.main, not main.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
