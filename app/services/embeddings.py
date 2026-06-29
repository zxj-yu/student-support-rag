"""Embedding service: turns text into vectors via SentenceTransformer."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load the model once and reuse it (cold start is slow)."""
    return SentenceTransformer(settings.embedding_model)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input."""
    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_one(text: str) -> list[float]:
    """Embed a single text."""
    return embed([text])[0]
