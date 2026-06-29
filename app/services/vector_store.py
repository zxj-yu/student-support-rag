"""Vector store service: thin wrapper around the Qdrant client."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection = settings.collection_name

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dim, distance=Distance.COSINE
                ),
            )

    def upsert(self, vectors: list[list[float]], payloads: list[dict]) -> int:
        """Insert vectors with their associated payloads. Returns count."""
        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
            for vec, payload in zip(vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        """Return the top_k most similar payloads with scores."""
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {"score": hit.score, **(hit.payload or {})} for hit in hits
        ]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count


# module-level singleton
store = VectorStore()
