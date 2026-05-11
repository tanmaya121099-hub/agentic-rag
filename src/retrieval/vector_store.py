from __future__ import annotations

import uuid
from typing import Any

import structlog
from openai import OpenAI
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger(__name__)

_VECTOR_SIZE = settings.embedding_dimensions
_DISTANCE = models.Distance.COSINE


class VectorStore:
    def __init__(self) -> None:
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except (UnexpectedResponse, Exception):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=_VECTOR_SIZE,
                    distance=_DISTANCE,
                ),
            )
            logger.info("qdrant.collection_created", collection=self._collection)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(
            input=text,
            model=settings.embedding_model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._openai.embeddings.create(
            input=texts,
            model=settings.embedding_model,
        )
        return [item.embedding for item in response.data]

    def upsert(self, documents: list[dict[str, Any]]) -> None:
        texts = [d["text"] for d in documents]
        vectors = self.embed_batch(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": doc["text"], **doc.get("metadata", {})},
            )
            for doc, vec in zip(documents, vectors)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        logger.info("qdrant.upserted", count=len(points))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vec = self.embed(query)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            {"text": r.payload["text"], "metadata": r.payload, "score": r.score}
            for r in results
        ]

    def health(self) -> bool:
        try:
            self._client.get_collection(self._collection)
            return True
        except Exception:
            return False
