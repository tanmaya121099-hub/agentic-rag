from __future__ import annotations

from typing import Any

import structlog

from src.config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger(__name__)

_RRF_K = 60  # Reciprocal Rank Fusion constant (standard value)


def _reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    docs: dict[str, dict[str, Any]] = {}

    for rank, doc in enumerate(dense_results, start=1):
        key = doc["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        docs[key] = doc

    for rank, doc in enumerate(sparse_results, start=1):
        key = doc["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
        docs[key] = doc

    ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [
        {**docs[k], "rrf_score": scores[k]}
        for k in ranked_keys
    ]


class HybridRetriever:
    """Combines dense (vector) and sparse (BM25) search with RRF fusion."""

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index) -> None:
        self._vector = vector_store
        self._bm25 = bm25_index

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or settings.retrieval_top_k
        fetch_k = k * 2  # over-fetch before fusion

        dense = self._vector.search(query, top_k=fetch_k)
        sparse = self._bm25.search(query, top_k=fetch_k)

        logger.debug(
            "hybrid.retrieved",
            dense_count=len(dense),
            sparse_count=len(sparse),
            query=query[:80],
        )

        fused = _reciprocal_rank_fusion(dense, sparse)
        return fused[:k]
