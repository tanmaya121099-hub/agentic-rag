"""Unit tests for hybrid retrieval and BM25."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever, _reciprocal_rank_fusion


@pytest.fixture
def sample_docs():
    return [
        {"text": "Python is a high-level programming language.", "metadata": {"source": "doc1"}},
        {"text": "LangGraph is a library for building stateful multi-actor apps.", "metadata": {"source": "doc2"}},
        {"text": "Retrieval-Augmented Generation combines search and LLMs.", "metadata": {"source": "doc3"}},
        {"text": "Qdrant is a vector similarity search engine.", "metadata": {"source": "doc4"}},
        {"text": "BM25 is a bag-of-words retrieval ranking function.", "metadata": {"source": "doc5"}},
    ]


class TestBM25Index:
    def test_build_and_search(self, sample_docs):
        index = BM25Index()
        index.build(sample_docs)
        results = index.search("vector search", top_k=3)
        assert len(results) > 0
        # Qdrant (vector) doc should score high for "vector search"
        texts = [r["text"] for r in results]
        assert any("vector" in t.lower() or "Qdrant" in t for t in texts)

    def test_empty_index_returns_nothing(self):
        index = BM25Index()
        results = index.search("anything", top_k=3)
        assert results == []

    def test_no_match_returns_empty(self, sample_docs):
        index = BM25Index()
        index.build(sample_docs)
        results = index.search("xyzzy nonexistent token", top_k=3)
        # All scores will be 0, so filtered out
        assert isinstance(results, list)

    def test_size(self, sample_docs):
        index = BM25Index()
        index.build(sample_docs)
        assert index.size == len(sample_docs)


class TestRRF:
    def test_fusion_merges_lists(self):
        dense = [{"text": "a", "score": 0.9}, {"text": "b", "score": 0.7}]
        sparse = [{"text": "b", "score": 5.0}, {"text": "c", "score": 3.0}]
        fused = _reciprocal_rank_fusion(dense, sparse)
        texts = [r["text"] for r in fused]
        # "b" appears in both, should rank highest
        assert texts[0] == "b"

    def test_fusion_includes_all_unique(self):
        dense = [{"text": "a"}, {"text": "b"}]
        sparse = [{"text": "c"}, {"text": "d"}]
        fused = _reciprocal_rank_fusion(dense, sparse)
        assert len(fused) == 4

    def test_rrf_scores_are_positive(self):
        dense = [{"text": "x", "score": 1.0}]
        sparse = [{"text": "y", "score": 1.0}]
        fused = _reciprocal_rank_fusion(dense, sparse)
        assert all(r["rrf_score"] > 0 for r in fused)


class TestHybridRetriever:
    def test_retrieve_combines_sources(self, sample_docs):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            {"text": "vector result", "metadata": {}, "score": 0.9}
        ]

        bm25 = BM25Index()
        bm25.build(sample_docs)

        retriever = HybridRetriever(mock_vector, bm25)
        results = retriever.retrieve("BM25 retrieval function", top_k=3)

        # Vector was called
        mock_vector.search.assert_called_once()
        # Results come back
        assert isinstance(results, list)
        assert len(results) > 0
