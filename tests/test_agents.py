"""Integration tests for the LangGraph agent pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.nodes import AgentNodes


@pytest.fixture
def mock_nodes():
    """AgentNodes with all external calls mocked."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"text": "RAG stands for Retrieval-Augmented Generation.", "metadata": {"source": "doc1"}}
    ]
    with (
        patch("src.agents.nodes.ChatAnthropic"),
        patch("src.agents.nodes.ChatOpenAI"),
        patch("src.agents.nodes.TavilyClient"),
        patch("src.agents.nodes.RetrievalGrader"),
    ):
        nodes = AgentNodes(mock_retriever)
        yield nodes, mock_retriever


class TestRoutingFunctions:
    def test_route_intent_local(self):
        state = {"intent": "local"}
        assert AgentNodes.route_by_intent(state) == "local"

    def test_route_intent_web(self):
        assert AgentNodes.route_by_intent({"intent": "web"}) == "web"

    def test_route_intent_default(self):
        assert AgentNodes.route_by_intent({}) == "local"

    def test_route_after_grade_relevant(self):
        state = {"grade_relevant": True, "rewrite_attempts": 0}
        assert AgentNodes.route_after_grade(state) == "generate"

    def test_route_after_grade_rewrite(self):
        state = {"grade_relevant": False, "rewrite_attempts": 0}
        assert AgentNodes.route_after_grade(state) == "rewrite"

    def test_route_after_grade_web_fallback(self):
        from src.config import settings
        state = {
            "grade_relevant": False,
            "rewrite_attempts": settings.max_rewrite_attempts,
        }
        assert AgentNodes.route_after_grade(state) == "web_search"


class TestRetrieveNode:
    def test_retrieve_adds_chunks_to_state(self, mock_nodes):
        nodes, mock_retriever = mock_nodes
        result = nodes.retrieve({"query": "What is RAG?", "original_query": "What is RAG?"})
        assert "chunks" in result
        assert len(result["chunks"]) > 0
        mock_retriever.retrieve.assert_called_once_with("What is RAG?")
