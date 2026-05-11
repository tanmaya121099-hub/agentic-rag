from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.agents.nodes import AgentNodes
from src.config import settings
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore


class RAGState(TypedDict, total=False):
    query: str                  # may be rewritten across iterations
    original_query: str         # never mutated — used for final answer generation
    intent: str                 # local | web | both
    chunks: list[dict[str, Any]]
    grade_relevant: bool
    grade_reason: str
    rewrite_attempts: int
    used_web: bool
    answer: str
    sources: list[str]


def build_graph(nodes: AgentNodes) -> Any:
    graph = StateGraph(RAGState)

    graph.add_node("classify_intent", nodes.classify_intent)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        nodes.route_by_intent,
        {
            "local": "retrieve",
            "web": "web_search",
            "both": "retrieve",
        },
    )

    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        nodes.route_after_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
            "web_search": "web_search",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


class RAGAgent:
    """Top-level agent that orchestrates the RAG pipeline."""

    def __init__(self) -> None:
        vector_store = VectorStore()
        bm25_index = BM25Index()
        retriever = HybridRetriever(vector_store, bm25_index)
        nodes = AgentNodes(retriever)
        self._graph = build_graph(nodes)
        self._bm25 = bm25_index
        self._vector = vector_store

    def load_bm25_corpus(self, documents: list[dict[str, Any]]) -> None:
        self._bm25.build(documents)

    def run(self, query: str, source: str = "auto") -> dict[str, Any]:
        initial_state: RAGState = {
            "query": query,
            "original_query": query,
            "rewrite_attempts": 0,
            "used_web": False,
            "intent": source if source in ("local", "web", "both") else "local",
        }

        # skip intent classification if caller forced a source
        if source == "auto":
            result = self._graph.invoke(initial_state)
        else:
            result = self._graph.invoke({**initial_state, "intent": source})

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "rewrite_attempts": result.get("rewrite_attempts", 0),
            "used_web_fallback": result.get("used_web", False),
        }
