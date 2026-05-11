from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from src.config import settings
from src.retrieval.hybrid import HybridRetriever


def make_local_search_tool(retriever: HybridRetriever):
    """Factory that creates a LangChain tool bound to a specific retriever instance."""

    @tool
    def local_search(query: str) -> list[dict[str, Any]]:
        """Search internal documents using hybrid dense+sparse retrieval."""
        return retriever.retrieve(query, top_k=settings.retrieval_top_k)

    return local_search


def make_web_search_tool():
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)

    @tool
    def web_search(query: str) -> list[dict[str, Any]]:
        """Search the web for current information using Tavily."""
        results = client.search(query, max_results=5)
        return [
            {"text": r["content"], "metadata": {"source": r["url"]}}
            for r in results.get("results", [])
        ]

    return web_search
