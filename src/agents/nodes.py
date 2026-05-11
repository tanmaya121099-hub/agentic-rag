from __future__ import annotations

from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.config import settings
from src.evaluation.grader import RetrievalGrader
from src.retrieval.hybrid import HybridRetriever

logger = structlog.get_logger(__name__)

_REWRITE_PROMPT = """You are a query rewriting expert. The original query failed to retrieve
relevant documents. Rewrite it to be more specific and likely to find useful content.

Original query: {query}
Reason retrieval failed: {reason}

Respond with ONLY the rewritten query, nothing else."""

_GENERATE_PROMPT = """You are a helpful assistant. Answer the question using ONLY the provided context.
Always cite your sources by mentioning the document or URL they came from.
If context is insufficient, say so clearly — do not hallucinate.

Context:
{context}

Question: {question}"""

_INTENT_PROMPT = """Classify whether this query should be answered using:
- "local": internal documents / knowledge base
- "web": requires current/live information
- "both": benefits from both sources

Query: {query}
Respond with ONLY one word: local, web, or both."""


class AgentNodes:
    """All LangGraph node implementations for the RAG agent."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self._retriever = retriever
        self._grader = RetrievalGrader()
        self._tavily = TavilyClient(api_key=settings.tavily_api_key)
        self._generator = ChatAnthropic(
            model=settings.generator_model,
            api_key=settings.anthropic_api_key,
        )
        self._router = ChatOpenAI(
            model=settings.grader_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self._rewriter = ChatOpenAI(
            model=settings.grader_model,
            api_key=settings.openai_api_key,
            temperature=0.3,
        )

    def classify_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        response = self._router.invoke([HumanMessage(content=_INTENT_PROMPT.format(query=query))])
        intent = response.content.strip().lower()
        if intent not in ("local", "web", "both"):
            intent = "local"
        logger.info("intent.classified", intent=intent, query=query[:80])
        return {**state, "intent": intent}

    def retrieve(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["query"]
        chunks = self._retriever.retrieve(query)
        logger.info("retrieval.done", chunk_count=len(chunks))
        return {**state, "chunks": chunks}

    def grade(self, state: dict[str, Any]) -> dict[str, Any]:
        result = self._grader.grade(state["query"], state["chunks"])
        return {**state, "grade_relevant": result.relevant, "grade_reason": result.reason}

    def rewrite_query(self, state: dict[str, Any]) -> dict[str, Any]:
        attempts = state.get("rewrite_attempts", 0) + 1
        prompt = _REWRITE_PROMPT.format(
            query=state["query"], reason=state.get("grade_reason", "unknown")
        )
        response = self._rewriter.invoke([HumanMessage(content=prompt)])
        new_query = response.content.strip()
        logger.info("query.rewritten", attempt=attempts, new_query=new_query[:80])
        return {**state, "query": new_query, "rewrite_attempts": attempts}

    def web_search(self, state: dict[str, Any]) -> dict[str, Any]:
        results = self._tavily.search(state["query"], max_results=5)
        chunks = [
            {"text": r["content"], "metadata": {"source": r["url"]}}
            for r in results.get("results", [])
        ]
        logger.info("web_search.done", result_count=len(chunks))
        return {**state, "chunks": chunks, "used_web": True}

    def generate(self, state: dict[str, Any]) -> dict[str, Any]:
        context_parts = []
        sources = []
        for chunk in state["chunks"]:
            context_parts.append(chunk["text"])
            src = chunk.get("metadata", {}).get("source", "internal doc")
            if src not in sources:
                sources.append(src)

        context = "\n\n".join(context_parts)
        messages = [
            SystemMessage(content="You are a helpful, precise assistant."),
            HumanMessage(
                content=_GENERATE_PROMPT.format(
                    context=context, question=state["original_query"]
                )
            ),
        ]
        response = self._generator.invoke(messages)
        return {
            **state,
            "answer": response.content,
            "sources": sources,
        }

    # ---------- routing functions (return strings for LangGraph edges) ----------

    @staticmethod
    def route_by_intent(state: dict[str, Any]) -> str:
        return state.get("intent", "local")

    @staticmethod
    def route_after_grade(state: dict[str, Any]) -> str:
        if state.get("grade_relevant"):
            return "generate"
        rewrites = state.get("rewrite_attempts", 0)
        if rewrites >= settings.max_rewrite_attempts:
            return "web_search"
        return "rewrite"
