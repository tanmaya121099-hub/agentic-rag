from __future__ import annotations

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import settings

logger = structlog.get_logger(__name__)

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a retrieval quality grader. Given a user question and retrieved document chunks, "
        "decide if the chunks are RELEVANT to answering the question.\n"
        "Respond with a JSON object with key 'relevant' (true/false) and 'reason' (one sentence).\n"
        "Be strict: if the chunks don't contain facts needed to answer the question, mark as irrelevant.",
    ),
    (
        "human",
        "Question: {question}\n\nRetrieved chunks:\n{context}\n\nAre these relevant?",
    ),
])


class GradeResult(BaseModel):
    relevant: bool = Field(description="Whether the chunks are relevant to the question")
    reason: str = Field(description="Brief explanation")


class RetrievalGrader:
    """LLM-as-judge that scores whether retrieved chunks actually answer a query."""

    def __init__(self) -> None:
        llm = ChatOpenAI(
            model=settings.grader_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        self._chain = _GRADE_PROMPT | llm.with_structured_output(GradeResult)

    def grade(self, question: str, chunks: list[dict]) -> GradeResult:
        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        result: GradeResult = self._chain.invoke(
            {"question": question, "context": context}
        )
        logger.info(
            "grader.result",
            relevant=result.relevant,
            reason=result.reason,
            question=question[:80],
        )
        return result
