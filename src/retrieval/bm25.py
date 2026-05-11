from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index over a corpus of documents."""

    def __init__(self) -> None:
        self._corpus: list[dict[str, Any]] = []
        self._index: BM25Okapi | None = None

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._corpus = documents
        tokenized = [_tokenize(d["text"]) for d in documents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._index is None or not self._corpus:
            return []

        tokens = _tokenize(query)
        scores: list[float] = self._index.get_scores(tokens).tolist()

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            {
                "text": self._corpus[i]["text"],
                "metadata": self._corpus[i].get("metadata", {}),
                "score": score,
            }
            for i, score in ranked
            if score > 0.0
        ]

    @property
    def size(self) -> int:
        return len(self._corpus)
