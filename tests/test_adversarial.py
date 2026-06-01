"""Tests for adversarial evaluation logic."""

from __future__ import annotations

import pytest

from src.evaluation.adversarial import AdversarialQuery


class TestEvaluateResult:
    def _make_query(self, expected: str) -> AdversarialQuery:
        return AdversarialQuery(
            query="test query",
            variant_type="negation",
            expected_behavior=expected,
            base_query="base query",
        )

    def test_passes_when_insufficient_expected_and_returned(self):
        from src.evaluation.adversarial import AdversarialQueryGenerator
        gen = AdversarialQueryGenerator.__new__(AdversarialQueryGenerator)

        q = self._make_query("insufficient_evidence")
        result = gen.evaluate_result(q, {"answer_is_sufficient": False, "aggregate_confidence": 0.1})
        assert result["passed"] is True
        assert result["actual"] == "insufficient_evidence"

    def test_fails_when_answer_returned_for_unanswerable(self):
        from src.evaluation.adversarial import AdversarialQueryGenerator
        gen = AdversarialQueryGenerator.__new__(AdversarialQueryGenerator)

        q = self._make_query("insufficient_evidence")
        result = gen.evaluate_result(q, {"answer_is_sufficient": True, "aggregate_confidence": 0.85})
        assert result["passed"] is False

    def test_passes_when_answer_expected_and_returned(self):
        from src.evaluation.adversarial import AdversarialQueryGenerator
        gen = AdversarialQueryGenerator.__new__(AdversarialQueryGenerator)

        q = self._make_query("answer_with_citation")
        result = gen.evaluate_result(q, {"answer_is_sufficient": True, "aggregate_confidence": 0.9})
        assert result["passed"] is True
        assert result["actual"] == "answer_with_citation"

    def test_result_includes_snippet(self):
        from src.evaluation.adversarial import AdversarialQueryGenerator
        gen = AdversarialQueryGenerator.__new__(AdversarialQueryGenerator)

        q = self._make_query("answer_with_citation")
        agent_result = {
            "answer_is_sufficient": True,
            "aggregate_confidence": 0.8,
            "answer": "The answer is X because of Y and Z.",
        }
        result = gen.evaluate_result(q, agent_result)
        assert len(result["answer_snippet"]) > 0

    def test_result_includes_confidence(self):
        from src.evaluation.adversarial import AdversarialQueryGenerator
        gen = AdversarialQueryGenerator.__new__(AdversarialQueryGenerator)

        q = self._make_query("answer_with_citation")
        result = gen.evaluate_result(q, {"answer_is_sufficient": True, "aggregate_confidence": 0.77})
        assert result["aggregate_confidence"] == pytest.approx(0.77)
