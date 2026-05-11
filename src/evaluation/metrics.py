from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = structlog.get_logger(__name__)


class RAGASEvaluator:
    """Runs RAGAS evaluation on a dataset of (question, answer, contexts, ground_truth) tuples."""

    METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

    def run(self, samples: list[dict[str, Any]]) -> dict[str, float]:
        """
        Args:
            samples: list of dicts with keys:
                - question (str)
                - answer (str)        — the generated answer
                - contexts (list[str]) — the retrieved chunks used
                - ground_truth (str)  — reference answer
        Returns:
            dict mapping metric name -> mean score
        """
        dataset = Dataset.from_list(samples)
        result = evaluate(dataset, metrics=self.METRICS)
        scores = {str(k): float(v) for k, v in result.items()}
        logger.info("ragas.scores", **scores)
        return scores

    def save_report(self, scores: dict[str, float], output_dir: str = "evaluation_results") -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "ragas_report.json"
        report_path.write_text(json.dumps(scores, indent=2))
        logger.info("ragas.report_saved", path=str(report_path))
        return report_path
