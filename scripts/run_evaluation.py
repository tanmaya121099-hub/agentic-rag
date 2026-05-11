#!/usr/bin/env python
"""Run RAGAS evaluation against a ground-truth dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from src.agents.graph import RAGAgent
from src.evaluation.metrics import RAGASEvaluator

logger = structlog.get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--output-dir", default="evaluation_results")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = json.loads(dataset_path.read_text())
    logger.info("eval.loaded", samples=len(samples))

    agent = RAGAgent()
    enriched = []

    for sample in samples:
        query = sample["question"]
        result = agent.run(query)
        enriched.append({
            "question": query,
            "answer": result["answer"],
            "contexts": [s for s in result["sources"]],
            "ground_truth": sample["ground_truth"],
        })

    evaluator = RAGASEvaluator()
    scores = evaluator.run(enriched)
    report_path = evaluator.save_report(scores, args.output_dir)

    print("\n=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        print(f"  {metric:30s}: {score:.4f}")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
