"""
Run RAGAS evaluation against the eval dataset.

Usage:
    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --dataset tests/eval_dataset.json --out evaluation_results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.graph import RAGAgent
from src.evaluation.metrics import RAGASEvaluator


def load_dataset(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def run(dataset_path: str, output_dir: str) -> None:
    samples = load_dataset(dataset_path)
    agent = RAGAgent()
    evaluator = RAGASEvaluator()

    ragas_samples = []
    for item in samples:
        result = agent.run(item["question"])
        ragas_samples.append({
            "question": item["question"],
            "answer": result.get("answer", ""),
            "contexts": [c["text"] for c in result.get("chunks", [])],
            "ground_truth": item["ground_truth"],
        })

    scores = evaluator.run(ragas_samples)
    report_path = evaluator.save_report(scores, output_dir)
    print(f"Scores: {json.dumps(scores, indent=2)}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tests/eval_dataset.json")
    parser.add_argument("--out", default="evaluation_results")
    args = parser.parse_args()
    run(args.dataset, args.out)
