#!/usr/bin/env python
"""Load documents from a directory or URL list into Qdrant + rebuild BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

# activate project venv before running: source .venv/bin/activate
from src.retrieval.vector_store import VectorStore

logger = structlog.get_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


def load_from_directory(path: Path) -> list[dict]:
    documents = []
    for file in path.rglob("*"):
        if file.suffix not in SUPPORTED_EXTENSIONS:
            continue
        text = file.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append({"text": text, "metadata": {"source": str(file)}})
    return documents


def load_from_jsonl(path: Path) -> list[dict]:
    documents = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            obj = json.loads(line)
            documents.append({
                "text": obj["text"],
                "metadata": obj.get("metadata", {"source": str(path)}),
            })
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--source", required=True, help="Directory or .jsonl file path")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    if source_path.is_dir():
        documents = load_from_directory(source_path)
    elif source_path.suffix == ".jsonl":
        documents = load_from_jsonl(source_path)
    else:
        raise ValueError(f"Unsupported source type: {source_path}")

    logger.info("ingest.loaded", count=len(documents))

    store = VectorStore()
    for i in range(0, len(documents), args.batch_size):
        batch = documents[i : i + args.batch_size]
        store.upsert(batch)
        logger.info("ingest.batch_done", batch=i // args.batch_size + 1)

    logger.info("ingest.complete", total=len(documents))


if __name__ == "__main__":
    main()
