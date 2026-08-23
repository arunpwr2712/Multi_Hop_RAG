"""Utilities for loading the WorldTree V2 science explanation corpus."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .preprocessing import Document, chunk_text, normalize_text


def load_worldtree(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load WorldTree data from JSON, JSONL, CSV, or TSV into raw records."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"WorldTree file not found: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    records: List[Dict[str, Any]]

    if suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
    elif suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with dataset_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            records = list(reader)
    else:
        raise ValueError("WorldTree loader supports .json, .jsonl, .csv, and .tsv files")

    if max_samples is not None:
        records = records[:max_samples]
    return records


def worldtree_to_documents(path: str, max_samples: Optional[int] = None) -> List[Document]:
    """Convert WorldTree explanations and facts into retrieval documents."""

    records = load_worldtree(path=path, max_samples=max_samples)
    documents: List[Document] = []

    for index, record in enumerate(records):
        explanation = normalize_text(
            str(
                record.get("explanation")
                or record.get("explanation_text")
                or record.get("fact")
                or record.get("sentence")
                or ""
            )
        )
        if not explanation:
            continue

        question = normalize_text(str(record.get("question", record.get("Question", ""))))
        answer = normalize_text(str(record.get("answer", record.get("Answer", ""))))
        source_id = normalize_text(str(record.get("uid", record.get("id", f"worldtree-{index}"))))

        for chunk_index, chunk in enumerate(chunk_text(explanation)):
            documents.append(
                Document(
                    doc_id=f"worldtree-{index}-{chunk_index}",
                    text=chunk,
                    source="WorldTreeV2",
                    metadata={
                        "record_id": source_id,
                        "question": question,
                        "answer": answer,
                    },
                )
            )

    return documents
