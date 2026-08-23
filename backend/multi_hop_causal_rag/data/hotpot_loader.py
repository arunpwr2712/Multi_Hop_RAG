"""Utilities for loading the HotpotQA corpus into retrieval documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .preprocessing import Document, chunk_text, normalize_text


def load_hotpotqa(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load a HotpotQA JSON file into raw record dictionaries."""

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"HotpotQA file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict):
        records = data.get("data", [])
    else:
        records = data

    if max_samples is not None:
        records = records[:max_samples]
    return records


def hotpotqa_to_documents(path: str, max_samples: Optional[int] = None) -> List[Document]:
    """Convert HotpotQA context paragraphs and supporting facts into dense retrieval documents."""

    records = load_hotpotqa(path=path, max_samples=max_samples)
    documents: List[Document] = []

    for example_index, record in enumerate(records):
        question = normalize_text(str(record.get("question", "")))
        answer = normalize_text(str(record.get("answer", "")))
        context = record.get("context", [])

        for paragraph_index, context_item in enumerate(context):
            if not isinstance(context_item, list) or len(context_item) != 2:
                continue

            title, sentences = context_item
            paragraph_text = normalize_text(" ".join(sentences))
            if not paragraph_text:
                continue

            for chunk_index, chunk in enumerate(chunk_text(paragraph_text)):
                doc_id = f"hotpot-{example_index}-{paragraph_index}-{chunk_index}"
                documents.append(
                    Document(
                        doc_id=doc_id,
                        text=chunk,
                        source="HotpotQA",
                        metadata={
                            "question": question,
                            "answer": answer,
                            "title": normalize_text(str(title)),
                            "example_index": example_index,
                            "paragraph_index": paragraph_index,
                        },
                    )
                )

    return documents
