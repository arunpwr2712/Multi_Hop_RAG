"""Preprocessing helpers for converting benchmark corpora into retrieval-ready documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class Document:
    """A retrieval-ready text document with lightweight metadata."""

    doc_id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip noisy formatting from raw text."""

    collapsed = re.sub(r"\s+", " ", text or "")
    return collapsed.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into simple sentence-like units for chunking and extraction."""

    normalized = normalize_text(text)
    if not normalized:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", normalized) if segment.strip()]


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 40) -> List[str]:
    """Chunk text into overlapping sentence windows measured in approximate words."""

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > chunk_size:
            chunks.append(" ".join(current))

            # Preserve a small overlap to retain local causal context across chunks.
            retained: List[str] = []
            retained_words = 0
            for retained_sentence in reversed(current):
                retained_words += len(retained_sentence.split())
                retained.insert(0, retained_sentence)
                if retained_words >= overlap:
                    break
            current = retained
            current_words = retained_words

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current))

    return chunks


def prepare_document_records(
    records: Sequence[Dict[str, Any]],
    text_key: str = "text",
    source_key: str = "source",
) -> List[Document]:
    """Convert generic dictionaries into normalized retrieval documents."""

    documents: List[Document] = []
    for index, record in enumerate(records):
        text = normalize_text(str(record.get(text_key, "")))
        if not text:
            continue

        source = str(record.get(source_key, "unknown"))
        doc_id = str(record.get("doc_id", f"doc-{index}"))
        metadata = {key: value for key, value in record.items() if key not in {text_key, source_key, "doc_id"}}
        documents.append(Document(doc_id=doc_id, text=text, source=source, metadata=metadata))
    return documents


def merge_datasets(datasets: Iterable[Sequence[Document]]) -> List[Document]:
    """Merge multiple document collections while preserving order."""

    merged: List[Document] = []
    for dataset in datasets:
        merged.extend(list(dataset))
    return merged
