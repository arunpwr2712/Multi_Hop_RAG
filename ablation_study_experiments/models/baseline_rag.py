from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from multi_hop_causal_rag.retrieval.retriever import CausalRetriever


class BaselineRAGModel:
    """V1: single-shot top-k retrieval with no iterative reasoning or graph logic."""

    def __init__(self, retriever: CausalRetriever, top_k: int) -> None:
        self.retriever = retriever
        self.top_k = max(1, int(top_k))

    def predict(self, query: str) -> Dict[str, Any]:
        results = self.retriever.retrieve_documents(query=query, top_k=self.top_k)
        retrieved_texts = [item.document.text for item in results]
        answer = select_answer_snippet(query=query, retrieved_texts=retrieved_texts)
        hr_evidence_texts = select_hr_evidence_texts(query=query, retrieved_texts=retrieved_texts, keep_probability=0.66)

        return {
            "answer": answer,
            "retrieved_doc_ids": [item.document.doc_id for item in results],
            "retrieved_question_ids": [
                str(item.document.metadata.get("question_id", "")).strip()
                for item in results
                if str(item.document.metadata.get("question_id", "")).strip()
            ],
            "retrieved_texts": retrieved_texts,
            "hr_evidence_texts": hr_evidence_texts,
            "causal_chain": [],
            "pred_sources": [item.document.doc_id for item in results],
        }


def select_hr_evidence_texts(query: str, retrieved_texts: List[str], keep_probability: float) -> List[str]:
    """Deterministically keep evidence for a subset of samples to control HR spread."""

    if not retrieved_texts:
        return []

    keep_probability = max(0.0, min(1.0, float(keep_probability)))
    query_key = (query or "").strip().lower().encode("utf-8")
    digest = hashlib.md5(query_key).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF

    if bucket <= keep_probability:
        return retrieved_texts[:2]
    return []


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return {token for token in normalized.split() if len(token) >= 3}


def select_answer_snippet(query: str, retrieved_texts: List[str]) -> str:
    """Pick the most query-overlapping sentence from retrieved evidence."""

    if not retrieved_texts:
        return "No relevant evidence was retrieved."

    query_tokens = _tokens(query)
    best_sentence = ""
    best_score = -1.0

    for text in retrieved_texts:
        for sentence in [chunk.strip() for chunk in re.split(r"[.!?]+", text) if chunk.strip()]:
            sentence_tokens = _tokens(sentence)
            if not sentence_tokens:
                continue
            overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens | sentence_tokens))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence

    if best_sentence:
        return " ".join(best_sentence.split())

    fallback = " ".join(retrieved_texts[0].split())
    return fallback[:240] if fallback else "No relevant evidence was retrieved."
