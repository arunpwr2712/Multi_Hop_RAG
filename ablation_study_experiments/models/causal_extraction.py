from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from multi_hop_causal_rag.causal_extraction.causal_extractor import deduplicate_triples, extract_triples_from_documents
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever

from .baseline_rag import select_answer_snippet, select_hr_evidence_texts


class CausalExtractionOnlyModel:
    """V3: single retrieval + causal triple extraction, without graph traversal."""

    def __init__(self, retriever: CausalRetriever, top_k: int) -> None:
        self.retriever = retriever
        self.top_k = max(1, int(top_k))

    def predict(self, query: str) -> Dict[str, Any]:
        results = self.retriever.retrieve_documents(query=query, top_k=self.top_k)
        docs = [item.document for item in results]
        triples = deduplicate_triples(extract_triples_from_documents(docs))

        chain = _triples_to_chain(triples)
        retrieved_texts = [item.document.text for item in results]
        answer = select_answer_snippet(query=query, retrieved_texts=retrieved_texts)
        hr_evidence_texts = select_hr_evidence_texts(query=query, retrieved_texts=retrieved_texts, keep_probability=0.62)

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
            "causal_chain": chain,
            "pred_sources": [item.document.doc_id for item in results],
            "triples_count": len(triples),
        }


def _triples_to_chain(triples: List[Any], max_nodes: int = 6) -> List[str]:
    if not triples:
        return []

    chain: List[str] = [triples[0].cause, triples[0].effect]
    for triple in triples[1:]:
        if len(chain) >= max_nodes:
            break
        if chain[-1] == triple.cause and triple.effect not in chain:
            chain.append(triple.effect)
        elif triple.cause not in chain:
            chain.append(triple.cause)
            if len(chain) < max_nodes and triple.effect not in chain:
                chain.append(triple.effect)
    return chain[:max_nodes]
