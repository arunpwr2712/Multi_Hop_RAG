from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from multi_hop_causal_rag.config import build_default_config
from multi_hop_causal_rag.pipeline.query_processor import generate_retrieval_queries_llm
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever
from multi_hop_causal_rag.retrieval.vector_index import RetrievalResult

from .baseline_rag import select_answer_snippet, select_hr_evidence_texts


class IterativeRAGModel:
    """V2: iterative query decomposition + multi-step retrieval without causal graph."""

    def __init__(self, retriever: CausalRetriever, top_k: int, hop_top_k: int, max_hops: int) -> None:
        self.retriever = retriever
        self.top_k = max(1, int(top_k))
        self.hop_top_k = max(1, int(hop_top_k))
        self.max_hops = max(1, int(max_hops))

        self.app_config = build_default_config()
        self.app_config.llm.query_planning_enabled = False
        self.app_config.llm.require_ollama = False

    def predict(self, query: str) -> Dict[str, Any]:
        subqueries = generate_retrieval_queries_llm(
            query=query,
            config=self.app_config,
            min_queries=self.max_hops,
            max_queries=max(self.max_hops + 1, self.max_hops),
        )

        merged: Dict[str, RetrievalResult] = {}
        for hop_query in subqueries[: self.max_hops]:
            results = self.retriever.retrieve_documents(query=hop_query, top_k=self.hop_top_k)
            for item in results:
                prev = merged.get(item.document.doc_id)
                if prev is None or item.score > prev.score:
                    merged[item.document.doc_id] = item

        ranked_results = sorted(merged.values(), key=lambda x: x.score, reverse=True)[: self.top_k]
        retrieved_texts = [item.document.text for item in ranked_results]
        answer = select_answer_snippet(query=query, retrieved_texts=retrieved_texts)
        hr_evidence_texts = select_hr_evidence_texts(query=query, retrieved_texts=retrieved_texts, keep_probability=0.64)

        return {
            "answer": answer,
            "retrieved_doc_ids": [item.document.doc_id for item in ranked_results],
            "retrieved_question_ids": [
                str(item.document.metadata.get("question_id", "")).strip()
                for item in ranked_results
                if str(item.document.metadata.get("question_id", "")).strip()
            ],
            "retrieved_texts": retrieved_texts,
            "hr_evidence_texts": hr_evidence_texts,
            "causal_chain": [],
            "pred_sources": [item.document.doc_id for item in ranked_results],
            "subqueries": subqueries,
        }
