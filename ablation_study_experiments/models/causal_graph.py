from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from multi_hop_causal_rag.causal_extraction.causal_extractor import deduplicate_triples, extract_triples_from_documents
from multi_hop_causal_rag.config import build_default_config
from multi_hop_causal_rag.graph.causal_graph import CausalKnowledgeGraph
from multi_hop_causal_rag.graph.graph_traversal import bfs_candidate_paths
from multi_hop_causal_rag.pipeline.query_processor import decompose_query_llm
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever

from .baseline_rag import select_answer_snippet, select_hr_evidence_texts
from .causal_extraction import _triples_to_chain


class CausalGraphModel:
    """V4: single retrieval pass, build graph, and perform one-pass graph reasoning."""

    def __init__(self, retriever: CausalRetriever, top_k: int, max_path_depth: int, max_candidate_paths: int) -> None:
        self.retriever = retriever
        self.top_k = max(1, int(top_k))
        self.max_path_depth = max(1, int(max_path_depth))
        self.max_candidate_paths = max(1, int(max_candidate_paths))

        self.app_config = build_default_config()
        self.app_config.llm.query_planning_enabled = False
        self.app_config.llm.require_ollama = False

    def predict(self, query: str) -> Dict[str, Any]:
        results = self.retriever.retrieve_documents(query=query, top_k=self.top_k)
        docs = [item.document for item in results]
        triples = deduplicate_triples(extract_triples_from_documents(docs))

        graph = CausalKnowledgeGraph()
        graph.update_graph(triples)

        decomposition = decompose_query_llm(query=query, config=self.app_config)
        source_node = decomposition.get("A")
        target_node = decomposition.get("D")

        candidate_paths: List[List[str]] = []
        if source_node and target_node:
            candidate_paths = bfs_candidate_paths(
                knowledge_graph=graph,
                start=source_node,
                target=target_node,
                max_depth=self.max_path_depth,
                max_paths=self.max_candidate_paths,
            )

        causal_chain = candidate_paths[0] if candidate_paths else _triples_to_chain(triples)
        retrieved_texts = [item.document.text for item in results]
        answer = select_answer_snippet(query=query, retrieved_texts=retrieved_texts)
        hr_evidence_texts = select_hr_evidence_texts(query=query, retrieved_texts=retrieved_texts, keep_probability=0.60)

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
            "causal_chain": causal_chain,
            "pred_sources": [item.document.doc_id for item in results],
            "triples_count": len(triples),
            "graph_nodes": graph.graph.number_of_nodes(),
            "graph_edges": graph.graph.number_of_edges(),
        }
