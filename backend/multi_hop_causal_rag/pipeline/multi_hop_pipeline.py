"""Iterative multi-hop causal evidence retrieval pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import networkx as nx

from ..causal_extraction.causal_extractor import deduplicate_triples, extract_triples_from_documents
from ..causal_extraction.triple_builder import canonicalize_event_text
from ..config import AppConfig, build_default_config
from ..generator.provenance_builder import build_provenance_payload
from ..generator.rag_generator import generate_answer
from ..graph.causal_graph import CausalKnowledgeGraph
from ..graph.graph_traversal import bfs_candidate_paths
from ..multi_hop.candidate_path_selector import choose_best_path
from ..multi_hop.chain_builder import EvidenceChain, add_evidence_to_chain, check_chain_complete, derive_best_chain_path, initialize_chain
from ..multi_hop.hop_retriever import retrieve_evidence_for_next_hop
from ..multi_hop.missing_link_detector import MissingLink, detect_missing_link
from ..reasoning.consistency_checker import check_chain_consistency
from ..reasoning.counterfactual_validator import validate_counterfactual_chain
from ..retrieval.retriever import CausalRetriever
from ..retrieval.vector_index import RetrievalResult
from .query_processor import decompose_query_llm, generate_retrieval_queries_llm


@dataclass
class TimingBreakdown:
    """Timing measurements for pipeline steps."""
    total_ms: float = 0.0
    decomposition_ms: float = 0.0
    query_generation_ms: float = 0.0
    initial_retrieval_ms: float = 0.0
    causal_extraction_ms: float = 0.0
    graph_construction_ms: float = 0.0
    candidate_path_ms: float = 0.0
    iterative_hops_ms: float = 0.0
    answer_generation_ms: float = 0.0


@dataclass
class MultiHopResult:
    """Structured result returned by the multi-hop causal pipeline."""

    query: str
    decomposition: Dict[str, Optional[str]]
    retrieval_queries: List[str]
    candidate_paths: List[List[str]]
    causal_chain: EvidenceChain
    answer: str
    provenance: Dict[str, object]
    trace_steps: List[Dict[str, object]]
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)


class MultiHopCausalPipeline:
    """Iterative retrieval pipeline for multi-hop causal evidence construction."""

    def __init__(self, retriever: CausalRetriever, config: Optional[AppConfig] = None) -> None:
        """Initialize the pipeline with a retriever and runtime configuration."""

        self.retriever = retriever
        self.config = config or build_default_config()
        self.knowledge_graph = CausalKnowledgeGraph()

    def run(self, query: str, chat_history: Optional[Sequence[Mapping[str, str]]] = None) -> MultiHopResult:
        """Execute the strict multi-hop causal retrieval workflow for a query."""

        timing_start = time.perf_counter()
        timing = TimingBreakdown()
        source_results: List[RetrievalResult] = []
        trace_steps: List[Dict[str, object]] = []

        trace_steps.append({"step": 1, "name": "Receive Query", "details": {"query": query}})
        
        # Decomposition step
        step_start = time.perf_counter()
        decomposition = decompose_query_llm(query=query, config=self.config)
        timing.decomposition_ms = (time.perf_counter() - step_start) * 1000
        
        # Query generation step
        step_start = time.perf_counter()
        retrieval_queries = generate_retrieval_queries_llm(query=query, config=self.config)
        timing.query_generation_ms = (time.perf_counter() - step_start) * 1000
        
        source_node = decomposition.get("A")
        target_node = decomposition.get("D")
        trace_steps.append({"step": 2, "name": "Decompose Query", "details": decomposition})
        trace_steps.append(
            {
                "step": 2.1,
                "name": "Generate Retrieval Queries",
                "details": {"retrieval_queries": retrieval_queries},
            }
        )

        # Step 3: initial retrieval over the vector database.
        step_start = time.perf_counter()
        initial_results = self._retrieve_documents_for_queries(retrieval_queries)
        timing.initial_retrieval_ms = (time.perf_counter() - step_start) * 1000
        
        source_results.extend(initial_results)
        trace_steps.append(
            {
                "step": 3,
                "name": "Initial Retrieval",
                "details": {
                    "retrieval_queries": retrieval_queries,
                    "top_k_per_query": self.config.retrieval.top_k,
                    "hits": [
                        {
                            "doc_id": result.document.doc_id,
                            "score": result.score,
                            "source": result.document.source,
                        }
                        for result in initial_results
                    ],
                },
            }
        )

        # Step 4: local causal triple extraction from the retrieved evidence.
        step_start = time.perf_counter()
        initial_documents = [result.document for result in initial_results]
        triples = deduplicate_triples(extract_triples_from_documents(initial_documents))
        timing.causal_extraction_ms = (time.perf_counter() - step_start) * 1000
        
        trace_steps.append(
            {
                "step": 4,
                "name": "Extract Local Causal Triples",
                "details": {
                    "triple_count": len(triples),
                    "triples": [
                        {
                            "cause": triple.cause,
                            "relation": triple.relation,
                            "effect": triple.effect,
                            "source_doc_id": triple.source_doc_id,
                            "confidence": triple.confidence,
                        }
                        for triple in triples
                    ],
                },
            }
        )

        # Step 5: insert/update the causal knowledge graph.
        step_start = time.perf_counter()
        self.knowledge_graph.update_graph(triples)
        timing.graph_construction_ms = (time.perf_counter() - step_start) * 1000
        
        trace_steps.append(
            {
                "step": 5,
                "name": "Update Causal Knowledge Graph",
                "details": {
                    "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
                    "graph_edges": self.knowledge_graph.graph.number_of_edges(),
                    "edge_preview": self._graph_edge_preview(limit=8),
                },
            }
        )

        chain = initialize_chain(decomposition)
        add_evidence_to_chain(chain=chain, triples=triples, evidence_results=initial_results)

        # Step 6: select candidate causal paths A -> ... -> D.
        step_start = time.perf_counter()
        candidate_paths = self._select_candidate_paths(source_node=source_node, target_node=target_node, decomposition=decomposition)
        timing.candidate_path_ms = (time.perf_counter() - step_start) * 1000
        
        trace_steps.append({"step": 6, "name": "Select Candidate Causal Path", "details": {"candidate_paths": candidate_paths}})
        if candidate_paths:
            chain.nodes = derive_best_chain_path(chain=chain, target_path=choose_best_path(candidate_paths, decomposition))

        # Iterative hops tracking
        hops_start = time.perf_counter()
        hop_count = 0
        while hop_count < self.config.pipeline.max_hops:
            # Step 8: detect whether a missing causal link still exists.
            missing_link = detect_missing_link(
                knowledge_graph=self.knowledge_graph,
                chain_nodes=chain.nodes,
                decomposition=decomposition,
                original_query=query,
            )
            trace_steps.append(
                {
                    "step": 8,
                    "name": "Check Missing Link",
                    "details": (
                        {
                            "left_node": missing_link.left_node,
                            "right_node": missing_link.right_node,
                            "retrieval_query": missing_link.retrieval_query,
                            "reason": missing_link.reason,
                        }
                        if missing_link
                        else {"missing_link": None}
                    ),
                }
            )

            # Step 7 and Step 9: retrieve the next hop, or retrieve again if a link is missing.
            hop_results = self._retrieve_next_iteration(
                query=query,
                retrieval_queries=retrieval_queries,
                chain=chain,
                missing_link=missing_link,
                target_node=target_node,
            )
            trace_steps.append(
                {
                    "step": 7 if missing_link else 9,
                    "name": "Retrieve Evidence For Next Hop",
                    "details": {
                        "hop": hop_count + 1,
                        "hits": [
                            {
                                "doc_id": result.document.doc_id,
                                "score": result.score,
                                "source": result.document.source,
                            }
                            for result in hop_results
                        ],
                    },
                }
            )
            if hop_results:
                source_results.extend(hop_results)
                hop_documents = [result.document for result in hop_results]
                hop_triples = deduplicate_triples(extract_triples_from_documents(hop_documents))
                self.knowledge_graph.update_graph(hop_triples)
                trace_steps.append(
                    {
                        "step": 10,
                        "name": "Add Evidence To Chain",
                        "details": {
                            "hop": hop_count + 1,
                            "new_triples": [
                                {
                                    "cause": triple.cause,
                                    "relation": triple.relation,
                                    "effect": triple.effect,
                                    "source_doc_id": triple.source_doc_id,
                                }
                                for triple in hop_triples
                            ],
                            "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
                            "graph_edges": self.knowledge_graph.graph.number_of_edges(),
                            "edge_preview": self._graph_edge_preview(limit=8),
                        },
                    }
                )

                # Step 10: add evidence from the new hop into the chain.
                add_evidence_to_chain(chain=chain, triples=hop_triples, evidence_results=hop_results)

            candidate_paths = self._select_candidate_paths(source_node=source_node, target_node=target_node, decomposition=decomposition)
            if candidate_paths:
                chain.nodes = derive_best_chain_path(chain=chain, target_path=choose_best_path(candidate_paths, decomposition))

            # Step 11: run counterfactual and consistency checks.
            chain.validations = self._validate_chain(chain=chain, source_node=source_node, target_node=target_node)
            trace_steps.append({"step": 11, "name": "Counterfactual / Consistency Check", "details": chain.validations})

            # Step 12 and Step 13: terminate only if the chain is complete, otherwise continue retrieval.
            chain_complete = self._is_chain_complete(chain=chain, source_node=source_node, target_node=target_node)
            trace_steps.append(
                {
                    "step": 12,
                    "name": "Check Chain Complete",
                    "details": {
                        "chain_complete": chain_complete,
                        "current_chain": chain.nodes,
                    },
                }
            )
            if chain_complete:
                break
            if not hop_results:
                break
            trace_steps.append({"step": 13, "name": "Continue Retrieval", "details": {"next_hop": hop_count + 2}})
            hop_count += 1
        
        timing.iterative_hops_ms = (time.perf_counter() - hops_start) * 1000

        # Answer generation
        step_start = time.perf_counter()
        answer = generate_answer(query=query, causal_chain=chain, app_config=self.config, chat_history=chat_history)
        timing.answer_generation_ms = (time.perf_counter() - step_start) * 1000
        
        provenance = build_provenance_payload(answer=answer, chain=chain, retrieval_queries=retrieval_queries)
        evidence_paths = self._derive_paths_from_evidence(chain=chain)
        provenance["retrieved_causal_chains"] = candidate_paths or evidence_paths
        trace_steps.append(
            {
                "step": 14,
                "name": "Return Answer With Provenance",
                "details": {
                    "answer_preview": answer[:240],
                    "retrieval_queries": provenance.get("retrieval_queries", []),
                    "document_sources": provenance.get("document_sources", []),
                    "causal_path": provenance.get("causal_path", []),
                    "retrieved_causal_chains": provenance.get("retrieved_causal_chains", []),
                },
            }
        )
        
        timing.total_ms = (time.perf_counter() - timing_start) * 1000
        
        return MultiHopResult(
            query=query,
            decomposition=decomposition,
            retrieval_queries=retrieval_queries,
            candidate_paths=candidate_paths or evidence_paths,
            causal_chain=chain,
            answer=answer,
            provenance=provenance,
            trace_steps=trace_steps,
            timing=timing,
        )

    def _retrieve_documents_for_queries(self, queries: Sequence[str]) -> List[RetrievalResult]:
        """Retrieve and merge evidence hits across multiple LLM-generated queries."""

        merged_results: Dict[str, RetrievalResult] = {}
        for retrieval_query in queries:
            for result in self.retriever.retrieve_documents(query=retrieval_query, top_k=self.config.retrieval.top_k):
                doc_id = result.document.doc_id
                existing = merged_results.get(doc_id)
                if existing is None or result.score > existing.score:
                    merged_results[doc_id] = result

        return sorted(merged_results.values(), key=lambda item: item.score, reverse=True)

    def _select_candidate_paths(
        self,
        source_node: Optional[str],
        target_node: Optional[str],
        decomposition: Dict[str, Optional[str]],
    ) -> List[List[str]]:
        """Select candidate graph paths between the decomposed source and target events."""

        if source_node and target_node:
            return bfs_candidate_paths(
                knowledge_graph=self.knowledge_graph,
                start=source_node,
                target=target_node,
                max_depth=self.config.pipeline.max_path_depth,
                max_paths=self.config.pipeline.max_candidate_paths,
            )
        return []

    def _derive_paths_from_evidence(self, chain: EvidenceChain) -> List[List[str]]:
        """Build displayable causal chains strictly from extracted evidence triples."""

        graph = nx.DiGraph()
        for triple in chain.triples:
            if triple.cause and triple.effect:
                graph.add_edge(triple.cause, triple.effect)

        if graph.number_of_edges() == 0:
            return []

        max_depth = max(1, self.config.pipeline.max_path_depth)
        max_paths = max(1, self.config.pipeline.max_candidate_paths)
        roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
        leaves = [node for node in graph.nodes if graph.out_degree(node) == 0]

        candidate_paths: List[List[str]] = []
        path_keys: set[tuple[str, ...]] = set()

        for root in roots:
            for leaf in leaves:
                if root == leaf:
                    continue
                try:
                    for path in nx.all_simple_paths(graph, source=root, target=leaf, cutoff=max_depth):
                        if len(path) < 2:
                            continue
                        key = tuple(path)
                        if key in path_keys:
                            continue
                        path_keys.add(key)
                        candidate_paths.append(path)
                        if len(candidate_paths) >= max_paths:
                            return candidate_paths
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        if candidate_paths:
            return candidate_paths

        edge_paths: List[List[str]] = []
        for source, target in graph.edges():
            edge_path = [str(source), str(target)]
            key = tuple(edge_path)
            if key in path_keys:
                continue
            path_keys.add(key)
            edge_paths.append(edge_path)
            if len(edge_paths) >= max_paths:
                break
        return edge_paths

    def _retrieve_next_iteration(
        self,
        query: str,
        retrieval_queries: Sequence[str],
        chain: EvidenceChain,
        missing_link: Optional[MissingLink],
        target_node: Optional[str],
    ) -> List[RetrievalResult]:
        """Retrieve evidence for the next best hop in the chain."""

        if missing_link is not None:
            missing_link_queries = [
                missing_link.retrieval_query,
                *[
                    f"{subquery}. Focus on how {missing_link.left_node} leads to {missing_link.right_node}."
                    for subquery in retrieval_queries
                ],
            ]
            return self._retrieve_hop_results_for_queries(
                hop_queries=missing_link_queries,
                current_node=missing_link.left_node,
                next_node=missing_link.right_node,
            )

        if not chain.nodes:
            return []

        current_node = chain.nodes[-1]
        next_node = canonicalize_event_text(target_node) if target_node else ""
        hop_queries = [query, *retrieval_queries]
        return self._retrieve_hop_results_for_queries(
            hop_queries=hop_queries,
            current_node=current_node,
            next_node=next_node,
        )

    def _retrieve_hop_results_for_queries(
        self,
        hop_queries: Sequence[str],
        current_node: str,
        next_node: str,
    ) -> List[RetrievalResult]:
        """Run hop retrieval across multiple subqueries and merge deduplicated hits."""

        merged_results: Dict[str, RetrievalResult] = {}
        unique_queries = list(dict.fromkeys(query for query in hop_queries if query and query.strip()))

        for hop_query in unique_queries:
            hop_result = retrieve_evidence_for_next_hop(
                retriever=self.retriever,
                original_query=hop_query,
                current_node=current_node,
                next_node=next_node,
                top_k=self.config.retrieval.hop_top_k,
            )
            for result in hop_result.results:
                doc_id = result.document.doc_id
                existing = merged_results.get(doc_id)
                if existing is None or result.score > existing.score:
                    merged_results[doc_id] = result

        return sorted(merged_results.values(), key=lambda item: item.score, reverse=True)

    def _validate_chain(
        self,
        chain: EvidenceChain,
        source_node: Optional[str],
        target_node: Optional[str],
    ) -> Dict[str, object]:
        """Run counterfactual and consistency validation for the current chain."""

        validations: Dict[str, object] = {
            "consistency": check_chain_consistency(knowledge_graph=self.knowledge_graph, chain=chain),
        }
        if source_node and target_node and source_node in self.knowledge_graph.graph and target_node in self.knowledge_graph.graph:
            validations["counterfactual"] = validate_counterfactual_chain(
                knowledge_graph=self.knowledge_graph,
                chain=chain,
                source_node=canonicalize_event_text(source_node),
                target_node=canonicalize_event_text(target_node),
            )
        return validations

    def _is_chain_complete(
        self,
        chain: EvidenceChain,
        source_node: Optional[str],
        target_node: Optional[str],
    ) -> bool:
        """Check whether the reasoning chain is complete and graph-supported."""

        if not source_node or not target_node:
            return False
        normalized_source = canonicalize_event_text(source_node)
        normalized_target = canonicalize_event_text(target_node)
        if normalized_source not in self.knowledge_graph.graph or normalized_target not in self.knowledge_graph.graph:
            return False
        if not check_chain_complete(chain=chain, source_node=source_node, target_node=target_node):
            return False
        try:
            return nx.has_path(self.knowledge_graph.graph, normalized_source, normalized_target)
        except nx.NodeNotFound:
            return False

    def _graph_edge_preview(self, limit: int = 8) -> List[Dict[str, object]]:
        """Return a compact preview of graph edges for trace output."""

        preview: List[Dict[str, object]] = []
        for source, target, attrs in list(self.knowledge_graph.graph.edges(data=True))[:limit]:
            preview.append(
                {
                    "source": source,
                    "target": target,
                    "relation": attrs.get("relation"),
                    "confidence": attrs.get("confidence"),
                }
            )
        return preview
