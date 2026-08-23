"""Build and update causal evidence chains across retrieval hops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from ..causal_extraction.triple_builder import CausalTriple, canonicalize_event_text
from ..retrieval.vector_index import RetrievalResult


@dataclass
class EvidenceChain:
    """State container for the evolving multi-hop causal chain."""

    nodes: List[str] = field(default_factory=list)
    triples: List[CausalTriple] = field(default_factory=list)
    evidence: List[RetrievalResult] = field(default_factory=list)
    validations: Dict[str, object] = field(default_factory=dict)


def initialize_chain(decomposition: Dict[str, str | None]) -> EvidenceChain:
    """Initialize the reasoning chain from decomposed causal components."""

    ordered_nodes = [canonicalize_event_text(value) for value in decomposition.values() if value]
    unique_nodes = list(dict.fromkeys(node for node in ordered_nodes if node))
    return EvidenceChain(nodes=unique_nodes)


def add_evidence_to_chain(
    chain: EvidenceChain,
    triples: Sequence[CausalTriple],
    evidence_results: Sequence[RetrievalResult],
) -> EvidenceChain:
    """Merge new triples and retrieved evidence into the chain state."""

    for triple in triples:
        if (triple.cause, triple.relation, triple.effect) not in {
            (existing.cause, existing.relation, existing.effect) for existing in chain.triples
        }:
            chain.triples.append(triple)
        for node in (triple.cause, triple.effect):
            if node not in chain.nodes:
                chain.nodes.append(node)

    existing_doc_ids = {result.document.doc_id for result in chain.evidence}
    for result in evidence_results:
        if result.document.doc_id not in existing_doc_ids:
            chain.evidence.append(result)
            existing_doc_ids.add(result.document.doc_id)

    return chain


def derive_best_chain_path(chain: EvidenceChain, target_path: Sequence[str]) -> List[str]:
    """Project the chain onto a preferred target ordering when available."""

    normalized_target = [canonicalize_event_text(node) for node in target_path if node]
    projected = [node for node in normalized_target if node in chain.nodes]
    for node in chain.nodes:
        if node not in projected:
            projected.append(node)
    return projected


def check_chain_complete(chain: EvidenceChain, source_node: str | None, target_node: str | None) -> bool:
    """Return whether the chain contains both start and end events and at least one causal edge."""

    if not source_node or not target_node:
        return False
    normalized_source = canonicalize_event_text(source_node)
    normalized_target = canonicalize_event_text(target_node)
    has_endpoints = normalized_source in chain.nodes and normalized_target in chain.nodes
    has_edges = any(triple.cause == normalized_source or triple.effect == normalized_target for triple in chain.triples)
    return has_endpoints and has_edges
