"""Consistency checks over the evolving causal reasoning chain."""

from __future__ import annotations

from typing import Dict, List

from ..graph.causal_graph import CausalKnowledgeGraph
from ..multi_hop.chain_builder import EvidenceChain


NEGATION_MARKERS = {"not", "never", "unlikely", "without", "prevents"}


def check_chain_consistency(knowledge_graph: CausalKnowledgeGraph, chain: EvidenceChain) -> Dict[str, object]:
    """Verify that each adjacent pair in the chain is supported and non-contradictory."""

    unsupported_edges: List[str] = []
    contradictory_edges: List[str] = []

    for triple in chain.triples:
        if not knowledge_graph.has_edge(triple.cause, triple.effect):
            unsupported_edges.append(f"{triple.cause} -> {triple.effect}")
        sentence = str(triple.metadata.get("sentence", "")).lower()
        if any(marker in sentence for marker in NEGATION_MARKERS):
            contradictory_edges.append(f"{triple.cause} -> {triple.effect}")

    return {
        "is_consistent": not unsupported_edges and not contradictory_edges,
        "unsupported_edges": unsupported_edges,
        "contradictory_edges": contradictory_edges,
    }
