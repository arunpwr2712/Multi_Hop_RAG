"""Detection of missing causal links that require additional retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..causal_extraction.triple_builder import canonicalize_event_text
from ..graph.causal_graph import CausalKnowledgeGraph


@dataclass
class MissingLink:
    """Description of a missing edge or bridge node between causal components."""

    left_node: str
    right_node: str
    retrieval_query: str
    reason: str


def detect_missing_link(
    knowledge_graph: CausalKnowledgeGraph,
    chain_nodes: Sequence[str],
    decomposition: Dict[str, Optional[str]],
    original_query: str,
) -> Optional[MissingLink]:
    """Detect the next missing causal edge in the current reasoning chain."""

    filtered_chain = [node for node in chain_nodes if node]
    for left, right in zip(filtered_chain, filtered_chain[1:]):
        if not knowledge_graph.has_edge(left, right):
            return MissingLink(
                left_node=left,
                right_node=right,
                retrieval_query=f"{original_query}. Explain how {left} leads to {right}.",
                reason="graph_missing_edge",
            )

    ordered_components = [decomposition.get(key) for key in ("A", "B", "C", "D") if decomposition.get(key)]
    for left, right in zip(ordered_components, ordered_components[1:]):
        if not knowledge_graph.has_edge(left, right):
            return MissingLink(
                left_node=canonicalize_event_text(left),
                right_node=canonicalize_event_text(right),
                retrieval_query=f"Find evidence for the causal step from {left} to {right} in: {original_query}",
                reason="decomposition_gap",
            )

    return None


def chain_has_missing_links(knowledge_graph: CausalKnowledgeGraph, chain_nodes: List[str]) -> bool:
    """Return whether any adjacent pair in the chain lacks graph support."""

    return any(not knowledge_graph.has_edge(left, right) for left, right in zip(chain_nodes, chain_nodes[1:]))
