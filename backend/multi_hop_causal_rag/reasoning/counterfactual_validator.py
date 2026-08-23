"""Counterfactual checks for evaluating whether a reasoning chain is causally necessary."""

from __future__ import annotations

from typing import Dict, List

import networkx as nx

from ..graph.causal_graph import CausalKnowledgeGraph
from ..multi_hop.chain_builder import EvidenceChain


def validate_counterfactual_chain(
    knowledge_graph: CausalKnowledgeGraph,
    chain: EvidenceChain,
    source_node: str,
    target_node: str,
) -> Dict[str, object]:
    """Test whether removing intermediate nodes breaks connectivity from source to target."""

    graph = knowledge_graph.graph.copy()
    critical_nodes: List[str] = []

    for node in chain.nodes:
        if node in {source_node, target_node} or node not in graph:
            continue
        ablated_graph = graph.copy()
        ablated_graph.remove_node(node)
        try:
            still_connected = nx.has_path(ablated_graph, source_node, target_node)
        except nx.NodeNotFound:
            still_connected = False
        if not still_connected:
            critical_nodes.append(node)

    return {
        "is_counterfactually_supported": bool(critical_nodes) or source_node == target_node,
        "critical_nodes": critical_nodes,
        "criticality_ratio": (len(critical_nodes) / max(1, max(len(chain.nodes) - 2, 1))),
    }
