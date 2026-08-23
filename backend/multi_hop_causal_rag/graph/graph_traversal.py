"""Graph traversal helpers for candidate path discovery."""

from __future__ import annotations

from typing import List, Sequence

import networkx as nx

from ..causal_extraction.triple_builder import canonicalize_event_text
from .causal_graph import CausalKnowledgeGraph


def bfs_candidate_paths(
    knowledge_graph: CausalKnowledgeGraph,
    start: str,
    target: str,
    max_depth: int = 5,
    max_paths: int = 5,
) -> List[List[str]]:
    """Find candidate causal paths between two nodes using a bounded shortest-path search."""

    graph = knowledge_graph.graph
    source_node = canonicalize_event_text(start)
    target_node = canonicalize_event_text(target)
    if source_node not in graph or target_node not in graph:
        return []

    try:
        iterator = nx.shortest_simple_paths(graph, source=source_node, target=target_node)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    paths: List[List[str]] = []
    for path in iterator:
        if len(path) - 1 > max_depth:
            continue
        paths.append(path)
        if len(paths) >= max_paths:
            break
    return paths


def path_is_complete(path: Sequence[str], target: str) -> bool:
    """Return whether a path ends at the requested target node."""

    return bool(path) and canonicalize_event_text(path[-1]) == canonicalize_event_text(target)


def next_missing_edge(path: Sequence[str], knowledge_graph: CausalKnowledgeGraph) -> tuple[str, str] | None:
    """Return the first missing edge along a candidate path, if any."""

    for left, right in zip(path, path[1:]):
        if not knowledge_graph.has_edge(left, right):
            return left, right
    return None
