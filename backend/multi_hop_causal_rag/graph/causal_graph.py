"""Directed causal knowledge graph backed by NetworkX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
from networkx.readwrite import json_graph

from ..causal_extraction.triple_builder import CausalTriple, canonicalize_event_text


class CausalKnowledgeGraph:
    """A directed graph of events linked by extracted causal relations."""

    def __init__(self) -> None:
        """Initialize an empty directed graph."""

        self.graph = nx.DiGraph()

    def add_triple(self, triple: CausalTriple) -> None:
        """Insert a single causal triple into the graph."""

        cause = canonicalize_event_text(triple.cause)
        effect = canonicalize_event_text(triple.effect)

        self.graph.add_node(cause)
        self.graph.add_node(effect)

        existing = self.graph.get_edge_data(cause, effect, default={})
        evidence = existing.get("evidence", [])
        evidence.append(
            {
                "relation": triple.relation,
                "source_doc_id": triple.source_doc_id,
                "confidence": triple.confidence,
                "metadata": triple.metadata,
            }
        )
        self.graph.add_edge(
            cause,
            effect,
            relation=triple.relation,
            confidence=max(triple.confidence, float(existing.get("confidence", 0.0))),
            evidence=evidence,
        )

    def update_graph(self, triples: List[CausalTriple]) -> None:
        """Insert a batch of triples into the graph."""

        for triple in triples:
            self.add_triple(triple)

    def get_neighbors(self, node: str) -> Dict[str, Dict]:
        """Return outgoing neighbors for a node with edge attributes."""

        canonical_node = canonicalize_event_text(node)
        return {neighbor: self.graph.get_edge_data(canonical_node, neighbor) for neighbor in self.graph.successors(canonical_node)}

    def has_edge(self, cause: str, effect: str) -> bool:
        """Return whether a causal edge already exists between two events."""

        return self.graph.has_edge(canonicalize_event_text(cause), canonicalize_event_text(effect))

    def edge_evidence(self, cause: str, effect: str) -> List[Dict]:
        """Return the recorded evidence for a causal edge."""

        data = self.graph.get_edge_data(canonicalize_event_text(cause), canonicalize_event_text(effect), default={})
        return list(data.get("evidence", []))

    def save_state(self, path: str) -> str:
        """Persist the full causal graph state to a JSON file."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep a stable on-disk schema even as NetworkX defaults evolve.
        try:
            payload = json_graph.node_link_data(self.graph, edges="edges")
        except TypeError:
            payload = json_graph.node_link_data(self.graph)
            if "links" in payload and "edges" not in payload:
                payload["edges"] = payload.pop("links")
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target)

    def load_state(self, path: str) -> bool:
        """Load graph state from disk if it exists; return True when loaded."""

        source = Path(path)
        if not source.exists():
            return False

        payload = json.loads(source.read_text(encoding="utf-8"))

        edge_key = "links" if "links" in payload else "edges"
        try:
            loaded_graph = json_graph.node_link_graph(payload, directed=True, multigraph=False, edges=edge_key)
        except TypeError:
            if edge_key == "edges" and "links" not in payload:
                payload["links"] = payload.pop("edges")
            loaded_graph = json_graph.node_link_graph(payload, directed=True, multigraph=False)
        self.graph = nx.DiGraph(loaded_graph)
        return True
