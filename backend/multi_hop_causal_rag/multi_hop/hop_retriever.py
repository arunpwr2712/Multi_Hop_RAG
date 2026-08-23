"""Evidence retrieval focused on the next missing causal hop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..retrieval.retriever import CausalRetriever
from ..retrieval.vector_index import RetrievalResult


@dataclass
class HopRetrievalResult:
    """Retrieved evidence associated with a specific causal hop request."""

    hop_query: str
    current_node: str
    next_node: str
    results: List[RetrievalResult]


def build_hop_query(original_query: str, current_node: str, next_node: str) -> str:
    """Construct a retrieval query targeted at a specific missing causal transition."""

    if current_node and next_node:
        return f"{original_query} Evidence that {current_node} leads to {next_node}."
    if next_node:
        return f"{original_query} Evidence about {next_node}."
    return original_query


def retrieve_evidence_for_next_hop(
    retriever: CausalRetriever,
    original_query: str,
    current_node: str,
    next_node: str,
    top_k: Optional[int] = None,
) -> HopRetrievalResult:
    """Retrieve supporting evidence for the next step in the causal chain."""

    hop_query = build_hop_query(original_query=original_query, current_node=current_node, next_node=next_node)
    results = retriever.retrieve_documents(query=hop_query, top_k=top_k)
    return HopRetrievalResult(hop_query=hop_query, current_node=current_node, next_node=next_node, results=results)
