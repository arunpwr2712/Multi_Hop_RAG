from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question for the multi-hop causal RAG pipeline")


class ChatResponse(BaseModel):
    query: str
    answer: str
    decomposition: Dict[str, Optional[str]]
    retrieval_queries: List[str]
    candidate_paths: List[List[str]]
    provenance: Dict[str, Any]
    trace_steps: List[Dict[str, Any]]


class GraphSummary(BaseModel):
    nodes: int
    edges: int
    is_directed: bool
    is_acyclic: bool


class GraphNode(BaseModel):
    id: str
    label: str


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    confidence: Any


class GraphResponse(BaseModel):
    summary: GraphSummary
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class LastResultResponse(BaseModel):
    data: Dict[str, Any] | None


class ResetResponse(BaseModel):
    ok: bool
    message: str


class ReadinessResponse(BaseModel):
    backend: bool
    ollama: bool
    ready: bool

