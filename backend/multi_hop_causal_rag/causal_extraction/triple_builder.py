"""Structured representations for extracted causal triples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CausalTriple:
    """A local causal relation extracted from retrieved evidence."""

    cause: str
    relation: str
    effect: str
    source_doc_id: Optional[str] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


def canonicalize_event_text(text: str) -> str:
    """Normalize an event description for graph insertion and matching."""

    return " ".join((text or "").strip().lower().split())


def build_triple(
    cause: str,
    relation: str,
    effect: str,
    source_doc_id: Optional[str] = None,
    confidence: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> CausalTriple:
    """Construct a normalized causal triple object."""

    return CausalTriple(
        cause=canonicalize_event_text(cause),
        relation=relation.strip().lower(),
        effect=canonicalize_event_text(effect),
        source_doc_id=source_doc_id,
        confidence=confidence,
        metadata=metadata or {},
    )
