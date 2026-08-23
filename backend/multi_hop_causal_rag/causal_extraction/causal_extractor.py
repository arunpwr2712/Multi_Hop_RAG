"""Pattern-based extraction of local causal triples from retrieved evidence."""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence

from ..data.preprocessing import Document, split_into_sentences
from .triple_builder import CausalTriple, build_triple


CAUSAL_PATTERNS = [
    (re.compile(r"(?P<cause>.+?)\s+(?:causes|cause|caused)\s+(?P<effect>.+)", re.IGNORECASE), "causes", 0.92),
    (re.compile(r"(?P<cause>.+?)\s+(?:leads to|lead to|led to|results in|result in|resulted in)\s+(?P<effect>.+)", re.IGNORECASE), "leads_to", 0.88),
    (re.compile(r"(?P<cause>.+?)\s+(?:increases|increase|raises|raise|elevates|elevate)\s+(?P<effect>.+)", re.IGNORECASE), "increases", 0.84),
    (re.compile(r"(?P<effect>.+?)\s+(?:is caused by|are caused by|results from|result from)\s+(?P<cause>.+)", re.IGNORECASE), "causes", 0.9),
    (re.compile(r"because of\s+(?P<cause>.+?),\s*(?P<effect>.+)", re.IGNORECASE), "causes", 0.72),
]


def extract_local_causal_triples(text: str, source_doc_id: str | None = None) -> List[CausalTriple]:
    """Extract causal triples from a text passage using local lexical patterns."""

    triples: List[CausalTriple] = []
    for sentence in split_into_sentences(text):
        cleaned_sentence = sentence.strip(" .;")
        for pattern, relation, confidence in CAUSAL_PATTERNS:
            match = pattern.search(cleaned_sentence)
            if not match:
                continue

            cause = _clean_span(match.group("cause"))
            effect = _clean_span(match.group("effect"))
            if not cause or not effect or cause == effect:
                continue

            triples.append(
                build_triple(
                    cause=cause,
                    relation=relation,
                    effect=effect,
                    source_doc_id=source_doc_id,
                    confidence=confidence,
                    metadata={"sentence": cleaned_sentence},
                )
            )
            break
    return triples


def extract_triples_from_documents(documents: Sequence[Document]) -> List[CausalTriple]:
    """Extract causal triples from a collection of retrieved documents."""

    triples: List[CausalTriple] = []
    for document in documents:
        triples.extend(extract_local_causal_triples(text=document.text, source_doc_id=document.doc_id))
    return triples


def deduplicate_triples(triples: Iterable[CausalTriple]) -> List[CausalTriple]:
    """Deduplicate triples by their normalized causal endpoints and relation."""

    seen = set()
    unique: List[CausalTriple] = []
    for triple in triples:
        key = (triple.cause, triple.relation, triple.effect)
        if key in seen:
            continue
        seen.add(key)
        unique.append(triple)
    return unique


def _clean_span(span: str) -> str:
    """Normalize an extracted cause or effect phrase."""

    span = re.sub(r"\s+", " ", span or "")
    return span.strip(" ,.;:")
