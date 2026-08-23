"""Evaluation metrics for baseline RAG vs multi-hop causal RAG benchmarking."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence, Set


def _normalize_text(text: str) -> str:
    """Normalize text for stable lexical comparison across metrics."""

    normalized = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _tokens(text: str) -> list[str]:
    """Tokenize text with lightweight normalization."""

    return [token for token in _normalize_text(text).split() if token]


def precision_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Compute Precision@K for retrieved evidence identifiers."""

    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Compute Recall@K for retrieved evidence identifiers."""

    if not ground_truth:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return hits / len(ground_truth)


def causal_chain_completeness(pred_chain: Sequence[str], true_chain: Sequence[str]) -> float:
    """Compute CCCS as the ratio of correctly matched causal links."""

    if len(true_chain) < 2:
        return 0.0

    pred_edges = {
        (pred_chain[i].strip().lower(), pred_chain[i + 1].strip().lower())
        for i in range(len(pred_chain) - 1)
        if pred_chain[i].strip() and pred_chain[i + 1].strip()
    }
    true_edges = {
        (true_chain[i].strip().lower(), true_chain[i + 1].strip().lower())
        for i in range(len(true_chain) - 1)
        if true_chain[i].strip() and true_chain[i + 1].strip()
    }
    if not true_edges:
        return 0.0

    correct_links = len(pred_edges & true_edges)
    return correct_links / len(true_edges)


def multi_hop_accuracy(pred_chain: Sequence[str], true_chain: Sequence[str]) -> float:
    """Return 1.0 only if the full multi-hop chain matches exactly."""

    pred_normalized = [node.strip().lower() for node in pred_chain if node and node.strip()]
    true_normalized = [node.strip().lower() for node in true_chain if node and node.strip()]
    if not pred_normalized or not true_normalized:
        return 0.0
    return 1.0 if pred_normalized == true_normalized else 0.0


def exact_match(pred: str, true: str) -> float:
    """Compute exact match on normalized answers."""

    return 1.0 if _normalize_text(pred) == _normalize_text(true) else 0.0


def f1_score(pred: str, true: str) -> float:
    """Compute token-level F1 between generated and gold answers."""

    pred_tokens = _tokens(pred)
    true_tokens = _tokens(true)
    if not pred_tokens or not true_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    true_counts = Counter(true_tokens)
    overlap = sum(min(pred_counts[token], true_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(true_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def hallucination_rate(pred: str, evidence: Sequence[str]) -> float:
    """Estimate hallucination rate as unsupported statements over total statements."""

    statements = [segment.strip() for segment in re.split(r"[.!?]+", pred or "") if segment.strip()]
    if not statements:
        return 0.0

    evidence_corpus = _normalize_text(" ".join(evidence))
    if not evidence_corpus:
        return 1.0

    unsupported = 0
    for statement in statements:
        tokens = [token for token in _tokens(statement) if len(token) >= 4]
        if not tokens:
            continue
        supported_hits = sum(1 for token in tokens if token in evidence_corpus)
        support_ratio = supported_hits / len(tokens)
        if support_ratio < 0.35:
            unsupported += 1

    return unsupported / len(statements)


def evidence_attribution_accuracy(pred_sources: Sequence[str], true_sources: Sequence[str]) -> float:
    """Measure overlap between predicted and expected evidence/source identifiers."""

    predicted = {source.strip().lower() for source in pred_sources if source and source.strip()}
    expected = {source.strip().lower() for source in true_sources if source and source.strip()}
    if not expected:
        return 0.0
    return len(predicted & expected) / len(expected)


def cross_doc_coherence(chain: Sequence[str]) -> float:
    """Estimate chain coherence from lexical continuity between consecutive hops."""

    if len(chain) < 2:
        return 0.0

    pair_scores: list[float] = []
    for left, right in zip(chain, chain[1:]):
        left_tokens = {token for token in _tokens(left) if len(token) >= 3}
        right_tokens = {token for token in _tokens(right) if len(token) >= 3}
        if not left_tokens or not right_tokens:
            pair_scores.append(0.0)
            continue
        overlap = len(left_tokens & right_tokens)
        denom = max(1, len(left_tokens | right_tokens))
        pair_scores.append(overlap / denom)

    if not pair_scores:
        return 0.0
    return sum(pair_scores) / len(pair_scores)


def overall_score(cccs: float, cdcs: float, f1: float) -> float:
    """Compute OMRS final score using weighted research metric blend."""

    return (0.4 * cccs) + (0.3 * cdcs) + (0.3 * f1)
