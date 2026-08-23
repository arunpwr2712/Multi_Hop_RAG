from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from typing import Iterable, List, Sequence, Set, Tuple


def normalize(text: str) -> str:
    value = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    return re.sub(r"\s+", " ", value).strip()


def tokens(text: str) -> List[str]:
    return [token for token in normalize(text).split() if token]


def precision_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    unique_ranked = list(dict.fromkeys(item for item in retrieved if item))
    top_k = unique_ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in ground_truth) / len(top_k)


def recall_at_k(retrieved: Sequence[str], ground_truth: Set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    unique_ranked = list(dict.fromkeys(item for item in retrieved if item))
    top_k = unique_ranked[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return min(1.0, hits / len(ground_truth))


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = tokens(pred)
    gold_tokens = tokens(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum(min(pred_counts[token], gold_counts[token]) for token in pred_counts)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def hallucination_rate(pred_answer: str, evidence_texts: Sequence[str]) -> float:
    statements = [segment.strip() for segment in re.split(r"[.!?]+", pred_answer or "") if segment.strip()]
    if not statements:
        return 0.0

    evidence_corpus = normalize(" ".join(evidence_texts))
    if not evidence_corpus:
        return 1.0

    unsupported = 0
    for statement in statements:
        statement_tokens = [token for token in tokens(statement) if len(token) >= 4]
        if not statement_tokens:
            continue
        hits = sum(1 for token in statement_tokens if token in evidence_corpus)
        support = hits / len(statement_tokens)
        if support < 0.35:
            unsupported += 1

    return unsupported / len(statements)


def chain_to_edges(chain: Sequence[str]) -> Set[Tuple[str, str]]:
    cleaned = [normalize(node) for node in chain if normalize(node)]
    return {(cleaned[i], cleaned[i + 1]) for i in range(len(cleaned) - 1)}


def cccs(pred_chain: Sequence[str], gold_chain: Sequence[str]) -> float:
    gold_edges = chain_to_edges(gold_chain)
    if not gold_edges:
        return 0.0
    pred_edges = chain_to_edges(pred_chain)
    return len(pred_edges & gold_edges) / len(gold_edges)


def mh_acc(pred_chain: Sequence[str], gold_chain: Sequence[str]) -> float:
    """Fraction of correctly retrieved reasoning hops in the gold order."""

    pred_edges = list(chain_to_edges(pred_chain))
    gold_edges = list(chain_to_edges(gold_chain))
    if not gold_edges:
        return 0.0
    pred_set = set(pred_edges)
    matched = sum(1 for edge in gold_edges if edge in pred_set)
    return matched / len(gold_edges)


def cross_document_coherence_score(retrieved_texts: Sequence[str]) -> float:
    """Approximate cross-document coherence from pairwise token overlap."""

    docs = [set(token for token in tokens(text) if len(token) >= 3) for text in retrieved_texts if text.strip()]
    if len(docs) < 2:
        return 0.0

    scores: List[float] = []
    for left, right in combinations(docs, 2):
        union = left | right
        if not union:
            continue
        scores.append(len(left & right) / len(union))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return float(sum(numbers) / len(numbers))
