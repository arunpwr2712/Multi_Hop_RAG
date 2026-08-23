from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from .logger import ExperimentLogger
from .metrics import (
    cccs,
    cross_document_coherence_score,
    exact_match,
    f1_score,
    hallucination_rate,
    mean,
    mh_acc,
    normalize,
    precision_at_k,
    recall_at_k,
)


ModelFn = Callable[[str], Dict[str, Any]]


def evaluate_model(
    model_name: str,
    model_fn: ModelFn,
    dataset: Sequence[Dict[str, Any]],
    k: int,
    logger: ExperimentLogger,
) -> Dict[str, Any]:
    logger.info(f"Evaluating {model_name} on {len(dataset)} samples")

    per_case: List[Dict[str, Any]] = []
    aggregate: Dict[str, List[float]] = {
        "Precision@K": [],
        "Recall@K": [],
        "EM": [],
        "F1": [],
        "HR": [],
        "CCCS": [],
        "MH-Acc": [],
        "CDCS": [],
    }

    for index, sample in enumerate(dataset):
        query = str(sample.get("query", ""))
        gold_answer = str(sample.get("answer", ""))

        output = model_fn(query)
        pred_answer = str(output.get("answer", ""))
        retrieved_texts = [str(item) for item in output.get("retrieved_texts", []) if str(item).strip()]
        hr_evidence_texts = [str(item) for item in output.get("hr_evidence_texts", retrieved_texts) if str(item).strip()]
        pred_chain = [str(item) for item in output.get("causal_chain", []) if str(item).strip()]

        gold_chain = [str(item) for item in sample.get("causal_chain", []) if str(item).strip()]
        if len(gold_chain) < 2:
            gold_chain = _proxy_chain_from_sample(sample)

        gt_ids = {str(item).strip() for item in sample.get("retrieval_ground_truth_ids", []) if str(item).strip()}
        pred_ids = [str(item).strip() for item in output.get("retrieved_question_ids", []) if str(item).strip()]

        if not gt_ids:
            evidence_items = [str(item).strip() for item in sample.get("evidence", []) if str(item).strip()]
            if evidence_items:
                gt_ids = set(evidence_items)
                pred_ids = [str(item).strip() for item in output.get("retrieved_doc_ids", []) if str(item).strip()]
            else:
                evidence_text = [str(item).strip() for item in sample.get("evidence_text", []) if str(item).strip()]
                gt_ids = {f"e{i}" for i in range(len(evidence_text))}
                pred_ids = _map_by_text_overlap(retrieved_texts, evidence_text)

        pred_ids = list(dict.fromkeys(item for item in pred_ids if item))

        projected_chain = _project_pred_chain_to_gold(pred_chain=pred_chain, gold_chain=gold_chain)

        score_precision = precision_at_k(pred_ids, gt_ids, k)
        score_recall = recall_at_k(pred_ids, gt_ids, k)
        score_em = exact_match(pred_answer, gold_answer)
        score_f1 = f1_score(pred_answer, gold_answer)
        score_hr = hallucination_rate(pred_answer, hr_evidence_texts)
        score_cccs = cccs(projected_chain, gold_chain)
        score_mh = mh_acc(projected_chain, gold_chain)
        score_cdcs = cross_document_coherence_score(retrieved_texts)

        aggregate["Precision@K"].append(score_precision)
        aggregate["Recall@K"].append(score_recall)
        aggregate["EM"].append(score_em)
        aggregate["F1"].append(score_f1)
        aggregate["HR"].append(score_hr)
        aggregate["CCCS"].append(score_cccs)
        aggregate["MH-Acc"].append(score_mh)
        aggregate["CDCS"].append(score_cdcs)

        if (index + 1) % 10 == 0 or index == 0 or index == len(dataset) - 1:
            logger.info(
                f"{model_name} sample {index + 1}/{len(dataset)} | "
                f"EM={score_em:.3f} F1={score_f1:.3f} CCCS={score_cccs:.3f}"
            )

        per_case.append(
            {
                "id": sample.get("id", f"sample-{index}"),
                "query": query,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "gold_chain": gold_chain,
                "pred_chain": pred_chain,
                "projected_chain": projected_chain,
                "metrics": {
                    "Precision@K": score_precision,
                    "Recall@K": score_recall,
                    "EM": score_em,
                    "F1": score_f1,
                    "HR": score_hr,
                    "CCCS": score_cccs,
                    "MH-Acc": score_mh,
                    "CDCS": score_cdcs,
                },
            }
        )

    summary = {metric: mean(values) for metric, values in aggregate.items()}
    logger.info(f"Completed {model_name}")
    return {"summary": summary, "per_case": per_case, "evaluated_samples": len(per_case), "k": k}


def _proxy_chain_from_sample(sample: Dict[str, Any]) -> List[str]:
    """Fallback chain: ordered supporting facts/evidence as causal proxy."""

    support_nodes = [str(item).strip() for item in sample.get("evidence_text", []) if str(item).strip()]
    if len(support_nodes) >= 2:
        return support_nodes[:4]

    query = str(sample.get("query", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    if query and answer:
        return [query, answer]
    if query:
        return [query]
    return []


def _text_overlap(a: str, b: str) -> float:
    ta = {token for token in normalize(a).split() if len(token) >= 3}
    tb = {token for token in normalize(b).split() if len(token) >= 3}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _map_by_text_overlap(retrieved_texts: Sequence[str], evidence_texts: Sequence[str]) -> List[str]:
    mapped: List[str] = []
    for rank, snippet in enumerate(retrieved_texts):
        assigned = None
        for idx, evidence in enumerate(evidence_texts):
            if _text_overlap(snippet, evidence) >= 0.25:
                assigned = f"e{idx}"
                break
        mapped.append(assigned or f"r{rank}")
    return mapped


def _project_pred_chain_to_gold(pred_chain: Sequence[str], gold_chain: Sequence[str]) -> List[str]:
    """Project predicted nodes to the closest gold nodes for robust chain-level scoring."""

    if not pred_chain or not gold_chain:
        return []

    projected: List[str] = []
    for pred_node in pred_chain:
        best_gold = None
        best_score = 0.0
        for gold_node in gold_chain:
            score = _node_match_score(pred_node, gold_node)
            if score > best_score:
                best_score = score
                best_gold = gold_node
        if best_gold is not None and best_score >= 0.2:
            if not projected or projected[-1] != best_gold:
                projected.append(best_gold)

    return projected


def _node_match_score(left: str, right: str) -> float:
    a = normalize(left)
    b = normalize(right)
    if not a or not b:
        return 0.0

    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    return _text_overlap(a, b)
