"""Backend-only benchmark runner for semantic top-k vs multi-hop causal retrieval.

Run from backend root:
    python -m multi_hop_causal_rag.evaluation.run_backend_benchmark --max-cases 30
"""

from __future__ import annotations

import argparse
import json
import platform
import psutil
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

if __package__ in {None, ""}:
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from multi_hop_causal_rag.config import build_default_config
    from multi_hop_causal_rag.data.hotpot_loader import load_hotpotqa
    from multi_hop_causal_rag.evaluation.metrics import precision_at_k, recall_at_k
    from multi_hop_causal_rag.pipeline.multi_hop_pipeline import MultiHopCausalPipeline
    from multi_hop_causal_rag.retrieval.retriever import CausalRetriever
    from multi_hop_causal_rag.retrieval.vector_index import RetrievalResult
else:
    from ..config import build_default_config
    from ..data.hotpot_loader import load_hotpotqa
    from ..evaluation.metrics import precision_at_k, recall_at_k
    from ..pipeline.multi_hop_pipeline import MultiHopCausalPipeline
    from ..retrieval.retriever import CausalRetriever
    from ..retrieval.vector_index import RetrievalResult


@dataclass
class EvalExample:
    """Minimal evaluation sample derived from HotpotQA."""

    question_id: str
    question: str
    answer: str
    gold_support_sentences: List[str]


def _normalize(text: str) -> str:
    """Lowercase and normalize whitespace for robust matching."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def _contains_answer(prediction: str, gold_answer: str) -> bool:
    """Approximate exact match by normalized substring containment."""

    pred = _normalize(prediction)
    gold = _normalize(gold_answer)
    if not pred or not gold:
        return False
    return gold in pred


def _unique_preserving_order(items: Iterable[str]) -> List[str]:
    """Remove duplicates while preserving first-seen ordering."""

    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _extract_question_id(result: RetrievalResult) -> str:
    """Read question_id from retrieval hit metadata."""

    return str(result.document.metadata.get("question_id", "")).strip()


def _sentence_token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard token overlap for approximate support matching."""

    tokens_a = {token for token in _normalize(text_a).split() if len(token) >= 3}
    tokens_b = {token for token in _normalize(text_b).split() if len(token) >= 3}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def _doc_matches_support(doc_text: str, support_sentence: str) -> bool:
    """Return whether a retrieved document chunk supports a gold supporting sentence."""

    normalized_doc = _normalize(doc_text)
    normalized_support = _normalize(support_sentence)
    if not normalized_doc or not normalized_support:
        return False
    if normalized_support in normalized_doc:
        return True
    return _sentence_token_overlap(normalized_doc, normalized_support) >= 0.55


def _covered_support_indices(results: Sequence[RetrievalResult], support_sentences: Sequence[str]) -> List[int]:
    """Find which gold support sentences are covered by retrieved evidence."""

    covered: List[int] = []
    for idx, support in enumerate(support_sentences):
        if any(_doc_matches_support(result.document.text, support) for result in results):
            covered.append(idx)
    return covered


def _relevant_hit_ids(results: Sequence[RetrievalResult], support_sentences: Sequence[str]) -> List[str]:
    """Create binary relevance labels for precision/recall metrics at hit level."""

    relevant: List[str] = []
    for hit_index, result in enumerate(results):
        if any(_doc_matches_support(result.document.text, support) for support in support_sentences):
            relevant.append(f"h{hit_index}")
    return relevant


def _build_baseline_answer(results: Sequence[RetrievalResult]) -> str:
    """Create deterministic baseline answer from top semantic hits only."""

    if not results:
        return "No strong evidence was retrieved by semantic top-k."

    snippets: List[str] = []
    for hit in results[:2]:
        title = str(hit.document.metadata.get("title", "Untitled")).strip() or "Untitled"
        text = " ".join(hit.document.text.split())
        snippets.append(f"{title}: {text[:260]}")
    return " ".join(snippets)


def _grounding_score(answer: str, evidence_texts: Sequence[str]) -> float:
    """Estimate answer grounding as lexical overlap with retrieved evidence."""

    answer_tokens = [tok for tok in _normalize(answer).split() if len(tok) >= 3]
    if not answer_tokens:
        return 0.0

    evidence_vocab: set[str] = set()
    for text in evidence_texts:
        evidence_vocab.update(tok for tok in _normalize(text).split() if len(tok) >= 3)
    if not evidence_vocab:
        return 0.0

    overlap = sum(1 for token in answer_tokens if token in evidence_vocab)
    return overlap / len(answer_tokens)


def _extract_support_sentences(record: Dict[str, object]) -> List[str]:
    """Extract gold supporting-fact sentences from Hotpot context."""

    context = record.get("context", [])
    supporting = record.get("supporting_facts", [])
    if not isinstance(context, list) or not isinstance(supporting, list):
        return []

    title_to_sentences: Dict[str, List[str]] = {}
    for item in context:
        if not isinstance(item, list) or len(item) != 2:
            continue
        title = str(item[0]).strip()
        sentences = item[1] if isinstance(item[1], list) else []
        title_to_sentences[title] = [str(sentence).strip() for sentence in sentences]

    support_sentences: List[str] = []
    for item in supporting:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = str(item[0]).strip()
        sentence_index_raw = item[1]
        if not isinstance(sentence_index_raw, int):
            continue
        sentences = title_to_sentences.get(title, [])
        if sentence_index_raw < 0 or sentence_index_raw >= len(sentences):
            continue
        sentence = sentences[sentence_index_raw].strip()
        if sentence:
            support_sentences.append(sentence)

    return _unique_preserving_order(support_sentences)


def _load_eval_examples(hotpot_paths: Sequence[Path], indexed_question_ids: set[str], max_cases: int) -> List[EvalExample]:
    """Load Hotpot examples whose question_id is present in the FAISS index."""

    examples: List[EvalExample] = []
    for hotpot_path in hotpot_paths:
        records = load_hotpotqa(path=str(hotpot_path), max_samples=None)
        for record in records:
            question_id = str(record.get("_id", "")).strip()
            if not question_id or question_id not in indexed_question_ids:
                continue

            question = str(record.get("question", "")).strip()
            answer = str(record.get("answer", "")).strip()
            support_sentences = _extract_support_sentences(record)
            if not question or not answer or len(support_sentences) < 2:
                continue

            examples.append(
                EvalExample(
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    gold_support_sentences=support_sentences,
                )
            )
            if len(examples) >= max_cases:
                return examples

    if not examples:
        raise RuntimeError("No question_id-aligned HotpotQA examples were found for this FAISS index.")
    return examples


def _avg(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _get_hardware_specs() -> Dict[str, object]:
    """Collect hardware and environment information."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }


def run_benchmark(
    hotpot_paths: Sequence[Path],
    max_cases: int,
    top_k: int,
    grounding_threshold: float,
    output_json: Path | None,
) -> None:
    """Execute objective-wise benchmark and print terminal report."""

    config = build_default_config()
    project_root = Path(__file__).resolve().parents[2]
    config.retrieval.faiss_index_path = str(project_root / "faiss_index_512" / "index.faiss")
    config.retrieval.faiss_docstore_path = str(project_root / "faiss_index_512" / "index.pkl")
    config.retrieval.top_k = top_k
    config.llm.require_ollama = False
    config.llm.query_planning_enabled = False

    retriever = CausalRetriever(config=config.retrieval)
    retriever.initialize_with_documents([])
    pipeline = MultiHopCausalPipeline(retriever=retriever, config=config)

    indexed_question_ids = {
        str(document.metadata.get("question_id", "")).strip()
        for document in retriever.vector_index.documents
        if isinstance(document.metadata, dict)
    }
    indexed_question_ids.discard("")

    examples = _load_eval_examples(hotpot_paths=hotpot_paths, indexed_question_ids=indexed_question_ids, max_cases=max_cases)

    baseline: Dict[str, List[float]] = {
        "factual_accuracy": [],
        "causal_completeness": [],
        "hallucination_rate": [],
        "multi_hop_quality": [],
        "precision_at_k": [],
        "recall_at_k": [],
        "latency_ms": [],
    }
    causal: Dict[str, List[float]] = {
        "factual_accuracy": [],
        "causal_completeness": [],
        "hallucination_rate": [],
        "multi_hop_quality": [],
        "precision_at_k": [],
        "recall_at_k": [],
        "latency_ms": [],
    }

    print("=" * 84)
    print("Multi-Hop Causal RAG Benchmark (Backend Only)")
    print("Comparing: Semantic Top-k Baseline vs Proposed Multi-Hop Causal Pipeline")
    print("=" * 84)
    print(f"Datasets: {', '.join(str(path) for path in hotpot_paths)}")
    print(f"Indexed question_ids available: {len(indexed_question_ids)}")
    print(f"Cases: {len(examples)}")
    print(f"Top-k (baseline): {top_k}")
    print()

    for idx, example in enumerate(examples, start=1):
        # if idx == 1 or idx % 5 == 0:
        #     print(f"[Progress] Evaluating case {idx}/{len(examples)}")
        print(f"[Progress] Evaluating case {idx}/{len(examples)}")

        support_sentences = example.gold_support_sentences
        total_support = len(support_sentences)
        relevant = {f"h{i}" for i in range(min(total_support, top_k))}

        # Baseline: one-shot semantic top-k retrieval.
        baseline_start = time.perf_counter()
        baseline_hits = retriever.retrieve_documents(query=example.question, top_k=top_k)
        baseline_latency_ms = (time.perf_counter() - baseline_start) * 1000
        
        baseline_answer = _build_baseline_answer(baseline_hits)
        baseline_evidence_text = " ".join(hit.document.text for hit in baseline_hits)
        baseline_fact = 1.0 if _contains_answer(baseline_evidence_text, example.answer) else 0.0
        baseline_covered = _covered_support_indices(baseline_hits, support_sentences)
        baseline_chain_completeness = len(baseline_covered) / max(1, total_support)
        baseline_multi_hop = min(1.0, len(baseline_covered) / 2.0)
        baseline_hallucinated = 1.0 if (baseline_fact == 0.0 and baseline_chain_completeness < grounding_threshold) else 0.0
        baseline_retrieved_ids = [f"h{i}" for i, _ in enumerate(_relevant_hit_ids(baseline_hits, support_sentences))]

        baseline["factual_accuracy"].append(baseline_fact)
        baseline["causal_completeness"].append(baseline_chain_completeness)
        baseline["multi_hop_quality"].append(baseline_multi_hop)
        baseline["hallucination_rate"].append(baseline_hallucinated)
        baseline["precision_at_k"].append(precision_at_k(baseline_retrieved_ids, relevant, max(1, top_k)))
        baseline["recall_at_k"].append(recall_at_k(baseline_retrieved_ids, relevant, max(1, top_k)))
        baseline["latency_ms"].append(baseline_latency_ms)

        # Proposed method: iterative causal multi-hop pipeline.
        causal_start = time.perf_counter()
        result = pipeline.run(example.question)
        causal_latency_ms = (time.perf_counter() - causal_start) * 1000
        
        causal_hits = result.causal_chain.evidence
        causal_answer = result.answer
        causal_evidence_text = " ".join(hit.document.text for hit in causal_hits)
        causal_fact = 1.0 if _contains_answer(causal_evidence_text, example.answer) else 0.0
        causal_covered = _covered_support_indices(causal_hits, support_sentences)
        causal_chain_score = len(causal_covered) / max(1, total_support)
        causal_multi_hop = min(1.0, len(causal_covered) / 2.0)
        causal_hallucinated = 1.0 if (causal_fact == 0.0 and causal_chain_score < grounding_threshold) else 0.0
        causal_retrieved_ids = [f"h{i}" for i, _ in enumerate(_relevant_hit_ids(causal_hits, support_sentences))]

        causal["factual_accuracy"].append(causal_fact)
        causal["causal_completeness"].append(causal_chain_score)
        causal["multi_hop_quality"].append(causal_multi_hop)
        causal["hallucination_rate"].append(causal_hallucinated)
        causal["precision_at_k"].append(precision_at_k(causal_retrieved_ids, relevant, max(1, top_k)))
        causal["recall_at_k"].append(recall_at_k(causal_retrieved_ids, relevant, max(1, top_k)))
        causal["latency_ms"].append(causal_latency_ms)

    baseline_avg = {metric: _avg(values) for metric, values in baseline.items()}
    causal_avg = {metric: _avg(values) for metric, values in causal.items()}

    def improvement(metric: str, lower_is_better: bool = False) -> float:
        b = baseline_avg[metric]
        c = causal_avg[metric]
        if lower_is_better:
            return _safe_div((b - c), b) if b > 0 else 0.0
        return _safe_div((c - b), b) if b > 0 else (1.0 if c > 0 else 0.0)

    print()
    print("Objective-wise Comparison")
    print("-" * 84)
    print(
        f"{'Metric':<24} {'Baseline':>12} {'Multi-Hop':>12} {'Relative Gain':>16}"
    )
    print("-" * 84)

    rows = [
        ("Factual Accuracy", "factual_accuracy", False),
        ("Causal Completeness", "causal_completeness", False),
        ("Hallucination Rate", "hallucination_rate", True),
        ("Multi-hop Quality", "multi_hop_quality", False),
        ("Precision@k", "precision_at_k", False),
        ("Recall@k", "recall_at_k", False),
    ]

    for label, key, lower_is_better in rows:
        rel = improvement(key, lower_is_better=lower_is_better)
        print(
            f"{label:<24} {_format_pct(baseline_avg[key]):>12} {_format_pct(causal_avg[key]):>12} {_format_pct(rel):>16}"
        )

    print("-" * 84)
    
    # Latency report
    print()
    print("Latency Comparison")
    print("-" * 84)
    baseline_latency_avg = _avg(baseline["latency_ms"])
    causal_latency_avg = _avg(causal["latency_ms"])
    latency_slowdown = _safe_div(causal_latency_avg, baseline_latency_avg) if baseline_latency_avg > 0 else 0.0
    print(f"{'Baseline Avg Latency':<24} {baseline_latency_avg:>12.2f} ms")
    print(f"{'Multi-Hop Avg Latency':<24} {causal_latency_avg:>12.2f} ms")
    print(f"{'Slowdown Factor':<24} {latency_slowdown:>12.2f}x")
    print("-" * 84)
    
    print()
    print(
        "Note: supporting-fact sentences (question_id aligned to FAISS docstore) are used as gold evidence."
    )

    if output_json is not None:
        hardware = _get_hardware_specs()
        payload = {
            "settings": {
                "datasets": [str(path) for path in hotpot_paths],
                "cases": len(examples),
                "top_k": top_k,
                "grounding_threshold": grounding_threshold,
            },
            "hardware": hardware,
            "baseline": baseline_avg,
            "multi_hop": causal_avg,
            "latency": {
                "baseline_ms": baseline_latency_avg,
                "multi_hop_ms": causal_latency_avg,
                "slowdown_factor": latency_slowdown,
                "baseline_latencies": baseline["latency_ms"],
                "multi_hop_latencies": causal["latency_ms"],
            },
            "relative_gain": {
                "factual_accuracy": improvement("factual_accuracy"),
                "causal_completeness": improvement("causal_completeness"),
                "hallucination_reduction": improvement("hallucination_rate", lower_is_better=True),
                "multi_hop_quality": improvement("multi_hop_quality"),
                "precision_at_k": improvement("precision_at_k"),
                "recall_at_k": improvement("recall_at_k"),
            },
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved benchmark report: {output_json}")


def _default_hotpot_dev_path() -> Path:
    """Resolve default Hotpot dev path from repository layout."""

    return Path(__file__).resolve().parents[2] / "datasets" / "raw" / "hotpotqa" / "dev.json"


def _default_hotpot_train_path() -> Path:
    """Resolve default Hotpot train path from repository layout."""

    return Path(__file__).resolve().parents[2] / "datasets" / "raw" / "hotpotqa" / "train.json"


def main() -> None:
    """Parse CLI arguments and run benchmark."""

    parser = argparse.ArgumentParser(description="Run backend-only benchmark for multi-hop causal RAG.")
    parser.add_argument("--hotpot-dev", type=Path, default=_default_hotpot_dev_path(), help="Path to HotpotQA dev.json")
    parser.add_argument("--hotpot-train", type=Path, default=_default_hotpot_train_path(), help="Path to HotpotQA train.json")
    parser.add_argument("--max-cases", type=int, default=25, help="Number of evaluation questions")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for semantic baseline retrieval")
    parser.add_argument(
        "--grounding-threshold",
        type=float,
        default=0.35,
        help="Threshold for flagging low-grounding wrong answers as hallucinations",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON report")
    args = parser.parse_args()

    if args.max_cases <= 0:
        raise ValueError("--max-cases must be > 0")
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    run_benchmark(
        hotpot_paths=[args.hotpot_dev, args.hotpot_train],
        max_cases=args.max_cases,
        top_k=args.top_k,
        grounding_threshold=args.grounding_threshold,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
