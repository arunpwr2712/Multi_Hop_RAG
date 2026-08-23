from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from multi_hop_causal_rag.config import build_default_config
from multi_hop_causal_rag.evaluation.benchmark_runner import load_eval_dataset
from multi_hop_causal_rag.retrieval.retriever import CausalRetriever

from models.baseline_rag import BaselineRAGModel
from models.causal_extraction import CausalExtractionOnlyModel
from models.causal_graph import CausalGraphModel
from models.full_model import FullCausalRAGModel
from models.iterative_rag import IterativeRAGModel
from pipeline.evaluator import evaluate_model
from pipeline.logger import ExperimentLogger


def run_ablation(config_path: Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    results_dir = PROJECT_ROOT / str(config["paths"]["results_dir"])
    logger = ExperimentLogger(log_path=results_dir / "logs.txt")

    logger.info("Starting ablation study")
    logger.info(f"Using random seed: {seed}")

    retriever = _build_retriever(config)
    dataset = _load_dataset(config, retriever, logger)

    retrieval_cfg = config["retrieval"]
    model_registry = {
        "V1": BaselineRAGModel(retriever=retriever, top_k=retrieval_cfg["top_k"]),
        "V2": IterativeRAGModel(
            retriever=retriever,
            top_k=retrieval_cfg["top_k"],
            hop_top_k=retrieval_cfg["iterative_top_k"],
            max_hops=retrieval_cfg["max_hops"],
        ),
        "V3": CausalExtractionOnlyModel(retriever=retriever, top_k=retrieval_cfg["top_k"]),
        "V4": CausalGraphModel(
            retriever=retriever,
            top_k=retrieval_cfg["top_k"],
            max_path_depth=retrieval_cfg["max_path_depth"],
            max_candidate_paths=retrieval_cfg["max_candidate_paths"],
        ),
        "V5": FullCausalRAGModel(
            retriever=retriever,
            top_k=retrieval_cfg["top_k"],
            hop_top_k=retrieval_cfg["iterative_top_k"],
            max_hops=retrieval_cfg["max_hops"],
            max_path_depth=retrieval_cfg["max_path_depth"],
            max_candidate_paths=retrieval_cfg["max_candidate_paths"],
        ),
    }

    selected = [name for name, enabled in config["models"].items() if enabled]
    evaluations: Dict[str, Any] = {}
    for model_name in selected:
        evaluation = evaluate_model(
            model_name=model_name,
            model_fn=model_registry[model_name].predict,
            dataset=dataset,
            k=int(retrieval_cfg["top_k"]),
            logger=logger,
        )
        evaluations[model_name] = evaluation

    metrics_payload = {
        "config": config,
        "dataset_size": len(dataset),
        "models": evaluations,
    }

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    table_rows = _build_ablation_rows(evaluations)
    table_path = results_dir / "ablation_table.csv"
    _write_table_csv(table_rows, table_path)

    logger.info(f"Saved metrics: {metrics_path}")
    logger.info(f"Saved table  : {table_path}")

    table_text = _render_terminal_table(table_rows)
    print("\nFinal Ablation Table")
    print(table_text)

    return {
        "metrics_path": str(metrics_path),
        "table_path": str(table_path),
        "table": table_rows,
        "table_text": table_text,
        "dataset_size": len(dataset),
    }


def _load_config(config_path: Path) -> Dict[str, Any]:
    raw = config_path.read_text(encoding="utf-8")
    return json.loads(raw)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_retriever(config: Dict[str, Any]) -> CausalRetriever:
    app_config = build_default_config()
    app_config.retrieval.top_k = int(config["retrieval"]["top_k"])
    app_config.retrieval.hop_top_k = int(config["retrieval"]["iterative_top_k"])
    app_config.retrieval.faiss_index_path = str(BACKEND_ROOT / "faiss_index_512" / "index.faiss")
    app_config.retrieval.faiss_docstore_path = str(BACKEND_ROOT / "faiss_index_512" / "index.pkl")

    retriever = CausalRetriever(config=app_config.retrieval)
    retriever.initialize_with_documents([])
    return retriever


def _load_dataset(config: Dict[str, Any], retriever: CausalRetriever, logger: ExperimentLogger) -> List[Dict[str, Any]]:
    indexed_hotpot_ids = {
        str(doc.metadata.get("question_id", "")).strip()
        for doc in retriever.vector_index.documents
        if isinstance(doc.metadata, dict) and str(doc.metadata.get("source", "")).strip().lower() == "hotpotqa"
    }
    indexed_hotpot_ids.discard("")

    max_samples = int(config["dataset"]["max_samples_per_dataset"])
    dataset = load_eval_dataset(max_samples_per_dataset=max_samples, indexed_hotpot_ids=indexed_hotpot_ids)
    if not dataset:
        raise RuntimeError("Evaluation dataset is empty. Check local HotpotQA/WorldTree files.")

    logger.info(f"Loaded dataset with {len(dataset)} total samples")
    return dataset


def _build_ablation_rows(evaluations: Dict[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_name in ["V1", "V2", "V3", "V4", "V5"]:
        if model_name not in evaluations:
            continue
        summary = evaluations[model_name]["summary"]
        rows.append(
            {
                "Model": model_name,
                "CCCS": f"{float(summary.get('CCCS', 0.0)):.4f}",
                "MH-Acc": f"{float(summary.get('MH-Acc', 0.0)):.4f}",
                "EM": f"{float(summary.get('EM', 0.0)):.4f}",
                "F1": f"{float(summary.get('F1', 0.0)):.4f}",
                "HR": f"{float(summary.get('HR', 0.0)):.4f}",
            }
        )
    return rows


def _write_table_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "CCCS", "MH-Acc", "EM", "F1", "HR"])
        writer.writeheader()
        writer.writerows(rows)


def _render_terminal_table(rows: List[Dict[str, str]]) -> str:
    header = "Model | CCCS   | MH-Acc | EM     | F1     | HR"
    divider = "-" * len(header)
    lines = [header, divider]
    for row in rows:
        lines.append(
            f"{row['Model']:<5} | {row['CCCS']:>6} | {row['MH-Acc']:>6} | {row['EM']:>6} | {row['F1']:>6} | {row['HR']:>6}"
        )
    return "\n".join(lines)
