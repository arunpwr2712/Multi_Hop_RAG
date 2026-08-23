"""Benchmark and evaluation package for multi-hop causal RAG."""

from .benchmark_runner import load_eval_dataset, run_benchmark

__all__ = ["load_eval_dataset", "run_benchmark"]
