"""CLI entry point for the benchmark and evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark_runner import run_benchmark


def main() -> None:
    """Run benchmark from command line."""

    parser = argparse.ArgumentParser(description="Run baseline vs multi-hop causal RAG benchmark.")
    parser.add_argument("--max-samples-per-dataset", type=int, default=25, help="Max samples to load per dataset")
    parser.add_argument("--top-k", type=int, default=5, help="Shared top-k for retrieval")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for JSON/CSV logs")
    args = parser.parse_args()

    run_benchmark(
        max_samples_per_dataset=max(1, args.max_samples_per_dataset),
        top_k=max(1, args.top_k),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
