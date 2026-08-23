from __future__ import annotations

from pathlib import Path

from pipeline.runner import run_ablation


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent / "configs" / "experiment_config.yaml"
    run_ablation(config_path=config_path)
