"""CLI entry point for ADT-style EEG→audio Transformer experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from eeg_audio_benchmark.adt.config import load_adt_experiment_config_from_yaml
from eeg_audio_benchmark.adt.train import run_adt_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ADT Transformer EEG→audio experiment")
    parser.add_argument("--config", type=Path, required=True, help="YAML config for the ADT experiment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_adt_experiment_config_from_yaml(args.config)
    results = run_adt_experiment(config)
    if not results.empty:
        median_r2 = results["mean_r2"].median()
        median_r = results["median_pred_r"].median()
        print(f"Median subject mean R^2: {median_r2:.4f}")
        print(f"Median subject median_pred_r: {median_r:.4f}")
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
