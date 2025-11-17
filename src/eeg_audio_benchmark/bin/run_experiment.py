"""CLI entry-point to run benchmark experiments."""

from __future__ import annotations

import argparse

from ..experiment import run_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG-audio experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_from_config(args.config)
    for fold in results:
        print(
            f"Fold {fold['fold']}: R^2={fold['r2']:.3f}, Pearson={fold['pearson']:.3f}"
        )


if __name__ == "__main__":
    main()
