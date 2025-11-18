"""CLI entry-point to run benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..experiment import run_from_config
from ..hf_data import prepare_data_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEG-audio experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--preproc-config",
        type=Path,
        help="Optional preprocessing config for HF dataset preparation.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading the HF dataset snapshot.",
    )
    parser.add_argument(
        "--force-preproc",
        action="store_true",
        help="Force re-running preprocessing even if derivatives exist.",
    )
    parser.add_argument(
        "--skip-hf-sync",
        action="store_true",
        help=(
            "Skip HuggingFace dataset sync (useful for fully custom local datasets)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = None
    if not args.skip_hf_sync:
        data_root = prepare_data_for_training(
            preproc_config_path=args.preproc_config,
            force_preproc=args.force_preproc,
            force_download=args.force_download,
        )
    results = run_from_config(args.config, data_root=data_root)
    for fold in results:
        print(
            f"Fold {fold['fold']}: R^2={fold['r2']:.3f}, Pearson={fold['pearson']:.3f}"
        )


if __name__ == "__main__":
    main()
