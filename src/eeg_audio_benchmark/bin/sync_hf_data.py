"""CLI utility to synchronize HF dataset and preprocessing derivatives."""

from __future__ import annotations

import argparse
from pathlib import Path

from eeg_audio_benchmark.hf_data import prepare_data_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the HF EEG/audio dataset and derivatives"
    )
    parser.add_argument(
        "--preproc-config",
        type=Path,
        help="Optional preprocessing YAML config to run if derivatives are missing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-downloading the dataset snapshot from HuggingFace.",
    )
    parser.add_argument(
        "--force-preproc",
        action="store_true",
        help="Force re-running the preprocessing pipeline even if derivatives exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = prepare_data_for_training(
        preproc_config_path=args.preproc_config,
        force_preproc=args.force_preproc,
        force_download=args.force_download,
    )
    print(root)


if __name__ == "__main__":
    main()
