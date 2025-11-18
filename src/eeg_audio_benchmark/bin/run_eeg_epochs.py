"""CLI for EEG epoch slicing."""

from __future__ import annotations

import argparse

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.preprocessing import EEGEpochConfig, parse_config, slice_eeg_to_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice raw EEG into epochs")
    parser.add_argument("--config", required=True, help="Path to EEG epoch YAML config")
    args = parser.parse_args()
    raw_config = load_config(args.config)
    epoch_config = parse_config(raw_config, EEGEpochConfig)
    slice_eeg_to_epochs(epoch_config)


if __name__ == "__main__":
    main()
