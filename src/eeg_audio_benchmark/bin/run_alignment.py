"""CLI for EEG/audio alignment."""

from __future__ import annotations

import argparse

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.preprocessing import AlignmentConfig, align_eeg_audio_pairs, parse_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Align EEG epochs with audio features")
    parser.add_argument("--config", required=True, help="Alignment YAML config")
    args = parser.parse_args()
    config_dict = load_config(args.config)
    alignment_config = parse_config(config_dict, AlignmentConfig)
    align_eeg_audio_pairs(alignment_config)


if __name__ == "__main__":
    main()
