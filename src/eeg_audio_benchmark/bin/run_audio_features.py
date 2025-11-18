"""CLI for audio feature extraction."""

from __future__ import annotations

import argparse

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.preprocessing import AudioFeatureConfig, extract_audio_features, parse_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features for WAV files")
    parser.add_argument("--config", required=True, help="Audio feature YAML config")
    args = parser.parse_args()
    config_dict = load_config(args.config)
    feature_config = parse_config(config_dict, AudioFeatureConfig)
    extract_audio_features(feature_config)


if __name__ == "__main__":
    main()
