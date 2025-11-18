"""Run the entire EEG/audio preprocessing pipeline."""

from __future__ import annotations

import argparse
from typing import Callable, Dict, Tuple, Type

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.preprocessing import (
    AlignmentConfig,
    ArtifactReportConfig,
    AudioFeatureConfig,
    EEGEpochConfig,
    align_eeg_audio_pairs,
    compute_artifact_report,
    extract_audio_features,
    parse_config,
    slice_eeg_to_epochs,
)


Step = Tuple[str, Type, Callable[[object], object]]

STEPS: Tuple[Step, ...] = (
    ("eeg_epochs", EEGEpochConfig, slice_eeg_to_epochs),
    ("artifacts", ArtifactReportConfig, compute_artifact_report),
    ("audio_features", AudioFeatureConfig, extract_audio_features),
    ("alignment", AlignmentConfig, align_eeg_audio_pairs),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEG/audio preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()
    config_dict = load_config(args.config)
    for key, cls, fn in STEPS:
        section = config_dict.get(key)
        if not section:
            continue
        if not section.get("enabled", True):
            continue
        step_config = parse_config(section, cls)
        fn(step_config)


if __name__ == "__main__":
    main()
