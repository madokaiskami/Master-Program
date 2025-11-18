"""CLI for artifact report computation."""

from __future__ import annotations

import argparse

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.preprocessing import (
    ArtifactReportConfig,
    compute_artifact_report,
    parse_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute artifact scores for EEG epochs")
    parser.add_argument("--config", required=True, help="Artifact YAML config")
    args = parser.parse_args()
    config_dict = load_config(args.config)
    artifact_config = parse_config(config_dict, ArtifactReportConfig)
    compute_artifact_report(artifact_config)


if __name__ == "__main__":
    main()
