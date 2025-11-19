"""Run the entire HF-first EEG/audio preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT
from eeg_audio_benchmark.preprocessing import (
    AlignmentConfig,
    ArtifactReportConfig,
    AudioFeatureConfig,
    EEGEpochConfig,
    align_eeg_audio_pairs,
    compute_artifact_report,
    extract_audio_features,
    slice_eeg_to_epochs,
)


def _resolve_dataset_root(config_dict: Dict[str, Any]) -> Path:
    root = config_dict.get("dataset_root", str(LOCAL_DATA_ROOT))
    return Path(root)


def _build_eeg_config(config_dict: Dict[str, Any], dataset_root: Path) -> EEGEpochConfig | None:
    section = config_dict.get("eeg_epochs", {})
    if not section.get("enabled", True):
        return None
    inputs = section.get("input", {})
    outputs = section.get("output", {})
    params = section.get("params", {})
    return EEGEpochConfig(
        dataset_root=str(dataset_root),
        manifest_raw_runs=inputs.get("manifest_raw_runs", f"{dataset_root}/manifest_raw_runs.csv"),
        output_dir=outputs.get("epoch_dir", f"{dataset_root}/derivatives/epochs"),
        epoch_manifest=outputs.get("epoch_manifest", f"{dataset_root}/derivatives/epoch_manifest.csv"),
        **params,
    )


def _build_artifact_config(config_dict: Dict[str, Any], dataset_root: Path) -> ArtifactReportConfig | None:
    section = config_dict.get("artifacts", {})
    if not section.get("enabled", True):
        return None
    inputs = section.get("input", {})
    outputs = section.get("output", {})
    params = section.get("params", {})
    return ArtifactReportConfig(
        dataset_root=str(dataset_root),
        epoch_dir=inputs.get("epoch_dir", f"{dataset_root}/derivatives/epochs"),
        epoch_manifest=inputs.get("epoch_manifest", f"{dataset_root}/derivatives/epoch_manifest.csv"),
        output_csv=outputs.get("report_csv", f"{dataset_root}/derivatives/qc/artifacts_report.csv"),
        artifact_plots_dir=outputs.get("artifact_plots_dir"),
        **params,
    )


def _build_audio_feature_config(config_dict: Dict[str, Any], dataset_root: Path) -> AudioFeatureConfig | None:
    section = config_dict.get("audio_features", {})
    if not section.get("enabled", True):
        return None
    inputs = section.get("input", {})
    outputs = section.get("output", {})
    params = section.get("params", {})
    return AudioFeatureConfig(
        dataset_root=str(dataset_root),
        epoch_manifest=inputs.get("epoch_manifest", f"{dataset_root}/derivatives/epoch_manifest.csv"),
        wav_dir=inputs.get("wav_dir", f"{dataset_root}/raw/audio/stimuli"),
        output_dir=outputs.get("feature_dir", f"{dataset_root}/derivatives/audio_features"),
        feature_manifest=outputs.get(
            "feature_manifest", f"{dataset_root}/derivatives/audio_features/manifest_audio_features.csv"
        ),
        **params,
    )


def _build_alignment_config(config_dict: Dict[str, Any], dataset_root: Path) -> AlignmentConfig | None:
    section = config_dict.get("alignment", {})
    if not section.get("enabled", True):
        return None
    inputs = section.get("input", {})
    outputs = section.get("output", {})
    params = section.get("params", {})
    return AlignmentConfig(
        dataset_root=str(dataset_root),
        epoch_manifest=inputs.get("epoch_manifest", f"{dataset_root}/derivatives/epoch_manifest.csv"),
        artifact_report=inputs.get("artifact_report", f"{dataset_root}/derivatives/qc/artifacts_report.csv"),
        audio_feature_dir=inputs.get("audio_feature_dir", f"{dataset_root}/derivatives/audio_features"),
        output_eeg_dir=outputs.get("aligned_eeg_dir", f"{dataset_root}/derivatives/aligned/eeg"),
        output_audio_dir=outputs.get("aligned_audio_dir", f"{dataset_root}/derivatives/aligned/audio"),
        output_manifest=outputs.get("manifest_epochs", f"{dataset_root}/manifest_epochs.csv"),
        **params,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EEG/audio preprocessing pipeline on HF data")
    parser.add_argument("--config", required=True, help="Path to pipeline YAML config")
    args = parser.parse_args()

    config_dict = load_config(args.config)
    dataset_root = _resolve_dataset_root(config_dict)

    eeg_config = _build_eeg_config(config_dict, dataset_root)
    if eeg_config:
        slice_eeg_to_epochs(eeg_config)

    artifact_config = _build_artifact_config(config_dict, dataset_root)
    if artifact_config:
        compute_artifact_report(artifact_config)

    audio_config = _build_audio_feature_config(config_dict, dataset_root)
    if audio_config:
        extract_audio_features(audio_config)

    alignment_config = _build_alignment_config(config_dict, dataset_root)
    if alignment_config:
        align_eeg_audio_pairs(alignment_config)


if __name__ == "__main__":
    main()
