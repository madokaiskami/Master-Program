"""Preprocessing utilities for EEG/audio experiments."""

from .eeg_epochs import EEGEpochConfig, slice_eeg_to_epochs
from .artifacts import ArtifactReportConfig, compute_artifact_report
from .audio_features import AudioFeatureConfig, extract_audio_features
from .alignment import AlignmentConfig, align_eeg_audio_pairs
from .utils import load_step_config, parse_config

__all__ = [
    "EEGEpochConfig",
    "ArtifactReportConfig",
    "AudioFeatureConfig",
    "AlignmentConfig",
    "slice_eeg_to_epochs",
    "compute_artifact_report",
    "extract_audio_features",
    "align_eeg_audio_pairs",
    "load_step_config",
    "parse_config",
]
