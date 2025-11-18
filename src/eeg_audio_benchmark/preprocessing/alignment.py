"""EEG/audio alignment utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import SlidingWindow, ensure_directory, gaussian_smooth


@dataclass
class AlignmentConfig:
    eeg_epoch_dir: str
    artifact_report: str
    audio_feature_dir: str
    output_dir: str
    log_level: str = "INFO"
    eeg_sampling_rate: float = 512.0
    eeg_window_ms: float = 250.0
    eeg_step_ms: float = 50.0
    target_duration: float = 4.0
    target_hop: float = 0.011
    smoothing_sigma: float = 0.0
    missing_policy: str = "warn"


def _load_clean_epochs(config: AlignmentConfig) -> pd.DataFrame:
    df = pd.read_csv(config.artifact_report)
    if "Is_Artifact" not in df:
        raise ValueError("Artifact report missing Is_Artifact column")
    return df[df["Is_Artifact"] == False]  # noqa: E712


def _sliding_window_features(epoch: np.ndarray, config: AlignmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    eeg = epoch[:, 2:]
    if config.smoothing_sigma > 0:
        eeg = gaussian_smooth(eeg, config.smoothing_sigma)
    window = int(round(config.eeg_window_ms * config.eeg_sampling_rate / 1000.0))
    step = int(round(config.eeg_step_ms * config.eeg_sampling_rate / 1000.0))
    if window <= 0 or step <= 0:
        raise ValueError("Invalid EEG window configuration")
    sw = SlidingWindow(window, step)
    features = []
    centers = []
    for start, end in sw.generate(eeg.shape[0]):
        segment = eeg[start:end]
        features.append(segment.mean(axis=0))
        centers.append(epoch[start:end, 0].mean())
    if not features:
        return np.empty((0, eeg.shape[1])), np.empty(0)
    return np.vstack(features), np.array(centers)


def _interpolate(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    if len(times) == 0 or len(values) == 0:
        return np.zeros((target_times.size, values.shape[1] if values.ndim > 1 else 1), dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    interp = [
        np.interp(target_times, times, values[:, idx], left=np.nan, right=np.nan)
        for idx in range(values.shape[1])
    ]
    return np.nan_to_num(np.stack(interp, axis=1), nan=0.0)


def align_eeg_audio_pairs(config: AlignmentConfig) -> List[Tuple[Path, Path]]:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    ensure_directory(Path(config.output_dir))
    clean_df = _load_clean_epochs(config)
    target_times = np.arange(0, config.target_duration, config.target_hop)
    outputs: List[Tuple[Path, Path]] = []
    for _, row in clean_df.iterrows():
        epoch_name = row["Epoch_Filename"]
        epoch_path = Path(config.eeg_epoch_dir) / epoch_name
        if not epoch_path.exists():
            msg = f"Epoch file missing: {epoch_path}"
            if config.missing_policy == "error":
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        epoch = np.load(epoch_path)
        eeg_features, eeg_times = _sliding_window_features(epoch, config)
        if eeg_features.size == 0:
            logger.warning("No EEG features for %s", epoch_name)
            continue
        audio_base = row["WAV_Filename_Base"]
        audio_features_path = Path(config.audio_feature_dir) / f"{audio_base}_features.npy"
        audio_times_path = Path(config.audio_feature_dir) / f"{audio_base}_feature_times.npy"
        if not audio_features_path.exists() or not audio_times_path.exists():
            msg = f"Missing audio features for {audio_base}"
            if config.missing_policy == "error":
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue
        audio_features = np.load(audio_features_path)
        audio_times = np.load(audio_times_path)
        eeg_interp = _interpolate(eeg_times, eeg_features, target_times)
        audio_interp = _interpolate(audio_times, audio_features, target_times)
        base = Path(epoch_name).stem
        eeg_out = Path(config.output_dir) / f"{base}_EEG_aligned.npy"
        audio_out = Path(config.output_dir) / f"{base}_Sound_aligned.npy"
        np.save(eeg_out, eeg_interp.astype(np.float32))
        np.save(audio_out, audio_interp.astype(np.float32))
        outputs.append((eeg_out, audio_out))
    logger.info("Aligned %d EEG/audio pairs", len(outputs))
    return outputs


__all__ = ["AlignmentConfig", "align_eeg_audio_pairs"]
