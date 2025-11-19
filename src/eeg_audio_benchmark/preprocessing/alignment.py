"""EEG/audio alignment utilities using HF-derived manifests."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT

from .utils import SlidingWindow, ensure_directory, gaussian_smooth, resolve_dataset_path


@dataclass
class AlignmentConfig:
    dataset_root: str = str(LOCAL_DATA_ROOT)
    epoch_manifest: str = "{dataset_root}/derivatives/epoch_manifest.csv"
    artifact_report: str = "{dataset_root}/derivatives/qc/artifacts_report.csv"
    audio_feature_dir: str = "{dataset_root}/derivatives/audio_features"
    output_eeg_dir: str = "{dataset_root}/derivatives/aligned/eeg"
    output_audio_dir: str = "{dataset_root}/derivatives/aligned/audio"
    output_manifest: str = "{dataset_root}/manifest_epochs.csv"
    log_level: str = "INFO"
    eeg_sampling_rate: float = 512.0
    eeg_window_ms: float = 250.0
    eeg_step_ms: float = 50.0
    target_duration: float = 4.0
    target_hop: float = 0.011
    smoothing_sigma: float = 0.0
    missing_policy: str = "warn"


def _load_clean_epochs(artifact_report: Path) -> pd.DataFrame:
    df = pd.read_csv(artifact_report)
    if "Is_Artifact" not in df:
        raise ValueError("Artifact report missing Is_Artifact column")
    return df[df["Is_Artifact"] == False].copy()  # noqa: E712


def _sliding_window_features(epoch: np.ndarray, config: AlignmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    eeg = epoch[:, 2:]
    if config.smoothing_sigma > 0:
        eeg = gaussian_smooth(eeg, config.smoothing_sigma)
    window = int(round(config.eeg_window_ms * config.eeg_sampling_rate / 1000.0))
    step = int(round(config.eeg_step_ms * config.eeg_sampling_rate / 1000.0))
    if window <= 0 or step <= 0:
        raise ValueError("Invalid EEG window configuration")
    sw = SlidingWindow(window, step)
    features: List[np.ndarray] = []
    centers: List[float] = []
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


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_epoch(path: Path) -> np.ndarray:
    epoch = np.load(path)
    if epoch.ndim != 2 or epoch.shape[1] < 3:
        raise ValueError(f"Epoch file {path} has unexpected shape {epoch.shape}")
    return epoch


def _build_output_paths(config: AlignmentConfig, dataset_root: Path) -> Dict[str, Path]:
    eeg_dir = resolve_dataset_path(config.output_eeg_dir, dataset_root)
    audio_dir = resolve_dataset_path(config.output_audio_dir, dataset_root)
    manifest_path = resolve_dataset_path(config.output_manifest, dataset_root)
    if eeg_dir is None or audio_dir is None or manifest_path is None:
        raise ValueError("output directories and manifest path must be provided")
    ensure_directory(eeg_dir)
    ensure_directory(audio_dir)
    ensure_directory(manifest_path.parent)
    return {"eeg_dir": eeg_dir, "audio_dir": audio_dir, "manifest": manifest_path}


def align_eeg_audio_pairs(config: AlignmentConfig) -> List[Tuple[Path, Path]]:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    dataset_root = Path(config.dataset_root)
    epoch_manifest = resolve_dataset_path(config.epoch_manifest, dataset_root)
    artifact_report = resolve_dataset_path(config.artifact_report, dataset_root)
    audio_feature_dir = resolve_dataset_path(config.audio_feature_dir, dataset_root)
    if epoch_manifest is None or artifact_report is None or audio_feature_dir is None:
        raise ValueError("epoch_manifest, artifact_report, and audio_feature_dir are required")

    outputs = _build_output_paths(config, dataset_root)
    target_times = np.arange(0, config.target_duration, config.target_hop)

    clean_df = _load_clean_epochs(artifact_report)
    if clean_df.empty:
        logger.warning("No clean epochs found in artifact report")
        return []

    epoch_df = pd.read_csv(epoch_manifest)
    merged = pd.merge(clean_df, epoch_df, on=["epoch_path", "subject_id", "run_id", "event_index", "stim_id", "audio_file"], how="left")

    aligned_pairs: List[Tuple[Path, Path]] = []
    manifest_rows: List[Dict[str, str | int | float]] = []

    for _, row in merged.iterrows():
        epoch_path = Path(str(row["epoch_path"]))
        if not epoch_path.is_absolute():
            epoch_path = dataset_root / epoch_path
        if not epoch_path.exists():
            msg = f"Epoch file missing: {epoch_path}"
            if config.missing_policy == "error":
                raise FileNotFoundError(msg)
            logger.warning(msg)
            continue

        epoch = _load_epoch(epoch_path)
        eeg_features, eeg_times = _sliding_window_features(epoch, config)
        if eeg_features.size == 0:
            logger.warning("No EEG features for %s", epoch_path.name)
            continue

        audio_base = Path(str(row.get("audio_file"))).stem or row.get("stim_id", "audio")
        audio_features_path = audio_feature_dir / f"{audio_base}_features.npy"
        audio_times_path = audio_feature_dir / f"{audio_base}_feature_times.npy"
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

        base = Path(epoch_path).stem
        eeg_out = outputs["eeg_dir"] / f"{base}_EEG_aligned.npy"
        audio_out = outputs["audio_dir"] / f"{base}_Sound_aligned.npy"
        np.save(eeg_out, eeg_interp.astype(np.float32))
        np.save(audio_out, audio_interp.astype(np.float32))
        aligned_pairs.append((eeg_out, audio_out))

        manifest_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "run_id": row.get("run_id"),
                "event_index": row.get("event_index"),
                "stim_id": row.get("stim_id"),
                "audio_file": row.get("audio_file"),
                "eeg_aligned": _relative_to_root(eeg_out, dataset_root),
                "audio_aligned": _relative_to_root(audio_out, dataset_root),
            }
        )

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(outputs["manifest"], index=False)
        logger.info("Aligned manifest saved to %s", outputs["manifest"])
    logger.info("Aligned %d EEG/audio pairs", len(aligned_pairs))
    return aligned_pairs


__all__ = ["AlignmentConfig", "align_eeg_audio_pairs"]
