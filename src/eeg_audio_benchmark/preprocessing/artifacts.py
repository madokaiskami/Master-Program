"""Artifact evaluation utilities built on top of HF-based epochs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT

from .utils import ensure_directory, gaussian_smooth, resolve_dataset_path


METRIC_NAMES = [
    "std",
    "rms",
    "max_abs",
    "ratio_max_meanabs",
    "ratio_max_std",
    "ratio_maxdev_mad",
]


@dataclass
class ArtifactReportConfig:
    dataset_root: str = str(LOCAL_DATA_ROOT)
    epoch_dir: str = "{dataset_root}/derivatives/epochs"
    epoch_manifest: str = "{dataset_root}/derivatives/epoch_manifest.csv"
    output_csv: str = "{dataset_root}/derivatives/qc/artifacts_report.csv"
    log_level: str = "INFO"
    smoothing_sigma: float = 0.0
    metric_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "std": 1.0,
            "rms": 1.0,
            "max_abs": 1.0,
            "ratio_max_meanabs": 1.0,
            "ratio_max_std": 1.0,
            "ratio_maxdev_mad": 1.0,
        }
    )
    composite_threshold: float = 2.0
    artifact_plots_dir: str | None = None


def _load_epoch(path: Path) -> np.ndarray:
    epoch = np.load(path)
    if epoch.ndim != 2 or epoch.shape[1] < 3:
        raise ValueError(f"Epoch file {path} has unexpected shape {epoch.shape}")
    return epoch


def _compute_metrics(eeg: np.ndarray) -> Dict[str, float]:
    eeg = eeg.astype(np.float32)
    if eeg.size == 0:
        return {name: np.nan for name in METRIC_NAMES}
    std = float(np.std(eeg))
    rms = float(np.sqrt(np.mean(eeg**2)))
    max_abs = float(np.max(np.abs(eeg)))
    mean_abs = float(np.mean(np.abs(eeg))) + 1e-8
    mad = float(np.median(np.abs(eeg - np.median(eeg)))) + 1e-8
    metrics = {
        "std": std,
        "rms": rms,
        "max_abs": max_abs,
        "ratio_max_meanabs": max_abs / mean_abs,
        "ratio_max_std": max_abs / (std + 1e-8),
        "ratio_maxdev_mad": max_abs / mad,
    }
    return metrics


def _prepare_dataframe(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [name for name in METRIC_NAMES if name in df.columns]
    df_z = df.copy()
    for col in metric_cols:
        df_z[f"z_{col}"] = (df[col] - df[col].mean()) / (df[col].std(ddof=0) + 1e-8)
    return df_z


def _composite_scores(df: pd.DataFrame, weights: Dict[str, float]) -> np.ndarray:
    comps = np.zeros(len(df), dtype=np.float32)
    for name, weight in weights.items():
        col = f"z_{name}" if f"z_{name}" in df.columns else name
        if col in df:
            comps += weight * df[col].to_numpy(dtype=np.float32)
    return comps


def _resolve_epoch_path(row: pd.Series, epoch_dir: Path, dataset_root: Path) -> Path:
    epoch_path = Path(str(row["epoch_path"]))
    if not epoch_path.is_absolute():
        if (dataset_root / epoch_path).exists():
            return dataset_root / epoch_path
        return epoch_dir / epoch_path.name
    return epoch_path


def compute_artifact_report(config: ArtifactReportConfig) -> Path:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    dataset_root = Path(config.dataset_root)
    epoch_dir = resolve_dataset_path(config.epoch_dir, dataset_root)
    manifest_path = resolve_dataset_path(config.epoch_manifest, dataset_root)
    if epoch_dir is None or manifest_path is None:
        raise ValueError("epoch_dir and epoch_manifest must be provided")

    manifest_df = pd.read_csv(manifest_path)
    if manifest_df.empty:
        raise ValueError("Epoch manifest is empty; run eeg_epochs first")

    rows: List[Dict[str, float]] = []
    cached_epochs: List[np.ndarray] = []

    for _, row in manifest_df.iterrows():
        epoch_path = _resolve_epoch_path(row, epoch_dir, dataset_root)
        if not epoch_path.exists():
            raise FileNotFoundError(epoch_path)

        epoch = _load_epoch(epoch_path)
        cached_epochs.append(epoch)
        eeg = epoch[:, 2:]
        if config.smoothing_sigma > 0:
            eeg = gaussian_smooth(eeg, config.smoothing_sigma)
        metrics = _compute_metrics(eeg)
        artifact_row: Dict[str, float] = {
            "epoch_path": row["epoch_path"],
            "epoch_filename": epoch_path.name,
            "subject_id": row.get("subject_id"),
            "run_id": row.get("run_id"),
            "event_index": row.get("event_index"),
            "stim_id": row.get("stim_id"),
            "audio_file": row.get("audio_file"),
        }
        artifact_row.update(metrics)
        rows.append(artifact_row)

    df = _prepare_dataframe(rows)
    df["Composite_Score"] = _composite_scores(df, config.metric_weights)
    df["Is_Artifact"] = df["Composite_Score"] >= config.composite_threshold

    if config.artifact_plots_dir:
        plot_dir = resolve_dataset_path(config.artifact_plots_dir, dataset_root)
        if plot_dir:
            ensure_directory(plot_dir)
            for idx, (epoch_arr, is_artifact, row) in enumerate(
                zip(cached_epochs, df["Is_Artifact"], rows)
            ):
                if not bool(is_artifact):
                    continue
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(epoch_arr[:, 0], epoch_arr[:, 2:])
                ax.set_title(
                    f"Artifact epoch {row['epoch_filename']} score={df.loc[idx, 'Composite_Score']:.2f}"
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                fig.tight_layout()
                fig.savefig(plot_dir / f"artifact_{idx:04d}.png")
                plt.close(fig)

    output_path = resolve_dataset_path(config.output_csv, dataset_root)
    if output_path is None:
        raise ValueError("output_csv must be provided")
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    logger.info("Artifact report saved to %s", output_path)
    return output_path


__all__ = ["ArtifactReportConfig", "compute_artifact_report"]
