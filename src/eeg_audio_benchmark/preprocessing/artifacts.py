"""Artifact evaluation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .utils import ensure_directory, gaussian_smooth


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
    epoch_dir: str
    output_csv: str
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


def compute_artifact_report(config: ArtifactReportConfig) -> Path:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    epoch_dir = Path(config.epoch_dir)
    if not epoch_dir.exists():
        raise FileNotFoundError(epoch_dir)
    rows: List[Dict[str, float]] = []
    cached_epochs: List[np.ndarray] = []
    epoch_paths = sorted(epoch_dir.glob("*.npy"))
    for epoch_path in epoch_paths:
        epoch = _load_epoch(epoch_path)
        cached_epochs.append(epoch)
        eeg = epoch[:, 2:]
        if config.smoothing_sigma > 0:
            eeg = gaussian_smooth(eeg, config.smoothing_sigma)
        metrics = _compute_metrics(eeg)
        row = {
            "Epoch_Filename": epoch_path.name,
            "Subject_ID": epoch_path.stem.split("_")[0],
            "Sequence_Number": epoch_path.stem.split("_")[1] if "_" in epoch_path.stem else "0",
            "WAV_Filename_Base": epoch_path.stem.split("_")[-1],
        }
        row.update(metrics)
        rows.append(row)
    if not rows:
        raise ValueError("No epoch files found")
    df = _prepare_dataframe(rows)
    df["Composite_Score"] = _composite_scores(df, config.metric_weights)
    df["Is_Artifact"] = df["Composite_Score"] >= config.composite_threshold
    if config.artifact_plots_dir:
        plot_dir = Path(config.artifact_plots_dir)
        ensure_directory(plot_dir)
        for idx, (epoch_arr, is_artifact, row) in enumerate(
            zip(cached_epochs, df["Is_Artifact"], rows)
        ):
            if not bool(is_artifact):
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epoch_arr[:, 0], epoch_arr[:, 2:])
            ax.set_title(
                f"Artifact epoch {row['Epoch_Filename']} score={df.loc[idx, 'Composite_Score']:.2f}"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            fig.tight_layout()
            fig.savefig(plot_dir / f"artifact_{idx:04d}.png")
            plt.close(fig)
    output_path = Path(config.output_csv)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False)
    logger.info("Artifact report saved to %s", output_path)
    return output_path


__all__ = ["ArtifactReportConfig", "compute_artifact_report"]
