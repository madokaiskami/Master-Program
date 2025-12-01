"""Spectral analysis utilities for aligned EEG segments.

This module provides helper functions to compute PSD-based features
from aligned EEG/audio segments and to visualize per-subject spectral
properties. It builds on the existing TRF data loader.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import welch

from eeg_audio_benchmark.trf.data import Segment, load_segments_from_hf_manifest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandDef:
    """Definition of a frequency band."""

    name: str
    fmin: float
    fmax: float


DEFAULT_BANDS: list[BandDef] = [
    BandDef("delta", 1.0, 4.0),
    BandDef("theta", 4.0, 8.0),
    BandDef("alpha", 8.0, 13.0),
    BandDef("beta", 13.0, 30.0),
    BandDef("gamma", 30.0, 80.0),
]


def load_all_segments(manifest_path: Path | str = Path("data/hf_eeg_audio/manifest_epochs.csv")) -> list[Segment]:
    """Load all aligned segments from a manifest.

    Parameters
    ----------
    manifest_path:
        Path to the manifest produced by preprocessing. Relative paths
        are resolved from the manifest's parent directory so that EEG
        and audio file references resolve correctly.
    """

    manifest = Path(manifest_path)
    dataset_root = manifest.parent
    logger.info("Loading segments from %s", manifest)
    segments = load_segments_from_hf_manifest(root=dataset_root, manifest_path=manifest)
    logger.info("Loaded %d total segments", len(segments))
    return segments


def compute_psd_for_segment(eeg: np.ndarray, fs: float, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for a single EEG segment.

    Parameters
    ----------
    eeg:
        Array of shape (T, C) with time along axis 0.
    fs:
        Sampling rate in Hz.
    nperseg:
        Window length for the Welch estimate.
    """

    if eeg.ndim != 2:
        raise ValueError(f"EEG must be 2D (time, channels); got shape {eeg.shape}")

    psd_list = []
    freqs: np.ndarray | None = None
    for ch in range(eeg.shape[1]):
        freqs, psd_ch = welch(eeg[:, ch], fs=fs, nperseg=nperseg)
        psd_list.append(psd_ch)
    psd = np.vstack(psd_list)
    return freqs, psd


def _band_mask(freqs: np.ndarray, band: BandDef) -> np.ndarray:
    return (freqs >= band.fmin) & (freqs < band.fmax)


def _aggregate_band_power(psd: np.ndarray, freqs: np.ndarray, band: BandDef) -> np.ndarray:
    mask = _band_mask(freqs, band)
    if not mask.any():
        return np.zeros(psd.shape[0])
    # Integrate power over the band for each channel.
    return np.trapz(psd[:, mask], freqs[mask], axis=1)


def compute_band_power_per_subject(
    segments: Sequence[Segment],
    fs: float,
    bands: Sequence[BandDef] = DEFAULT_BANDS,
    nperseg: int = 512,
) -> pd.DataFrame:
    """Compute average band power per subject and channel.

    Returns
    -------
    pd.DataFrame
        Columns: ``subject_id``, ``channel``, ``band``, ``power``.
    """

    records: list[dict[str, object]] = []
    grouped: dict[str, list[Segment]] = {}
    for seg in segments:
        grouped.setdefault(seg.subject_id, []).append(seg)

    for subject_id, subject_segments in grouped.items():
        logger.info("Computing band power for subject %s (%d segments)", subject_id, len(subject_segments))
        accum: dict[tuple[int, str], list[float]] = {}
        for seg in subject_segments:
            freqs, psd = compute_psd_for_segment(seg.eeg, fs=fs, nperseg=nperseg)
            for band in bands:
                band_power = _aggregate_band_power(psd, freqs, band)
                for channel_idx, value in enumerate(band_power):
                    accum.setdefault((channel_idx, band.name), []).append(float(value))
        for (channel_idx, band_name), values in accum.items():
            if not values:
                continue
            records.append(
                {
                    "subject_id": subject_id,
                    "channel": channel_idx,
                    "band": band_name,
                    "power": float(np.mean(values)),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_example_psd(subject_id: str, segments: Sequence[Segment], fs: float, out_path: Path) -> None:
    """Plot an example PSD for one subject averaged across channels/segments."""

    subject_segments = [seg for seg in segments if seg.subject_id == subject_id]
    if not subject_segments:
        raise ValueError(f"No segments found for subject {subject_id}")

    psd_stack = []
    freqs: np.ndarray | None = None
    for seg in subject_segments:
        freqs, psd = compute_psd_for_segment(seg.eeg, fs=fs)
        psd_stack.append(psd.mean(axis=0))

    mean_psd = np.mean(np.vstack(psd_stack), axis=0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(freqs, mean_psd, label="Mean PSD")
    for band in DEFAULT_BANDS:
        ax.axvspan(band.fmin, band.fmax, alpha=0.15, label=band.name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title(f"PSD for subject {subject_id}")
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate band labels
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title="Bands", loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved PSD plot for %s to %s", subject_id, out_path)


def plot_band_correlation_heatmap(df_band_power: pd.DataFrame, out_path: Path) -> None:
    """Plot a correlation heatmap of band power across subjects."""

    if df_band_power.empty:
        raise ValueError("Band power DataFrame is empty")

    aggregated = (
        df_band_power.groupby(["subject_id", "band"])["power"].mean().reset_index()
    )
    pivot = aggregated.pivot(index="subject_id", columns="band", values="power")
    corr = pivot.corr()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="magma", ax=ax)
    ax.set_title("Correlation of band power across subjects")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved band correlation heatmap to %s", out_path)


__all__ = [
    "BandDef",
    "DEFAULT_BANDS",
    "load_all_segments",
    "compute_band_power_per_subject",
    "compute_psd_for_segment",
    "plot_example_psd",
    "plot_band_correlation_heatmap",
]
