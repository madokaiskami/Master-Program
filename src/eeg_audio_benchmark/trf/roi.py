"""ROI channel selection utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .data import Segment
from .features import envelope_from_sound_matrix

logger = logging.getLogger(__name__)


def compute_voiced_mask(
    S: np.ndarray, voicing_column: int | None = None, threshold: float = 0.5
) -> np.ndarray | None:
    """Return a boolean mask (T,) indicating voiced frames if available."""

    if voicing_column is None:
        return None
    if voicing_column < 0 or voicing_column >= S.shape[1]:
        return None
    return S[:, voicing_column] > threshold


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (x - mean) / std


def channel_envelope_max_correlation(
    eeg_channel: np.ndarray, env: np.ndarray, max_lag_frames: int, mask: np.ndarray | None = None
) -> float:
    """Compute maximum absolute correlation between an EEG channel and envelope across lags."""

    if mask is not None:
        eeg_channel = eeg_channel[mask]
        env = env[mask]
    if eeg_channel.size == 0 or env.size == 0:
        return 0.0
    best = 0.0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            eeg_slice = eeg_channel[:lag]
            env_slice = env[-lag:]
        elif lag > 0:
            eeg_slice = eeg_channel[lag:]
            env_slice = env[: -lag]
        else:
            eeg_slice = eeg_channel
            env_slice = env
        if eeg_slice.size < 2 or env_slice.size < 2:
            continue
        if np.std(eeg_slice) == 0 or np.std(env_slice) == 0:
            continue
        corr = float(np.corrcoef(eeg_slice, env_slice)[0, 1])
        best = max(best, abs(corr))
    return best


def select_roi_channels_for_subject(
    segments: List[Segment],
    subject_id: str,
    max_lag_frames: int,
    top_k: int = 3,
    n_mels: int = 40,
    smooth_win: int = 9,
    voicing_column: int | None = None,
) -> List[int]:
    """Select the ROI channels with highest envelope–EEG correlation for a subject."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    if not subject_segments:
        logger.warning("No segments found for subject %s", subject_id)
        return []

    envs = [
        envelope_from_sound_matrix(seg.sound, n_mels=n_mels, smooth_win=smooth_win)
        for seg in subject_segments
    ]
    voiced_masks = [compute_voiced_mask(seg.sound, voicing_column=voicing_column) for seg in subject_segments]

    n_channels = subject_segments[0].eeg.shape[1] if subject_segments[0].eeg.ndim > 1 else 0
    scores: List[float] = []
    for ch in range(n_channels):
        per_seg: List[float] = []
        for seg, env, vmask in zip(subject_segments, envs, voiced_masks):
            if seg.eeg.shape[0] != env.shape[0]:
                T = min(seg.eeg.shape[0], env.shape[0])
                eeg_channel = seg.eeg[:T, ch]
                env_use = env[:T]
                mask = vmask[:T] if vmask is not None else None
            else:
                eeg_channel = seg.eeg[:, ch]
                env_use = env
                mask = vmask
            eeg_channel = _zscore(eeg_channel)
            per_seg.append(
                channel_envelope_max_correlation(
                    eeg_channel=eeg_channel, env=env_use, max_lag_frames=max_lag_frames, mask=mask
                )
            )
        scores.append(float(np.median(per_seg)) if per_seg else 0.0)

    sorted_idx = np.argsort(scores)[::-1]
    top = [int(idx) for idx in sorted_idx[:top_k]]
    logger.info("Subject %s ROI channels (top %d): %s", subject_id, top_k, top)
    return top


__all__ = [
    "channel_envelope_max_correlation",
    "compute_voiced_mask",
    "select_roi_channels_for_subject",
]
