"""ROI channel selection utilities."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .data import Segment
from .features import envelope_from_mel, preprocess_eeg_channel, voiced_mask_from_sound

logger = logging.getLogger(__name__)


def channel_envelope_max_correlation(
    eeg_channel: np.ndarray, env: np.ndarray, max_lag_frames: int, mask: np.ndarray | None = None
) -> float:
    """Compute maximum correlation over lags, mirroring notebook ROI scoring."""

    if mask is not None:
        eeg_channel = eeg_channel[mask]
        env = env[mask]
    if eeg_channel.size == 0 or env.size == 0:
        return np.nan
    best = -np.inf
    T = min(len(eeg_channel), len(env))
    eeg_channel = eeg_channel[:T]
    env = env[:T]
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            a, b = eeg_channel[lag:], env[: T - lag]
        else:
            a, b = eeg_channel[: T + lag], env[-lag:]
        if len(a) < 20 or len(b) < 20:
            continue
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if np.isfinite(r) and r > best:
            best = r
    return best


def select_roi_channels_for_subject(
    segments: List[Segment],
    subject_id: str,
    max_lag_frames: int,
    top_k: int = 3,
    n_mels: int = 40,
    smooth_win: int = 9,
    voicing_cols: Sequence[int] | None = None,
) -> List[int]:
    """Select the ROI channels with highest envelope–EEG correlation for a subject."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    if not subject_segments:
        logger.warning("No segments found for subject %s", subject_id)
        return []

    envs = [envelope_from_mel(seg.sound, n_mels=n_mels, smooth_win=smooth_win) for seg in subject_segments]
    vcols = voicing_cols or []
    voiced_masks = [voiced_mask_from_sound(seg.sound, vcols) for seg in subject_segments]

    n_channels = subject_segments[0].eeg.shape[1] if subject_segments[0].eeg.ndim > 1 else 0
    per_channel_scores: List[List[float]] = [[] for _ in range(n_channels)]
    for seg, env, vmask in zip(subject_segments, envs, voiced_masks):
        T = min(seg.eeg.shape[0], env.shape[0], len(vmask))
        if T < 30:
            continue
        env_use = env[:T]
        mask_use = vmask[:T]
        for ch in range(n_channels):
            y = preprocess_eeg_channel(seg.eeg[:T, ch])
            r = channel_envelope_max_correlation(y, env_use, max_lag_frames=max_lag_frames, mask=mask_use)
            if np.isfinite(r):
                per_channel_scores[ch].append(float(r))

    med_scores = np.array([np.median(rs) if len(rs) else -np.inf for rs in per_channel_scores])
    sorted_idx = np.argsort(-med_scores)
    top = [int(idx) for idx in sorted_idx[:top_k]]
    logger.info("Subject %s ROI channels (top %d): %s", subject_id, top_k, top)
    return top


__all__ = [
    "channel_envelope_max_correlation",
    "select_roi_channels_for_subject",
]
