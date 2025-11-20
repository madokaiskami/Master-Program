"""Global EEG–sound offset scanning utilities."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np

from .data import Segment
from .features import envelope_from_sound_matrix
from .roi import channel_envelope_max_correlation

logger = logging.getLogger(__name__)


def shift_sound_forward(S: np.ndarray, frames: int) -> np.ndarray:
    """Shift sound feature matrix forward by ``frames`` (causal shift)."""

    if frames == 0:
        return S
    if frames > 0:
        padding = np.zeros((frames, S.shape[1]), dtype=S.dtype)
        return np.concatenate([padding, S[:-frames]], axis=0)
    frames = abs(frames)
    padding = np.zeros((frames, S.shape[1]), dtype=S.dtype)
    return np.concatenate([S[frames:], padding], axis=0)


def _segment_roi_score(
    segment: Segment,
    roi_channels: Sequence[int],
    max_lag_frames: int,
    n_mels: int,
    smooth_win: int,
    shift_frames: int,
) -> float:
    sound = shift_sound_forward(segment.sound, shift_frames) if shift_frames else segment.sound
    env = envelope_from_sound_matrix(sound, n_mels=n_mels, smooth_win=smooth_win)
    eeg = segment.eeg[:, roi_channels]
    if eeg.shape[0] != env.shape[0]:
        T = min(eeg.shape[0], env.shape[0])
        eeg = eeg[:T]
        env = env[:T]
    eeg = (eeg - eeg.mean(axis=0, keepdims=True)) / (eeg.std(axis=0, keepdims=True) + 1e-9)
    per_channel = [
        channel_envelope_max_correlation(eeg[:, idx], env, max_lag_frames=max_lag_frames, mask=None)
        for idx in range(eeg.shape[1])
    ]
    return float(np.median(per_channel)) if per_channel else 0.0


def score_offset_for_roi(
    segments: Sequence[Segment],
    subject_id: str,
    roi_channels: Sequence[int],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> Dict[int, float]:
    """Return a mapping from offset (frames) to ROI correlation score."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    scores: Dict[int, float] = {}
    for off in candidate_offsets_frames:
        per_seg = [
            _segment_roi_score(seg, roi_channels, max_lag_frames, n_mels=n_mels, smooth_win=smooth_win, shift_frames=off)
            for seg in subject_segments
        ]
        scores[off] = float(np.median(per_seg)) if per_seg else 0.0
    return scores


def pick_best_global_offset(
    segments: Sequence[Segment],
    subject_id: str,
    roi_channels: Sequence[int],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> int:
    """Pick the best global offset; ties broken by smaller absolute offset."""

    scores = score_offset_for_roi(
        segments,
        subject_id,
        roi_channels,
        candidate_offsets_frames,
        max_lag_frames,
        n_mels=n_mels,
        smooth_win=smooth_win,
    )
    if not scores:
        return 0
    best_score = max(scores.values())
    best_offsets = [off for off, score in scores.items() if score == best_score]
    best_offsets.sort(key=lambda x: (abs(x), x))
    best = best_offsets[0]
    logger.info("Subject %s best offset: %d (score=%.4f)", subject_id, best, best_score)
    return best


__all__ = ["shift_sound_forward", "score_offset_for_roi", "pick_best_global_offset"]
