"""Global EEG–sound offset scanning utilities."""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

import numpy as np

from .data import Segment
from .features import (
    broadband_envelope,
    highpass_moving_average,
    voiced_mask_from_sound,
    zscore_eeg,
)
from .roi import channel_envelope_max_correlation

logger = logging.getLogger(__name__)


def shift_sound_forward(S: np.ndarray, frames: int) -> np.ndarray:
    """Shift sound feature matrix forward by ``frames`` (causal shift)."""

    if frames <= 0:
        return S.copy()
    T, D = S.shape
    out = np.zeros_like(S)
    if frames < T:
        out[frames:] = S[: T - frames]
    return out


def _segment_roi_score(
    segment: Segment,
    roi_channels: Sequence[int],
    max_lag_frames: int,
    n_mels: int,
    smooth_win: int,
    eeg_highpass_win: int,
    eeg_zscore_mode: str,
    stats_cache: Dict[Tuple[str, int], Tuple[float, float]],
    shift_frames: int,
    voicing_cols: Sequence[int],
) -> float:
    sound = shift_sound_forward(segment.sound, shift_frames) if shift_frames else segment.sound
    env = broadband_envelope(sound, n_mels=n_mels, smooth_win=smooth_win)[:, 0]
    vmask = voiced_mask_from_sound(segment.sound, voicing_cols)
    T = min(segment.eeg.shape[0], env.shape[0], len(vmask))
    env = env[:T]
    mask_use = vmask[:T]
    eeg = segment.eeg[:T, roi_channels]
    per_channel = []
    for idx in range(eeg.shape[1]):
        channel_idx = roi_channels[idx]
        z = zscore_eeg(
            eeg[:, idx],
            mode=eeg_zscore_mode,
            subject_id=segment.subject_id,
            channel_idx=channel_idx,
            channel_stats_cache=stats_cache,
        )
        y = highpass_moving_average(z, win=eeg_highpass_win)
        per_channel.append(
            channel_envelope_max_correlation(y, env, max_lag_frames=max_lag_frames, mask=mask_use)
        )
    per_channel = [r for r in per_channel if np.isfinite(r)]
    return float(np.median(per_channel)) if per_channel else np.nan


def score_offset_for_roi(
    segments: Sequence[Segment],
    subject_id: str,
    roi_channels: Sequence[int],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
    eeg_highpass_win: int = 15,
    eeg_zscore_mode: str = "per_segment_channel",
    voicing_cols: Sequence[int] | None = None,
) -> Dict[int, float]:
    """Return a mapping from offset (frames) to ROI correlation score."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    scores: Dict[int, float] = {}
    stats_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for off in candidate_offsets_frames:
        per_seg = [
            _segment_roi_score(
                seg,
                roi_channels,
                max_lag_frames,
                n_mels=n_mels,
                smooth_win=smooth_win,
                eeg_highpass_win=eeg_highpass_win,
                eeg_zscore_mode=eeg_zscore_mode,
                stats_cache=stats_cache,
                shift_frames=off,
                voicing_cols=voicing_cols or [],
            )
            for seg in subject_segments
        ]
        scores[off] = float(np.nanmedian(per_seg)) if per_seg else np.nan
    return scores


def pick_best_global_offset(
    segments: Sequence[Segment],
    subject_id: str,
    roi_channels: Sequence[int],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
    eeg_highpass_win: int = 15,
    eeg_zscore_mode: str = "per_segment_channel",
    voicing_cols: Sequence[int] | None = None,
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
        eeg_highpass_win=eeg_highpass_win,
        eeg_zscore_mode=eeg_zscore_mode,
        voicing_cols=voicing_cols,
    )
    if not scores:
        return 0
    valid_scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not valid_scores:
        return 0
    best_score = max(valid_scores.values())
    best_offsets = [off for off, score in valid_scores.items() if score == best_score]
    best_offsets.sort(key=lambda x: (abs(x), x))
    best = best_offsets[0]
    logger.info("Subject %s best offset: %d (score=%.4f)", subject_id, best, best_score)
    return best


def pick_best_offsets_for_subject(
    segments: Sequence[Segment],
    subject_id: str,
    roi_channels: Sequence[int],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
    voicing_cols: Sequence[int] | None = None,
) -> int:
    """Wrapper matching notebook-style offset search for a subject."""

    return pick_best_global_offset(
        segments,
        subject_id=subject_id,
        roi_channels=roi_channels,
        candidate_offsets_frames=candidate_offsets_frames,
        max_lag_frames=max_lag_frames,
        n_mels=n_mels,
        smooth_win=smooth_win,
        voicing_cols=voicing_cols,
    )


def pick_best_offsets_for_all_subjects(
    segments: Sequence[Segment],
    roi_map: Dict[str, Sequence[int]],
    candidate_offsets_frames: Sequence[int],
    max_lag_frames: int,
    n_mels: int = 40,
    smooth_win: int = 9,
    voicing_cols: Sequence[int] | None = None,
) -> Dict[str, int]:
    """Compute best offsets for all subjects."""

    results: Dict[str, int] = {}
    for sid, roi in roi_map.items():
        results[sid] = pick_best_offsets_for_subject(
            segments,
            subject_id=sid,
            roi_channels=roi,
            candidate_offsets_frames=candidate_offsets_frames,
            max_lag_frames=max_lag_frames,
            n_mels=n_mels,
            smooth_win=smooth_win,
            voicing_cols=voicing_cols,
        )
    return results


__all__ = [
    "shift_sound_forward",
    "score_offset_for_roi",
    "pick_best_global_offset",
    "pick_best_offsets_for_subject",
    "pick_best_offsets_for_all_subjects",
]
