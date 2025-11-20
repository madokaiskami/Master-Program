"""Evaluation utilities for TRF envelope models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .data import Segment
from .features import build_lagged_features, envelope_from_sound_matrix
from .models import TRFConfig, TRFEncoder
from .offset import shift_sound_forward

logger = logging.getLogger(__name__)


def _build_design_matrices(
    segments: Iterable[Segment],
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    n_mels: int,
    smooth_win: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    groups: List[np.ndarray] = []
    for idx, seg in enumerate(segments):
        sound = shift_sound_forward(seg.sound, offset_frames) if offset_frames else seg.sound
        env = envelope_from_sound_matrix(sound, n_mels=n_mels, smooth_win=smooth_win)
        X_seg = build_lagged_features(env, n_pre=trf_config.n_pre, n_post=trf_config.n_post)
        eeg = seg.eeg[:, roi_channels] if roi_channels else seg.eeg
        T = min(X_seg.shape[0], eeg.shape[0])
        if T == 0:
            continue
        X_list.append(X_seg[:T])
        Y_list.append(eeg[:T])
        groups.append(np.full(T, idx, dtype=int))
    if not X_list:
        return np.empty((0, trf_config.n_pre + trf_config.n_post + 1)), np.empty((0, 0)), np.empty(0)
    return np.vstack(X_list), np.vstack(Y_list), np.concatenate(groups)


def eval_subject_trf_envelope(
    segments: List[Segment],
    subject_id: str,
    trf_config: TRFConfig,
    n_splits: int = 5,
    max_segments: int | None = None,
    random_state: int = 42,
    roi_channels: Sequence[int] | None = None,
    offset_frames: int = 0,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> Dict[str, Any]:
    """Evaluate an envelope-level TRF model for a single subject."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    if max_segments is not None:
        subject_segments = subject_segments[:max_segments]
    X, Y, groups = _build_design_matrices(
        subject_segments,
        trf_config=trf_config,
        roi_channels=roi_channels,
        offset_frames=offset_frames,
        n_mels=n_mels,
        smooth_win=smooth_win,
    )
    if X.size == 0 or Y.size == 0:
        logger.warning("No data available for subject %s", subject_id)
        return {"subject_id": subject_id, "mean_r2": np.nan, "null_mean_r2": np.nan, "r2_per_channel": []}

    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, n_groups) if n_groups > 1 else 1
    if n_splits < 1:
        n_splits = 1
    splitter = GroupKFold(n_splits=n_splits) if n_splits > 1 else None

    rng = np.random.default_rng(random_state)
    r2_scores: List[np.ndarray] = []
    null_scores: List[np.ndarray] = []

    if splitter:
        for train_idx, test_idx in splitter.split(X, Y, groups):
            model = TRFEncoder(trf_config)
            model.fit(X[train_idx], Y[train_idx])
            preds = model.predict(X[test_idx])
            r2 = r2_score(Y[test_idx], preds, multioutput="raw_values")
            r2_scores.append(r2)

            shuffled = rng.permutation(Y[train_idx])
            null_model = TRFEncoder(trf_config)
            null_model.fit(X[train_idx], shuffled)
            null_preds = null_model.predict(X[test_idx])
            null_r2 = r2_score(Y[test_idx], null_preds, multioutput="raw_values")
            null_scores.append(null_r2)
    else:
        model = TRFEncoder(trf_config)
        model.fit(X, Y)
        preds = model.predict(X)
        r2_scores.append(r2_score(Y, preds, multioutput="raw_values"))
        shuffled = rng.permutation(Y)
        null_model = TRFEncoder(trf_config)
        null_model.fit(X, shuffled)
        null_preds = null_model.predict(X)
        null_scores.append(r2_score(Y, null_preds, multioutput="raw_values"))

    r2_array = np.vstack(r2_scores)
    null_array = np.vstack(null_scores)
    mean_r2 = float(np.nanmean(r2_array))
    null_mean_r2 = float(np.nanmean(null_array))

    return {
        "subject_id": subject_id,
        "mean_r2": mean_r2,
        "null_mean_r2": null_mean_r2,
        "r2_per_channel": r2_array.tolist(),
        "n_splits": n_splits,
        "n_segments": len(subject_segments),
        "offset_frames": offset_frames,
        "roi_channels": list(roi_channels) if roi_channels else None,
    }


def run_trf_analysis_per_subject(
    segments: List[Segment],
    trf_config: TRFConfig,
    n_splits: int = 5,
    roi_map: Mapping[str, Sequence[int]] | None = None,
    offset_map: Mapping[str, int] | None = None,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> pd.DataFrame:
    """Run TRF evaluation for each subject and collect a summary DataFrame."""

    subject_ids = sorted({s.subject_id for s in segments})
    results: List[Dict[str, Any]] = []
    for sid in subject_ids:
        res = eval_subject_trf_envelope(
            segments,
            subject_id=sid,
            trf_config=trf_config,
            n_splits=n_splits,
            roi_channels=roi_map.get(sid) if roi_map else None,
            offset_frames=offset_map.get(sid, 0) if offset_map else 0,
            n_mels=n_mels,
            smooth_win=smooth_win,
        )
        results.append(res)
    return pd.DataFrame(results)


__all__ = ["eval_subject_trf_envelope", "run_trf_analysis_per_subject"]
