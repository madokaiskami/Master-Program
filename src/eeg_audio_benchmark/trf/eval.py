"""Evaluation utilities for TRF envelope models."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from .data import Segment
from .features import (
    build_lagged_features,
    envelope_from_mel,
    preprocess_eeg_channel,
    voiced_mask_from_sound,
)
from .models import TRFConfig, TRFEncoder
from .offset import shift_sound_forward

logger = logging.getLogger(__name__)


def _prepare_segment_design(
    segment: Segment,
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    n_mels: int,
    smooth_win: int,
    voicing_cols: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build design matrix and target for a single segment."""

    sound = shift_sound_forward(segment.sound, offset_frames) if offset_frames else segment.sound
    env = envelope_from_mel(sound, n_mels=n_mels, smooth_win=smooth_win)
    vmask = voiced_mask_from_sound(segment.sound, voicing_cols)
    eeg = segment.eeg[:, roi_channels] if roi_channels else segment.eeg
    T = min(len(env), eeg.shape[0], len(vmask))
    env = env[:T]
    eeg = eeg[:T]
    vmask = vmask[:T]
    X = build_lagged_features(env, n_pre=trf_config.n_pre, n_post=trf_config.n_post, voicing=vmask)
    ys = [preprocess_eeg_channel(eeg[:, ch]) for ch in range(eeg.shape[1])]
    y = np.mean(np.vstack(ys), axis=0) if ys else np.zeros(T)
    T_use = min(X.shape[0], len(y))
    return X[:T_use], y[:T_use]


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
    voicing_cols: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Evaluate an envelope-level TRF model for a single subject."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    if max_segments is not None:
        subject_segments = subject_segments[:max_segments]
    if not subject_segments:
        return {"subject_id": subject_id, "note": "no segments"}

    voicing_cols = voicing_cols or []
    per_segment_Xy = [
        _prepare_segment_design(
            seg,
            trf_config=trf_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            n_mels=n_mels,
            smooth_win=smooth_win,
            voicing_cols=voicing_cols,
        )
        for seg in subject_segments
    ]
    per_segment_Xy = [(X, y) for X, y in per_segment_Xy if X.size and y.size]
    if not per_segment_Xy:
        logger.warning("No data available for subject %s", subject_id)
        return {"subject_id": subject_id, "mean_r2": np.nan, "null_mean_r2": np.nan, "r2_per_channel": []}

    groups = np.arange(len(per_segment_Xy))
    n_groups = len(groups)
    n_splits_use = min(n_splits, n_groups) if n_groups > 1 else 1
    splitter = GroupKFold(n_splits=n_splits_use) if n_splits_use > 1 else None

    rng = np.random.default_rng(random_state)
    seg_corrs: List[float] = []
    seg_corrs_null: List[float] = []
    r2_scores: List[float] = []
    null_r2_scores: List[float] = []

    def _concat(idx_list: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        Xs, ys = zip(*[per_segment_Xy[i] for i in idx_list])
        Xcat = np.vstack(Xs)
        ycat = np.concatenate(ys)
        return Xcat, ycat

    if splitter:
        for tr_idx, va_idx in splitter.split(np.zeros(len(groups)), groups=groups):
            Xtr, ytr = _concat(tr_idx)
            mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
            sd[sd == 0] = 1.0
            Xtr_std = (Xtr - mu) / sd
            y_mu, y_sd = ytr.mean(), ytr.std()
            ytr_std = (ytr - y_mu) / (y_sd if y_sd > 0 else 1.0)

            model = TRFEncoder(trf_config)
            model.fit(Xtr_std, ytr_std.reshape(-1, 1))

            for idx in va_idx:
                Xv, yv = per_segment_Xy[idx]
                Xv_std = (Xv - mu) / sd
                yv_std = (yv - y_mu) / (y_sd if y_sd > 0 else 1.0)
                pred = model.predict(Xv_std).reshape(-1)
                if np.std(yv_std) > 0 and np.std(pred) > 0:
                    seg_corrs.append(float(np.corrcoef(yv_std, pred)[0, 1]))
                shift = max(1, int(0.33 * len(yv_std)))
                yperm = np.roll(yv_std, shift)
                if np.std(yperm) > 0 and np.std(pred) > 0:
                    seg_corrs_null.append(float(np.corrcoef(yperm, pred)[0, 1]))

            preds_tr = model.predict(Xtr_std).reshape(-1)
            r2_scores.append(float(r2_score(ytr_std, preds_tr)))
            ytr_perm = rng.permutation(ytr_std)
            null_model = TRFEncoder(trf_config)
            null_model.fit(Xtr_std, ytr_perm.reshape(-1, 1))
            null_preds = null_model.predict(Xtr_std).reshape(-1)
            null_r2_scores.append(float(r2_score(ytr_std, null_preds)))
    else:
        Xall, yall = _concat(range(len(per_segment_Xy)))
        mu, sd = Xall.mean(axis=0), Xall.std(axis=0)
        sd[sd == 0] = 1.0
        Xall_std = (Xall - mu) / sd
        y_mu, y_sd = yall.mean(), yall.std()
        yall_std = (yall - y_mu) / (y_sd if y_sd > 0 else 1.0)
        model = TRFEncoder(trf_config)
        model.fit(Xall_std, yall_std.reshape(-1, 1))
        preds = model.predict(Xall_std).reshape(-1)
        if np.std(yall_std) > 0 and np.std(preds) > 0:
            seg_corrs.append(float(np.corrcoef(yall_std, preds)[0, 1]))
        shift = max(1, int(0.33 * len(yall_std)))
        yperm = np.roll(yall_std, shift)
        if np.std(yperm) > 0 and np.std(preds) > 0:
            seg_corrs_null.append(float(np.corrcoef(yperm, preds)[0, 1]))
        r2_scores.append(float(r2_score(yall_std, preds)))
        yperm_glob = rng.permutation(yall_std)
        null_model = TRFEncoder(trf_config)
        null_model.fit(Xall_std, yperm_glob.reshape(-1, 1))
        null_preds = null_model.predict(Xall_std).reshape(-1)
        null_r2_scores.append(float(r2_score(yall_std, null_preds)))

    median_pred_r = float(np.nanmedian(seg_corrs)) if len(seg_corrs) else np.nan
    median_pred_r_null = float(np.nanmedian(seg_corrs_null)) if len(seg_corrs_null) else np.nan
    mean_r2 = float(np.nanmean(r2_scores)) if len(r2_scores) else np.nan
    null_mean_r2 = float(np.nanmean(null_r2_scores)) if len(null_r2_scores) else np.nan

    return {
        "subject_id": subject_id,
        "mean_r2": mean_r2,
        "null_mean_r2": null_mean_r2,
        "median_pred_r": median_pred_r,
        "median_pred_r_null": median_pred_r_null,
        "n_splits": n_splits_use,
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
    voicing_cols: Sequence[int] | None = None,
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
            voicing_cols=voicing_cols,
        )
        results.append(res)
    return pd.DataFrame(results)


__all__ = ["eval_subject_trf_envelope", "run_trf_analysis_per_subject"]
