"""Streaming evaluation utilities for TRF envelope models."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from .data import Segment
from .evaluation import evaluate_predictions
from .features import (
    broadband_envelope,
    build_lagged_features_lazy,
    energy_feature,
    mel_features_from_sound_matrix,
    slow_envelope,
    voiced_mask_from_sound,
    zscore_eeg,
    highpass_moving_average,
)
from .models import TRFConfig
from .offset import shift_sound_forward

logger = logging.getLogger(__name__)


class RunningStats:
    """Online mean/std accumulator using Welford's algorithm."""

    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: np.ndarray) -> None:
        flat = np.asarray(x, dtype=np.float64).ravel()
        for value in flat:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.M2 += delta * delta2

    def finalize(self) -> Tuple[float, float]:
        if self.count == 0:
            return 0.0, 1.0
        var = self.M2 / max(self.count - 1, 1)
        sd = np.sqrt(var)
        return self.mean, float(sd if sd > 0 else 1.0)


class FeatureReducer:
    """Optional streaming dimensionality reduction before lagging."""

    def __init__(self, method: str = "none", out_dim: int | None = None, random_state: int | None = None):
        self.method = (method or "none").lower()
        self.out_dim = out_dim
        self.random_state = random_state
        self._ipca: IncrementalPCA | None = None
        self._rp: GaussianRandomProjection | None = None

    @property
    def fitted(self) -> bool:
        if self.method == "none":
            return True
        if self.method == "pca":
            return self._ipca is not None and hasattr(self._ipca, "components_")
        if self.method == "rp":
            return self._rp is not None
        return True

    def partial_fit(self, X: np.ndarray) -> None:
        if self.method == "none" or self.out_dim is None:
            return
        X = np.asarray(X, dtype=np.float64)
        if self.method == "pca":
            if self._ipca is None:
                self._ipca = IncrementalPCA(n_components=min(self.out_dim, X.shape[1]))
            self._ipca.partial_fit(X)
        elif self.method == "rp":
            if self._rp is None:
                # GaussianRandomProjection does not need fitting, but calling fit once
                # sets the components.
                self._rp = GaussianRandomProjection(n_components=self.out_dim, random_state=self.random_state)
                self._rp.fit(X[: min(10, X.shape[0])])

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "none" or self.out_dim is None:
            return X
        if self.method == "pca":
            if self._ipca is None:
                raise RuntimeError("IncrementalPCA has not been fitted")
            return self._ipca.transform(X)
        if self.method == "rp":
            if self._rp is None:
                self._rp = GaussianRandomProjection(n_components=self.out_dim, random_state=self.random_state)
                self._rp.fit(X[: min(10, X.shape[0])])
            return self._rp.transform(X)
        return X

    def info(self) -> str:
        if self.method == "none" or self.out_dim is None:
            return "none"
        return f"{self.method}(out_dim={self.out_dim})"


def _build_acoustic_features(
    segment: Segment,
    trf_config: TRFConfig,
    offset_frames: int,
    voicing_cols: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    sound = shift_sound_forward(segment.sound, offset_frames) if offset_frames else segment.sound
    representation = getattr(trf_config, "audio_representation", "handcrafted").lower()
    feature_list: List[np.ndarray] = []
    if representation.startswith("transformer"):
        feats = np.asarray(sound, dtype=np.float64)
        vmask = np.ones(feats.shape[0], dtype=bool)
    else:
        if "broadband_env" in trf_config.acoustic_features:
            feature_list.append(
                broadband_envelope(
                    sound, n_mels=trf_config.mel_n_bands, smooth_win=trf_config.mel_smooth_win
                )
            )
        if "slow_env" in trf_config.acoustic_features:
            feature_list.append(
                slow_envelope(
                    sound, n_mels=trf_config.mel_n_bands, smooth_win=max(trf_config.mel_smooth_win, 41)
                )
            )
        if "energy" in trf_config.acoustic_features:
            feature_list.append(energy_feature(sound, n_mels=trf_config.mel_n_bands))
        if "mel_multi" in trf_config.acoustic_features:
            feature_list.append(
                mel_features_from_sound_matrix(
                    sound,
                    n_mels=trf_config.mel_n_bands,
                    bands="multi",
                    smooth_win=trf_config.mel_smooth_win,
                )
            )
        if not feature_list:
            feature_list.append(
                mel_features_from_sound_matrix(
                    sound,
                    n_mels=trf_config.mel_n_bands,
                    bands=trf_config.mel_mode,
                    smooth_win=trf_config.mel_smooth_win,
                )
            )
        feats = np.concatenate(feature_list, axis=1)
        vmask = voiced_mask_from_sound(segment.sound, voicing_cols)
    return feats, vmask


def _prepare_targets(
    segment: Segment,
    roi_channels: Sequence[int] | None,
    trf_config: TRFConfig,
    eeg_stats_cache: Dict[Tuple[str, int], Tuple[float, float]],
) -> np.ndarray:
    eeg = segment.eeg[:, roi_channels] if roi_channels else segment.eeg
    ys = []
    for ch in range(eeg.shape[1]):
        ch_idx = roi_channels[ch] if roi_channels else ch
        z = zscore_eeg(
            eeg[:, ch],
            mode=trf_config.eeg_zscore_mode,
            subject_id=segment.subject_id,
            channel_idx=ch_idx,
            channel_stats_cache=eeg_stats_cache,
        )
        ys.append(highpass_moving_average(z, win=trf_config.eeg_highpass_win))
    if not ys:
        return np.zeros(eeg.shape[0])
    return np.mean(np.vstack(ys), axis=0)


def _segment_design_iter(
    segment: Segment,
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    voicing_cols: Sequence[int],
    eeg_stats_cache: Dict[Tuple[str, int], Tuple[float, float]],
    reducer: FeatureReducer,
    chunk_rows: int | None,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    feats, vmask = _build_acoustic_features(segment, trf_config, offset_frames=offset_frames, voicing_cols=voicing_cols)
    y = _prepare_targets(segment, roi_channels=roi_channels, trf_config=trf_config, eeg_stats_cache=eeg_stats_cache)

    T = min(feats.shape[0], y.shape[0], len(vmask))
    feats = feats[:T]
    vmask = vmask[:T]
    y = y[:T]
    reducer.partial_fit(feats)
    feats_reduced = reducer.transform(feats)

    for start, X_chunk in build_lagged_features_lazy(
        feats_reduced,
        n_pre=trf_config.n_pre,
        n_post=trf_config.n_post,
        voicing=vmask,
        chunk_rows=chunk_rows,
    ):
        y_chunk = y[start : start + X_chunk.shape[0]]
        yield np.asarray(X_chunk, dtype=np.float64), np.asarray(y_chunk, dtype=np.float64)


def _fit_scaler_and_targets(
    segments: List[Segment],
    idx: Sequence[int],
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    voicing_cols: Sequence[int],
    reducer: FeatureReducer,
    chunk_rows: int | None,
) -> tuple[StandardScaler, Tuple[float, float]]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    y_stats = RunningStats()
    eeg_stats_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for i in idx:
        seg = segments[i]
        for X_chunk, y_chunk in _segment_design_iter(
            seg,
            trf_config=trf_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            eeg_stats_cache=eeg_stats_cache,
            reducer=reducer,
            chunk_rows=chunk_rows,
        ):
            scaler.partial_fit(X_chunk)
            y_stats.update(y_chunk)
    y_mu, y_sd = y_stats.finalize()
    return scaler, (y_mu, y_sd)


def _stream_train_ridge(
    segments: List[Segment],
    idx: Sequence[int],
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    voicing_cols: Sequence[int],
    reducer: FeatureReducer,
    chunk_rows: int | None,
    scaler: StandardScaler,
    y_stats: Tuple[float, float],
    max_exact_dim: int = 5000,
) -> Tuple[np.ndarray | SGDRegressor, str]:
    y_mu, y_sd = y_stats
    eeg_stats_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
    n_features = scaler.mean_.shape[0]
    use_exact = n_features <= max_exact_dim and trf_config.solver == "ridge_sklearn"
    if use_exact:
        A = np.zeros((n_features, n_features), dtype=np.float64)
        b = np.zeros(n_features, dtype=np.float64)
    else:
        sgd = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=trf_config.ridge_alpha,
            learning_rate="optimal",
            random_state=trf_config.random_state,
            warm_start=True,
        )
        initialized = False

    for i in idx:
        seg = segments[i]
        for X_chunk, y_chunk in _segment_design_iter(
            seg,
            trf_config=trf_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            eeg_stats_cache=eeg_stats_cache,
            reducer=reducer,
            chunk_rows=chunk_rows,
        ):
            Xs = scaler.transform(X_chunk)
            ys = (y_chunk - y_mu) / y_sd
            if use_exact:
                A += Xs.T @ Xs
                b += Xs.T @ ys
            else:
                if not initialized:
                    sgd.partial_fit(Xs, ys)
                    initialized = True
                else:
                    sgd.partial_fit(Xs, ys)

    if use_exact:
        ridge_mat = A + trf_config.ridge_alpha * np.eye(A.shape[0])
        w = np.linalg.solve(ridge_mat, b)
        return w, "ridge_exact"
    if not initialized:
        raise RuntimeError("SGDRegressor failed to initialize; no data seen")
    return sgd, "sgd"


def _predict_stream(
    segments: List[Segment],
    idx: Sequence[int],
    trf_config: TRFConfig,
    roi_channels: Sequence[int] | None,
    offset_frames: int,
    voicing_cols: Sequence[int],
    reducer: FeatureReducer,
    chunk_rows: int | None,
    scaler: StandardScaler,
    y_stats: Tuple[float, float],
    model: np.ndarray | SGDRegressor,
    model_type: str,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    y_mu, y_sd = y_stats
    eeg_stats_cache: Dict[Tuple[str, int], Tuple[float, float]] = {}
    seg_corrs: List[float] = []
    seg_corrs_null: List[float] = []
    r2_scores: List[float] = []
    null_r2_scores: List[float] = []

    for i in idx:
        seg = segments[i]
        y_true_concat: List[float] = []
        y_pred_concat: List[float] = []
        y_pred_null_concat: List[float] = []
        for X_chunk, y_chunk in _segment_design_iter(
            seg,
            trf_config=trf_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            eeg_stats_cache=eeg_stats_cache,
            reducer=reducer,
            chunk_rows=chunk_rows,
        ):
            Xs = scaler.transform(X_chunk)
            ys = (y_chunk - y_mu) / y_sd
            if model_type == "ridge_exact":
                pred = Xs @ model
                null_pred = np.roll(Xs, shift=max(1, int(0.1 * len(Xs))), axis=0) @ model
            else:
                pred = model.predict(Xs)
                null_pred = model.predict(np.roll(Xs, shift=max(1, int(0.1 * len(Xs))), axis=0))
            y_true_concat.append(ys)
            y_pred_concat.append(pred)
            y_pred_null_concat.append(null_pred)

        if not y_true_concat:
            continue
        y_true_all = np.concatenate(y_true_concat)
        y_pred_all = np.concatenate(y_pred_concat)
        y_pred_null_all = np.concatenate(y_pred_null_concat)

        metrics = evaluate_predictions(y_true_all, y_pred_all)
        metrics_null = evaluate_predictions(y_true_all, y_pred_null_all)
        r2_scores.append(metrics.r2)
        null_r2_scores.append(metrics_null.r2)
        if np.std(y_pred_all) > 0 and np.std(y_true_all) > 0:
            seg_corrs.append(float(np.corrcoef(y_true_all, y_pred_all)[0, 1]))
        if np.std(y_pred_null_all) > 0 and np.std(y_true_all) > 0:
            seg_corrs_null.append(float(np.corrcoef(y_true_all, y_pred_null_all)[0, 1]))

    return seg_corrs, seg_corrs_null, r2_scores, null_r2_scores


def eval_subject_trf_envelope(
    segments: List[Segment],
    subject_id: str,
    trf_config: TRFConfig,
    n_splits: int = 5,
    max_segments: int | None = None,
    random_state: int = 42,
    roi_channels: Sequence[int] | None = None,
    offset_frames: int = 0,
    voicing_cols: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Evaluate an envelope-level TRF model for a single subject using streaming."""

    subject_segments = [s for s in segments if s.subject_id == subject_id]
    if max_segments is not None:
        subject_segments = subject_segments[:max_segments]
    if not subject_segments:
        return {"subject_id": subject_id, "note": "no segments"}

    voicing_cols = voicing_cols or []
    groups = np.arange(len(subject_segments))
    n_groups = len(groups)
    n_splits_use = min(n_splits, n_groups) if n_groups > 1 else 1
    splitter = GroupKFold(n_splits=n_splits_use) if n_splits_use > 1 else None

    seg_corrs: List[float] = []
    seg_corrs_null: List[float] = []
    r2_scores: List[float] = []
    null_r2_scores: List[float] = []

    reducer = FeatureReducer(
        method=trf_config.feature_reduce_method,
        out_dim=trf_config.feature_reduce_out_dim,
        random_state=trf_config.random_state,
    )
    chunk_rows = trf_config.data_chunk_rows
    logger.info(
        "Streaming TRF: subject=%s segments=%d chunk_rows=%s reducer=%s",
        subject_id,
        len(subject_segments),
        chunk_rows,
        reducer.info(),
    )
    # Probe dimensions for logging
    probe_seg = subject_segments[0]
    feats_probe, vmask_probe = _build_acoustic_features(probe_seg, trf_config, offset_frames=offset_frames, voicing_cols=voicing_cols)
    reducer.partial_fit(feats_probe)
    feats_red = reducer.transform(feats_probe)
    lag_cols = feats_red.shape[1] * (trf_config.n_pre + trf_config.n_post + 1) + 1
    logger.info(
        "Example segment shape T=%d D=%d -> reduced D=%d -> lagged cols=%d",
        feats_probe.shape[0],
        feats_probe.shape[1],
        feats_red.shape[1],
        lag_cols,
    )

    if trf_config.ridge_alpha_grid:
        logger.warning("Alpha grid tuning is skipped in streaming mode; using alpha=%s", trf_config.ridge_alpha)
        tuned_config = replace(trf_config, ridge_alpha_grid=None)
    else:
        tuned_config = trf_config

    if splitter:
        for tr_idx, va_idx in splitter.split(np.zeros(len(groups)), groups=groups):
            logger.info("Fold train segments=%d, val segments=%d", len(tr_idx), len(va_idx))
            scaler, y_stats = _fit_scaler_and_targets(
                subject_segments,
                idx=tr_idx,
                trf_config=tuned_config,
                roi_channels=roi_channels,
                offset_frames=offset_frames,
                voicing_cols=voicing_cols,
                reducer=reducer,
                chunk_rows=chunk_rows,
            )
            model, model_type = _stream_train_ridge(
                subject_segments,
                idx=tr_idx,
                trf_config=tuned_config,
                roi_channels=roi_channels,
                offset_frames=offset_frames,
                voicing_cols=voicing_cols,
                reducer=reducer,
                chunk_rows=chunk_rows,
                scaler=scaler,
                y_stats=y_stats,
            )
            seg_r, seg_r_null, r2s, null_r2s = _predict_stream(
                subject_segments,
                idx=va_idx,
                trf_config=tuned_config,
                roi_channels=roi_channels,
                offset_frames=offset_frames,
                voicing_cols=voicing_cols,
                reducer=reducer,
                chunk_rows=chunk_rows,
                scaler=scaler,
                y_stats=y_stats,
                model=model,
                model_type=model_type,
            )
            seg_corrs.extend(seg_r)
            seg_corrs_null.extend(seg_r_null)
            r2_scores.extend(r2s)
            null_r2_scores.extend(null_r2s)
    else:
        scaler, y_stats = _fit_scaler_and_targets(
            subject_segments,
            idx=list(range(len(subject_segments))),
            trf_config=tuned_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            reducer=reducer,
            chunk_rows=chunk_rows,
        )
        model, model_type = _stream_train_ridge(
            subject_segments,
            idx=list(range(len(subject_segments))),
            trf_config=tuned_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            reducer=reducer,
            chunk_rows=chunk_rows,
            scaler=scaler,
            y_stats=y_stats,
        )
        seg_r, seg_r_null, r2s, null_r2s = _predict_stream(
            subject_segments,
            idx=list(range(len(subject_segments))),
            trf_config=tuned_config,
            roi_channels=roi_channels,
            offset_frames=offset_frames,
            voicing_cols=voicing_cols,
            reducer=reducer,
            chunk_rows=chunk_rows,
            scaler=scaler,
            y_stats=y_stats,
            model=model,
            model_type=model_type,
        )
        seg_corrs.extend(seg_r)
        seg_corrs_null.extend(seg_r_null)
        r2_scores.extend(r2s)
        null_r2_scores.extend(null_r2s)

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
        "solver": trf_config.solver,
        "reducer": reducer.info(),
    }


def run_trf_analysis_per_subject(
    segments: List[Segment],
    trf_config: TRFConfig,
    n_splits: int = 5,
    roi_map: Mapping[str, Sequence[int]] | None = None,
    offset_map: Mapping[str, int] | None = None,
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
            voicing_cols=voicing_cols,
        )
        results.append(res)
    return pd.DataFrame(results)


__all__ = ["eval_subject_trf_envelope", "run_trf_analysis_per_subject"]
