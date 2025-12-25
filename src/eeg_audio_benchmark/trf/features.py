"""Audio feature utilities for TRF analyses."""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np

from .data import Segment


def causal_moving_average(x: np.ndarray, win: int = 9) -> np.ndarray:
    """Causal moving average filter mirroring the notebook implementation.

    The output is re-normalized (z-scored) to match the behavior of the
    ``causal_ma`` helper in the reference notebooks.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or win <= 1:
        return x
    out = np.empty_like(x)
    csum = 0.0
    for i in range(len(x)):
        csum += x[i]
        if i >= win:
            csum -= x[i - win]
        out[i] = csum / min(i + 1, win)
    mu, sd = out.mean(), out.std()
    return (out - mu) / (sd if sd > 0 else 1.0)


def mel_features_from_sound_matrix(
    S: np.ndarray,
    n_mels: int = 40,
    bands: str = "multi",
    smooth_win: int = 9,
) -> np.ndarray:
    """Construct mel-derived acoustic features.

    Parameters
    ----------
    S:
        Input sound matrix of shape (T, D).
    n_mels:
        Number of mel bands to keep (first ``n_mels`` columns).
    bands:
        "multi" for band-wise features, "envelope" for broadband envelope.
    smooth_win:
        Optional causal smoothing window applied per band (``multi``) or to the
        broadband envelope (``envelope``) to mirror notebook behavior.
    """

    S = np.asarray(S, dtype=np.float64)
    mel = S[:, : min(n_mels, S.shape[1])]
    if bands == "envelope":
        env = np.sqrt((mel**2).sum(axis=1))
        mu, sd = env.mean(), env.std()
        env = (env - mu) / (sd if sd > 0 else 1.0)
        if smooth_win and smooth_win > 1:
            env = causal_moving_average(env, win=smooth_win)
        return env.reshape(-1, 1)

    # Multi-band: z-score each band and optionally apply causal smoothing
    mu = mel.mean(axis=0)
    sd = mel.std(axis=0)
    sd[sd == 0] = 1.0
    feats = (mel - mu) / sd
    if smooth_win and smooth_win > 1:
        feats = np.vstack([causal_moving_average(feats[:, i], win=smooth_win) for i in range(feats.shape[1])]).T
    return feats


def broadband_envelope(S: np.ndarray, n_mels: int = 40, smooth_win: int = 9) -> np.ndarray:
    """Broadband envelope constructed from mel bands with causal smoothing."""

    mel = mel_features_from_sound_matrix(S, n_mels=n_mels, bands="multi", smooth_win=0)
    env = np.sqrt((mel**2).sum(axis=1))
    mu, sd = env.mean(), env.std()
    env = (env - mu) / (sd if sd > 0 else 1.0)
    if smooth_win and smooth_win > 1:
        env = causal_moving_average(env, win=smooth_win)
    return env.reshape(-1, 1)


def slow_envelope(S: np.ndarray, n_mels: int = 40, smooth_win: int = 41) -> np.ndarray:
    """Slowly varying broadband envelope using a long smoothing window."""

    return broadband_envelope(S, n_mels=n_mels, smooth_win=smooth_win)


def energy_feature(S: np.ndarray, n_mels: int = 40) -> np.ndarray:
    """Frame-level energy across mel bands (mean energy per frame)."""

    mel = mel_features_from_sound_matrix(S, n_mels=n_mels, bands="multi", smooth_win=0)
    energy = mel.mean(axis=1)
    mu, sd = energy.mean(), energy.std()
    energy = (energy - mu) / (sd if sd > 0 else 1.0)
    return energy.reshape(-1, 1)


def voiced_mask_from_sound(S: np.ndarray, voicing_cols: Sequence[int], threshold: float = 0.5) -> np.ndarray:
    """Build a voiced mask mirroring the notebook logic.

    If any of the provided ``voicing_cols`` contain finite values, use them to
    define voiced frames. Otherwise fall back to an energy percentile-based
    heuristic (40th percentile of broadband energy).
    """

    T, D = S.shape
    if voicing_cols:
        vc = [c for c in voicing_cols if c < D]
        if vc:
            mask = np.any(np.isfinite(S[:, vc]), axis=1)
            if mask.any():
                return mask.astype(bool)
    energy = np.sqrt((np.nan_to_num(S) ** 2).sum(axis=1))
    thr = np.percentile(energy[~np.isnan(energy)], 40) if np.any(~np.isnan(energy)) else 0.0
    return energy > thr


def highpass_moving_average(y: np.ndarray, win: int = 15) -> np.ndarray:
    """Causal moving-average high-pass filter used for EEG channels."""

    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or win <= 1:
        return y
    if y.size < 2 * win:
        mu, sd = y.mean(), y.std()
        return (y - mu) / (sd if sd > 0 else 1.0)
    kernel = np.ones(win, dtype=np.float64) / win
    trend = np.convolve(y, kernel, mode="same")
    hp = y - trend
    mu, sd = hp.mean(), hp.std()
    return (hp - mu) / (sd if sd > 0 else 1.0)


def zscore_eeg(
    eeg: np.ndarray,
    mode: str,
    subject_id: str,
    channel_idx: int,
    channel_stats_cache: Dict[Tuple[str, int], Tuple[float, float]],
) -> np.ndarray:
    """Z-score EEG according to the requested mode.

    Parameters
    ----------
    eeg:
        Array of shape (T,) for a single channel.
    mode:
        "per_subject_channel" uses cached stats across all segments per subject
        and channel. "per_segment_channel" computes stats independently per
        segment.
    subject_id:
        Subject identifier for cache keying.
    channel_stats_cache:
        Mutable cache mapping ``(subject_id, channel_idx)`` to ``(mean, std)``.
    """

    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.size == 0:
        return eeg
    if mode == "per_subject_channel":
        key = (subject_id, int(channel_idx))
        if key not in channel_stats_cache:
            mu, sd = eeg.mean(), eeg.std()
            channel_stats_cache[key] = (mu, sd if sd > 0 else 1.0)
        mu, sd = channel_stats_cache[key]
    else:
        mu, sd = eeg.mean(), eeg.std()
        sd = sd if sd > 0 else 1.0
    return (eeg - mu) / sd


def build_lagged_features(
    features: np.ndarray, n_pre: int, n_post: int, voicing: np.ndarray | None = None
) -> np.ndarray:
    """Build lagged features from acoustic inputs (optionally with voicing).

    Lags run from ``-n_pre`` (past) to ``+n_post`` (future) inclusive. For
    multi-dimensional acoustic inputs, lags are tiled per feature dimension in
    column-major order.
    """

    if features.ndim != 2:
        raise ValueError("Features must be a 2D array (T, D)")
    lags = list(range(-n_pre, n_post + 1))
    T, D = features.shape
    X = np.zeros((T, len(lags) * D), dtype=np.float64)
    for j in range(D):
        env = features[:, j]
        for i, lag in enumerate(lags):
            col = j * len(lags) + i
            if lag >= 0:
                X[lag:, col] = env[: T - lag]
            else:
                X[: T + lag, col] = env[-lag:]
    if voicing is None:
        return X
    v = np.asarray(voicing, dtype=np.float64).reshape(-1)
    v = v[:T]
    return np.hstack([X[:T], v.reshape(-1, 1)])


def build_lagged_features_lazy(
    features: np.ndarray,
    n_pre: int,
    n_post: int,
    voicing: np.ndarray | None = None,
    chunk_rows: int | None = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Generate lagged design chunks without materializing the full matrix.

    Parameters
    ----------
    features:
        Input acoustic features of shape (T, D).
    n_pre / n_post:
        Number of past/future lags (inclusive) to include.
    voicing:
        Optional voiced mask aligned with ``features`` (1D array of length ``T``).
    chunk_rows:
        Optional chunk size along the time dimension. If ``None``, yields a single
        full design matrix. Otherwise, lags are computed using a local window with
        sufficient context around each chunk to avoid boundary artifacts.
    """

    if chunk_rows is None or chunk_rows <= 0:
        yield 0, build_lagged_features(features, n_pre=n_pre, n_post=n_post, voicing=voicing)
        return

    T = features.shape[0]
    for start in range(0, T, chunk_rows):
        end = min(start + chunk_rows, T)
        ctx_start = max(0, start - n_pre)
        ctx_end = min(T, end + n_post)
        local_feats = features[ctx_start:ctx_end]
        local_voicing = voicing[ctx_start:ctx_end] if voicing is not None else None
        lagged_local = build_lagged_features(local_feats, n_pre=n_pre, n_post=n_post, voicing=local_voicing)
        offset = start - ctx_start
        yield start, lagged_local[offset : offset + (end - start)]


__all__ = [
    "build_lagged_features",
    "build_lagged_features_lazy",
    "causal_moving_average",
    "mel_features_from_sound_matrix",
    "broadband_envelope",
    "slow_envelope",
    "energy_feature",
    "highpass_moving_average",
    "zscore_eeg",
    "voiced_mask_from_sound",
]
