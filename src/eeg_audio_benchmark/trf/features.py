"""Audio feature utilities for TRF analyses."""

from __future__ import annotations

from typing import List, Sequence

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


def envelope_from_mel(S: np.ndarray, n_mels: int = 40, smooth_win: int = 9) -> np.ndarray:
    """Construct a broadband envelope from mel-band features.

    Steps (matching the notebooks):
    1. Take the first ``n_mels`` bands.
    2. Compute the L2 norm across bands per frame.
    3. Z-score the envelope.
    4. Apply causal moving-average smoothing.
    """

    S = np.asarray(S, dtype=np.float64)
    mel = S[:, : min(n_mels, S.shape[1])]
    env = np.sqrt((mel**2).sum(axis=1))
    mu, sd = env.mean(), env.std()
    env = (env - mu) / (sd if sd > 0 else 1.0)
    if smooth_win and smooth_win > 1:
        env = causal_moving_average(env, win=smooth_win)
    return env


def envelope_from_sound_matrix(
    S: np.ndarray,
    use_rms_column: int | None = None,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> np.ndarray:
    """Extract or construct a broadband envelope from an audio feature matrix."""

    if use_rms_column is not None and 0 <= use_rms_column < S.shape[1]:
        env = np.asarray(S[:, use_rms_column], dtype=np.float64)
        mu, sd = env.mean(), env.std()
        env = (env - mu) / (sd if sd > 0 else 1.0)
    else:
        env = envelope_from_mel(S, n_mels=n_mels, smooth_win=smooth_win)
    return env


def envelope_for_segments(
    segments: List[Segment],
    n_mels: int = 40,
    smooth_win: int = 9,
    use_rms_column: int | None = None,
) -> List[np.ndarray]:
    """Construct envelopes for a list of segments."""

    envelopes: List[np.ndarray] = []
    for seg in segments:
        envelopes.append(
            envelope_from_sound_matrix(
                seg.sound, use_rms_column=use_rms_column, n_mels=n_mels, smooth_win=smooth_win
            )
        )
    return envelopes


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
    """Moving-average high-pass filter used for EEG in the notebooks."""

    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y
    if y.size < 2 * win:
        mu, sd = y.mean(), y.std()
        return (y - mu) / (sd if sd > 0 else 1.0)
    kernel = np.ones(win, dtype=np.float64) / win
    trend = np.convolve(y, kernel, mode="same")
    hp = y - trend
    mu, sd = hp.mean(), hp.std()
    return (hp - mu) / (sd if sd > 0 else 1.0)


def preprocess_eeg_channel(y: np.ndarray, highpass_win: int = 15) -> np.ndarray:
    """Z-score and high-pass filter a single EEG channel."""

    y = np.asarray(y, dtype=np.float64)
    mu, sd = y.mean(), y.std()
    y = (y - mu) / (sd if sd > 0 else 1.0)
    return highpass_moving_average(y, win=highpass_win)


def build_lagged_features(env: np.ndarray, n_pre: int, n_post: int, voicing: np.ndarray | None = None) -> np.ndarray:
    """Build lagged features from an envelope vector (optionally with voicing).

    Lags run from ``-n_pre`` (past) to ``+n_post`` (future) inclusive, matching
    the "build_lagged_design" helper in the notebooks.
    """

    if env.ndim != 1:
        raise ValueError("Envelope must be a 1D array")
    lags = list(range(-n_pre, n_post + 1))
    T = env.shape[0]
    X = np.zeros((T, len(lags)), dtype=np.float64)
    for i, lag in enumerate(lags):
        if lag >= 0:
            X[lag:, i] = env[: T - lag]
        else:
            X[: T + lag, i] = env[-lag:]
    if voicing is None:
        return X
    v = np.asarray(voicing, dtype=np.float64).reshape(-1)
    v = v[:T]
    return np.hstack([X[:T], v.reshape(-1, 1)])


__all__ = [
    "build_lagged_features",
    "causal_moving_average",
    "envelope_from_mel",
    "envelope_for_segments",
    "envelope_from_sound_matrix",
    "highpass_moving_average",
    "preprocess_eeg_channel",
    "voiced_mask_from_sound",
]
