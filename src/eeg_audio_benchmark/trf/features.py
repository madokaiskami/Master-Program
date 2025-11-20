"""Audio feature utilities for TRF analyses."""

from __future__ import annotations

from typing import List

import numpy as np

from .data import Segment


def envelope_from_matrix(S: np.ndarray, n_mels: int = 40) -> np.ndarray:
    """Compute a broadband envelope from the first ``n_mels`` columns of a feature matrix."""

    if S.ndim != 2 or S.shape[1] == 0:
        raise ValueError("Sound matrix must be 2D with nonzero feature dimension.")
    n = min(n_mels, S.shape[1])
    mel = S[:, :n]
    e = np.sqrt((mel**2).sum(axis=1))
    std = e.std()
    if std == 0:
        return e - e.mean()
    return (e - e.mean()) / std


def causal_moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Causal moving average with integer window length ``win`` over time."""

    if win <= 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    smoothed = (cumsum[win:] - cumsum[:-win]) / float(win)
    # pad with the first smoothed value to preserve length
    pad = np.full(win - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def envelope_from_sound_matrix(
    S: np.ndarray,
    use_rms_column: int | None = None,
    n_mels: int = 40,
    smooth_win: int = 9,
) -> np.ndarray:
    """Extract or construct a broadband envelope from an audio feature matrix."""

    if use_rms_column is not None and 0 <= use_rms_column < S.shape[1]:
        env = S[:, use_rms_column]
    else:
        env = envelope_from_matrix(S, n_mels=n_mels)
    if smooth_win and smooth_win > 1:
        env = causal_moving_average(env, smooth_win)
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


def build_lagged_features(env: np.ndarray, n_pre: int, n_post: int) -> np.ndarray:
    """Build lagged features from an envelope vector."""

    if env.ndim != 1:
        raise ValueError("Envelope must be a 1D array")
    lags = list(range(-n_pre, n_post + 1))
    T = env.shape[0]
    lagged = np.zeros((T, len(lags)), dtype=float)
    for i, lag in enumerate(lags):
        if lag < 0:
            lagged[-lag:, i] = env[: T + lag]
        elif lag > 0:
            lagged[: T - lag, i] = env[lag:]
        else:
            lagged[:, i] = env
    return lagged


__all__ = [
    "build_lagged_features",
    "causal_moving_average",
    "envelope_for_segments",
    "envelope_from_matrix",
    "envelope_from_sound_matrix",
]
