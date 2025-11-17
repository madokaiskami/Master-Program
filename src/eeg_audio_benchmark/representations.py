"""Feature representations for EEG and audio signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np


RepresentationFunc = Callable[[np.ndarray, dict], np.ndarray]


@dataclass
class Representation:
    """Simple registry entry describing a representation."""

    name: str
    func: RepresentationFunc
    default_params: dict

    def __call__(self, data: np.ndarray, **kwargs) -> np.ndarray:
        params = {**self.default_params, **kwargs}
        return self.func(data, params)


# ---------------------------------------------------------------------------
# EEG representations
# ---------------------------------------------------------------------------

def eeg_identity(data: np.ndarray, params: dict) -> np.ndarray:
    return data


def eeg_bandpower(data: np.ndarray, params: dict) -> np.ndarray:
    """Compute simple band power features using FFT magnitude integration."""

    sfreq = params.get("sfreq")
    if sfreq is None:
        raise ValueError("'sfreq' must be provided for band-power representation")

    bands: Iterable[Tuple[float, float]] = params.get(
        "bands",
        [(1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0)],
    )

    fft = np.fft.rfft(data, axis=-1)
    freqs = np.fft.rfftfreq(data.shape[-1], d=1.0 / sfreq)
    power = np.abs(fft) ** 2

    features = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        features.append(power[..., mask].mean(axis=-1))
    return np.stack(features, axis=-1)


# ---------------------------------------------------------------------------
# Audio representations
# ---------------------------------------------------------------------------

def audio_identity(data: np.ndarray, params: dict) -> np.ndarray:
    return data


def audio_envelope(data: np.ndarray, params: dict) -> np.ndarray:
    """Compute a simple rectified-signal envelope."""

    window = int(params.get("smoothing", 25))
    window = max(window, 1)
    rectified = np.abs(data)
    if window == 1:
        return rectified

    kernel = np.ones(window) / window
    if rectified.ndim == 1:
        return np.convolve(rectified, kernel, mode="same")

    smoothed = np.empty_like(rectified, dtype=float)
    for feat in range(rectified.shape[1]):
        smoothed[:, feat] = np.convolve(rectified[:, feat], kernel, mode="same")
    return smoothed


EEG_REPRESENTATIONS: Dict[str, Representation] = {
    "time": Representation("time", eeg_identity, {}),
    "bandpower": Representation("bandpower", eeg_bandpower, {}),
}

AUDIO_REPRESENTATIONS: Dict[str, Representation] = {
    "time": Representation("time", audio_identity, {}),
    "envelope": Representation("envelope", audio_envelope, {"smoothing": 25}),
}


def get_representation(name: str, kind: str) -> Representation:
    if kind == "eeg":
        registry = EEG_REPRESENTATIONS
    elif kind == "audio":
        registry = AUDIO_REPRESENTATIONS
    else:
        raise ValueError(f"Unknown representation kind: {kind}")

    if name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown {kind} representation '{name}'. Available: {available}")
    return registry[name]


__all__ = [
    "Representation",
    "EEG_REPRESENTATIONS",
    "AUDIO_REPRESENTATIONS",
    "get_representation",
]
