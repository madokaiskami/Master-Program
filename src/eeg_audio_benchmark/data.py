"""Data loading helpers for EEG/audio benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_numpy(path: str | Path) -> np.ndarray:
    """Load a NumPy array from ``.npy`` file."""

    array_path = Path(path)
    if not array_path.exists():
        raise FileNotFoundError(f"Array file not found: {array_path}")
    return np.load(array_path)


def load_eeg_audio_pair(eeg_path: str | Path, audio_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load paired EEG and audio arrays.

    The arrays are expected to be aligned sample-by-sample.
    """

    eeg = load_numpy(eeg_path)
    audio = load_numpy(audio_path)

    if eeg.shape[0] != audio.shape[0]:
        raise ValueError(
            "EEG and audio arrays must share the same first dimension (samples)."
        )

    return eeg, audio


def load_metadata(path: str | Path) -> Dict[str, np.ndarray]:
    """Load metadata saved as ``.npz`` or ``.npy``.

    The metadata is expected to include at least the ``subject`` field when
    available in order to support group-aware splits.
    """

    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if metadata_path.suffix == ".npz":
        with np.load(metadata_path) as data:
            return {k: data[k] for k in data.files}
    return {"values": np.load(metadata_path, allow_pickle=True).item()}


__all__ = ["load_numpy", "load_eeg_audio_pair", "load_metadata"]
