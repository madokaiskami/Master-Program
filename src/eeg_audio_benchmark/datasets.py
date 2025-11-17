"""Dataset abstraction for EEG/audio regression experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .data import load_eeg_audio_pair, load_metadata
from .representations import get_representation
from .splits import SplitConfig, build_splits, scale_per_subject


@dataclass
class DatasetConfig:
    eeg_path: str
    audio_path: str
    metadata_path: str | None
    task: str
    eeg_representation: Dict
    audio_representation: Dict
    lags: Iterable[int]
    per_subject_scaling: bool = False


def _lag_matrix(data: np.ndarray, lags: Iterable[int]) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]

    lags = list(lags)
    if not lags:
        raise ValueError("At least one lag value must be provided.")

    lagged_features = []
    for lag in lags:
        if lag == 0:
            lagged_features.append(data)
            continue

        shifted = np.zeros_like(data)
        if lag > 0:
            shifted[lag:] = data[:-lag]
        else:
            shifted[:lag] = data[-lag:]
        lagged_features.append(shifted)

    return np.concatenate(lagged_features, axis=1)


class EegAudioDataset:
    """Unified access to aligned EEG and audio features."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.eeg, self.audio = load_eeg_audio_pair(config.eeg_path, config.audio_path)

        if config.metadata_path:
            self.meta = load_metadata(config.metadata_path)
        else:
            self.meta = {}
        self.meta.setdefault("n_samples", self.eeg.shape[0])

        self.X, self.Y = self._build_design_matrices()

    def _build_design_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        eeg_repr_cfg = self.config.eeg_representation
        audio_repr_cfg = self.config.audio_representation

        eeg_repr = get_representation(eeg_repr_cfg["name"], "eeg")
        audio_repr = get_representation(audio_repr_cfg["name"], "audio")

        eeg_features = eeg_repr(self.eeg, **eeg_repr_cfg.get("params", {}))
        audio_features = audio_repr(self.audio, **audio_repr_cfg.get("params", {}))

        if self.config.task == "encoding":
            X = _lag_matrix(audio_features, self.config.lags)
            Y = eeg_features
        elif self.config.task == "decoding":
            X = _lag_matrix(eeg_features, self.config.lags)
            Y = audio_features
        else:
            raise ValueError("Task must be either 'encoding' or 'decoding'.")

        if self.config.per_subject_scaling:
            X = scale_per_subject(X, self.meta)

        return X, Y

    def get_splits(self, split_config: SplitConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
        return build_splits(self.X.shape[0], self.meta, split_config)


__all__ = ["DatasetConfig", "EegAudioDataset"]
