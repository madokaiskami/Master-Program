"""Split utilities for robust evaluation protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut


@dataclass
class SplitConfig:
    strategy: str
    n_splits: int = 5


def get_groups(meta: dict) -> np.ndarray:
    groups = meta.get("subject")
    if groups is None:
        groups = meta.get("groups")
    if groups is None:
        groups = np.arange(meta["n_samples"], dtype=int)
    return np.asarray(groups)


def group_kfold_indices(n_samples: int, groups: Sequence, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.arange(n_samples), groups=groups))


def loso_indices(n_samples: int, groups: Sequence) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = LeaveOneGroupOut()
    return list(splitter.split(np.arange(n_samples), groups=groups))


def build_splits(n_samples: int, meta: dict, config: SplitConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = get_groups(meta)
    if config.strategy == "group_kfold":
        return group_kfold_indices(n_samples, groups, config.n_splits)
    if config.strategy == "loso":
        return loso_indices(n_samples, groups)
    raise ValueError(f"Unknown split strategy: {config.strategy}")


def scale_per_subject(X: np.ndarray, meta: dict) -> np.ndarray:
    """Apply z-score normalisation per subject."""

    groups = get_groups(meta)
    X_scaled = np.empty_like(X, dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        segment = X[mask]
        mean = segment.mean(axis=0, keepdims=True)
        std = segment.std(axis=0, keepdims=True) + 1e-8
        X_scaled[mask] = (segment - mean) / std
    return X_scaled


__all__ = [
    "SplitConfig",
    "build_splits",
    "scale_per_subject",
]
