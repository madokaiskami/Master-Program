"""Evaluation utilities for EEG/audio regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

@dataclass
class MetricResult:
    r2: float
    pearson: float


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    if y_true.ndim == 1:
        return _corr(y_true, y_pred)

    corrs = [_corr(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]
    return float(np.nanmean(corrs))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    return MetricResult(r2=r2_score(y_true, y_pred), pearson=pearson_corr(y_true, y_pred))


def noise_ceiling_split_half(data: np.ndarray) -> float:
    """Compute a simple split-half noise ceiling estimate."""

    n = data.shape[0]
    half = n // 2
    part_a = data[:half]
    part_b = data[-half:]
    return float(pearson_corr(part_a.mean(axis=0), part_b.mean(axis=0)))


__all__ = [
    "MetricResult",
    "evaluate_predictions",
    "noise_ceiling_split_half",
]
