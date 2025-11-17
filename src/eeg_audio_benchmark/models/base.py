"""Model abstractions for EEG/audio regression."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseRegressor(ABC):
    """Common interface for regression models."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, meta: Dict | None = None) -> None:
        """Fit the model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for the given inputs."""


__all__ = ["BaseRegressor"]
