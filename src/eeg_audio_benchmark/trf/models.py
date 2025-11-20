"""TRF encoder models built on top of sklearn Ridge regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class TRFConfig:
    """Configuration for a simple ridge-based TRF encoder."""

    alpha: float
    n_pre: int
    n_post: int


class TRFEncoder:
    """Time-lagged encoding model using ridge regression."""

    def __init__(self, config: TRFConfig):
        self.config = config
        self.model: Optional[Ridge] = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model.

        Parameters
        ----------
        X:
            Lagged envelope features of shape (N, L).
        Y:
            EEG data of shape (N, C).
        """

        self.model = Ridge(alpha=self.config.alpha)
        self.model.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict EEG given lagged features."""

        if self.model is None:
            raise RuntimeError("TRFEncoder has not been fit yet.")
        return self.model.predict(X)


__all__ = ["TRFConfig", "TRFEncoder"]
