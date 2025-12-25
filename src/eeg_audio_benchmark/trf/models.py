"""TRF encoder models built on top of sklearn Ridge regression."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


@dataclass
class TRFConfig:
    """Configuration for a ridge-based TRF encoder.

    Defaults maintain the legacy single-envelope, fixed-``alpha`` behavior
    while allowing richer acoustic features and nested hyperparameters to be
    controlled via YAML.
    """

    n_pre: int = 5
    n_post: int = 10

    audio_representation: str = "handcrafted"
    audio_path_column: str | None = None
    transformer_feature_dir: str | None = None
    transformer_layer: int | None = None

    ridge_alpha: float = 1.0
    ridge_alpha_grid: Optional[Sequence[float]] = None
    ridge_cv_folds: int = 3

    mel_n_bands: int = 40
    mel_mode: str = "envelope"
    mel_smooth_win: int = 9

    acoustic_features: List[str] = field(default_factory=lambda: ["broadband_env"])

    eeg_highpass_win: int = 15
    eeg_zscore_mode: str = "per_segment_channel"

    @property
    def alpha(self) -> float:
        """Legacy alias for ridge_alpha."""

        return self.ridge_alpha


class TRFEncoder:
    """Time-lagged encoding model using ridge regression."""

    def __init__(self, config: TRFConfig):
        self.config = config
        self.model: Optional[Ridge] = None

    def _build_model(self) -> Ridge | GridSearchCV:
        if self.config.ridge_alpha_grid:
            param_grid = {"alpha": list(self.config.ridge_alpha_grid)}
            return GridSearchCV(
                Ridge(),
                param_grid=param_grid,
                cv=self.config.ridge_cv_folds,
                scoring="neg_mean_squared_error",
            )
        return Ridge(alpha=self.config.ridge_alpha)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model.

        Parameters
        ----------
        X:
            Lagged acoustic features of shape (N, L).
        Y:
            EEG data of shape (N, C).
        """

        model = self._build_model()
        model.fit(X, Y)
        # GridSearchCV exposes the best estimator which implements predict.
        self.model = model.best_estimator_ if isinstance(model, GridSearchCV) else model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict EEG given lagged features."""

        if self.model is None:
            raise RuntimeError("TRFEncoder has not been fit yet.")
        return self.model.predict(X)


__all__ = ["TRFConfig", "TRFEncoder"]
