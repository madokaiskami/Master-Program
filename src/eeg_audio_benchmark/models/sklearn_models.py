"""scikit-learn regressor wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

from .base import BaseRegressor


SKLEARN_MODEL_REGISTRY: Dict[str, Type] = {
    "ridge": Ridge,
    "lasso": Lasso,
    "elasticnet": ElasticNet,
    "random_forest": RandomForestRegressor,
}


@dataclass
class SklearnRegressor(BaseRegressor):
    model_name: str
    params: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.model_name not in SKLEARN_MODEL_REGISTRY:
            available = ", ".join(sorted(SKLEARN_MODEL_REGISTRY))
            raise KeyError(f"Unknown sklearn model '{self.model_name}'. Available: {available}")
        estimator_cls = SKLEARN_MODEL_REGISTRY[self.model_name]
        self.estimator = estimator_cls(**self.params)

    def fit(self, X: np.ndarray, Y: np.ndarray, meta: Dict | None = None) -> None:
        self.estimator.fit(X, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)


def build_sklearn_model(config: Dict[str, Any]) -> SklearnRegressor:
    name = config.get("name", "ridge")
    params = config.get("params", {})
    return SklearnRegressor(name, params)


__all__ = ["SklearnRegressor", "build_sklearn_model", "SKLEARN_MODEL_REGISTRY"]
