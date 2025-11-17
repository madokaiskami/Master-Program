"""Model factory utilities."""

from __future__ import annotations

from typing import Any, Dict

from .sklearn_models import build_sklearn_model


def build_model(config: Dict[str, Any]):
    backend = config.get("backend", "sklearn")
    if backend == "sklearn":
        return build_sklearn_model(config)
    raise ValueError(f"Unsupported model backend: {backend}")


__all__ = ["build_model"]
