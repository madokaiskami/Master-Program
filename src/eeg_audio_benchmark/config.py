"""Utilities for loading YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["load_config"]
