"""Helper utilities shared by preprocessing steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Type, TypeVar

import numpy as np
import pandas as pd

from eeg_audio_benchmark.config import load_config


T = TypeVar("T")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return data
    radius = max(1, int(round(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(data, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=padded
    )
    return smoothed


def resolve_dataset_path(path_like: Optional[str], dataset_root: Path) -> Optional[Path]:
    """Resolve a path template against a dataset root.

    Parameters
    ----------
    path_like:
        A path string that may contain ``{dataset_root}`` placeholders. ``None``
        returns ``None``.
    dataset_root:
        Base path used to expand placeholders.

    Returns
    -------
    Optional[Path]
        Resolved ``Path`` or ``None`` if ``path_like`` is ``None``.
    """

    if path_like is None:
        return None
    return Path(str(path_like).format(dataset_root=str(dataset_root)))


def dataframe_from_file(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Table file not found: {path}")
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    return pd.read_csv(path)


def parse_config(config_like: Dict[str, Any], cls: Type[T]) -> T:
    field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in config_like.items() if k in field_names}
    return cls(**filtered)  # type: ignore[arg-type]


def load_step_config(config_path: str | Path, cls: Type[T]) -> T:
    config_dict = load_config(config_path)
    return parse_config(config_dict, cls)


@dataclass
class SlidingWindow:
    size_samples: int
    step_samples: int

    def generate(self, n_samples: int) -> Iterable[tuple[int, int]]:
        start = 0
        while start + self.size_samples <= n_samples:
            yield start, start + self.size_samples
            start += self.step_samples


__all__ = [
    "ensure_directory",
    "gaussian_smooth",
    "resolve_dataset_path",
    "dataframe_from_file",
    "parse_config",
    "load_step_config",
    "SlidingWindow",
]
