"""Configuration dataclasses for ADT-style EEG→audio decoding."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from eeg_audio_benchmark.config import load_config


@dataclass
class ADTModelConfig:
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    use_channel_attention: bool = True
    causal: bool = False


@dataclass
class ADTTrainingConfig:
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42


@dataclass
class ADTExperimentConfig:
    data_root: Path = Path("data/hf_eeg_audio")
    manifest_path: Path = Path("data/hf_eeg_audio/manifest_epochs.csv")
    results_dir: Path = Path("results/adt")
    target_feature: str = "broadband_envelope"
    n_mels: int = 40
    adt_model: ADTModelConfig = field(default_factory=ADTModelConfig)
    training: ADTTrainingConfig = field(default_factory=ADTTrainingConfig)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.manifest_path = Path(self.manifest_path)
        self.results_dir = Path(self.results_dir)


def _apply_overrides(obj: Any, overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if is_dataclass(current):
            if isinstance(value, Mapping):
                _apply_overrides(current, value)
        else:
            if isinstance(current, Path):
                setattr(obj, key, Path(value))
            else:
                setattr(obj, key, value)


def load_adt_experiment_config_from_yaml(path: str | Path) -> ADTExperimentConfig:
    """Load an ADT experiment configuration from a YAML file."""

    cfg_dict: Dict[str, Any] = load_config(path)
    config = ADTExperimentConfig()
    _apply_overrides(config, cfg_dict or {})
    return config


__all__ = [
    "ADTExperimentConfig",
    "ADTModelConfig",
    "ADTTrainingConfig",
    "load_adt_experiment_config_from_yaml",
]
