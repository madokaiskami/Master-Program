"""ADT-style EEG→audio Transformer baseline."""

from .config import ADTExperimentConfig, ADTModelConfig, ADTTrainingConfig, load_adt_experiment_config_from_yaml
from .model import EEGToEnvelopeADT
from .train import run_adt_experiment

__all__ = [
    "ADTExperimentConfig",
    "ADTModelConfig",
    "ADTTrainingConfig",
    "EEGToEnvelopeADT",
    "load_adt_experiment_config_from_yaml",
    "run_adt_experiment",
]
