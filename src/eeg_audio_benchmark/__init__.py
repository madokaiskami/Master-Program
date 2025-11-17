"""EEG–Audio regression benchmark framework."""

from .config import load_config
from .datasets import EegAudioDataset
from .experiment import ExperimentRunner

__all__ = [
    "load_config",
    "EegAudioDataset",
    "ExperimentRunner",
]
