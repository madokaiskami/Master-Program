"""Experiment runner for EEG/audio benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json

from .config import load_config
from .datasets import DatasetConfig, EegAudioDataset
from .evaluation import evaluate_predictions
from .models import build_model
from .splits import SplitConfig


@dataclass
class ExperimentConfig:
    dataset: Dict
    model: Dict
    splits: Dict
    output_dir: str


class ExperimentRunner:
    """Tie dataset, model, and evaluation into a unified workflow."""

    def __init__(self, config: Dict):
        self.config = config
        dataset_cfg = DatasetConfig(**config["dataset"])
        self.dataset = EegAudioDataset(dataset_cfg)
        self.model_config = config["model"]
        self.split_config = SplitConfig(**config["splits"])
        self.output_dir = Path(config.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[Dict]:
        splits = self.dataset.get_splits(self.split_config)
        results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            model = build_model(self.model_config)
            X_train, Y_train = self.dataset.X[train_idx], self.dataset.Y[train_idx]
            X_test, Y_test = self.dataset.X[test_idx], self.dataset.Y[test_idx]

            model.fit(X_train, Y_train)
            preds = model.predict(X_test)
            metrics = evaluate_predictions(Y_test, preds)
            fold_result = {
                "fold": fold_idx,
                "r2": metrics.r2,
                "pearson": metrics.pearson,
            }
            results.append(fold_result)

        self._save_results(results)
        return results

    def _save_results(self, results: List[Dict]) -> None:
        output_path = self.output_dir / "results.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


def run_from_config(path: str | Path) -> List[Dict]:
    config = load_config(path)
    runner = ExperimentRunner(config)
    return runner.run()


__all__ = ["ExperimentRunner", "run_from_config"]
