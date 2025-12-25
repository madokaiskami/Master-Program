"""Training and evaluation loop for the ADT Transformer baseline."""

from __future__ import annotations

import datetime
import logging
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import GroupKFold

from eeg_audio_benchmark.hf_data import prepare_data_for_training
from eeg_audio_benchmark.trf.data import Segment, filter_and_summarize, load_segments_from_hf_manifest
from eeg_audio_benchmark.trf.features import broadband_envelope, mel_features_from_sound_matrix

from .config import ADTExperimentConfig
from .eval import aggregate_subject_metrics, compute_segment_metrics
from .model import EEGToEnvelopeADT

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _standardize(array: np.ndarray) -> np.ndarray:
    mu = array.mean(axis=0, keepdims=True)
    sd = array.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (array - mu) / sd


def _resolve_manifest_path(data_root: Path, manifest_path: Path) -> Path:
    path = manifest_path if manifest_path.is_absolute() else data_root / manifest_path
    if path.exists():
        return path
    fallback = data_root / "derivatives" / "epoch_manifest.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Manifest not found at {path}")


def _extract_target(segment: Segment, feature: str, n_mels: int) -> np.ndarray:
    if feature == "multi_band_envelope":
        return mel_features_from_sound_matrix(segment.sound, n_mels=n_mels, bands="multi", smooth_win=9)
    return broadband_envelope(segment.sound, n_mels=n_mels, smooth_win=9)


class ADTSegmentDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Dataset yielding EEG tensors, target envelope tensors, and subject ids."""

    def __init__(self, segments: Sequence[Segment], target_feature: str, n_mels: int):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, str]] = []
        lengths: List[int] = []
        for seg in segments:
            target = _extract_target(seg, target_feature, n_mels=n_mels)
            T = min(seg.eeg.shape[0], target.shape[0])
            if T <= 1:
                continue
            eeg = _standardize(seg.eeg[:T]).astype(np.float32)
            env = target[:T].astype(np.float32)
            self.samples.append(
                (torch.from_numpy(eeg), torch.from_numpy(env), str(seg.subject_id))
            )
            lengths.append(T)

        if not self.samples:
            logger.warning("No valid segments available for ADT dataset.")
            self.min_length = 0
            return

        self.min_length = min(lengths)
        if len(set(lengths)) > 1:
            logger.info("Cropping all segments to min length %d frames for batching", self.min_length)
            self.samples = [
                (eeg[: self.min_length], env[: self.min_length], sid)
                for eeg, env, sid in self.samples
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.samples[idx]

    @property
    def subject_ids(self) -> List[str]:
        return [sid for _, _, sid in self.samples]


def _build_dataloader(dataset: Dataset, indices: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def _train_single_fold(
    model: EEGToEnvelopeADT,
    train_loader: DataLoader,
    exp_config: ADTExperimentConfig,
    device: torch.device,
) -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_config.training.learning_rate,
        weight_decay=exp_config.training.weight_decay,
    )
    criterion = nn.MSELoss()
    for epoch in range(exp_config.training.num_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            eeg, env, _ = batch
            eeg = eeg.to(device)
            env = env.to(device)

            optimizer.zero_grad()
            preds = model(eeg)
            loss = criterion(preds, env)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        logger.debug(
            "Epoch %d/%d - training loss %.4f",
            epoch + 1,
            exp_config.training.num_epochs,
            epoch_loss / max(1, len(train_loader)),
        )


def _evaluate_model(
    model: EEGToEnvelopeADT,
    dataset: ADTSegmentDataset,
    indices: Iterable[int],
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    model.eval()
    metrics: dict[str, list[dict[str, float]]] = {}
    with torch.no_grad():
        for idx in indices:
            eeg, env, subject_id = dataset[idx]
            eeg = eeg.unsqueeze(0).to(device)
            env_np = env.numpy()
            pred = model(eeg).squeeze(0).cpu().numpy()
            seg_metrics = compute_segment_metrics(pred, env_np)
            metrics.setdefault(subject_id, []).append(seg_metrics)
    return metrics


def run_adt_experiment(config: ADTExperimentConfig) -> pd.DataFrame:
    """Train and evaluate the ADT baseline, returning a per-subject summary DataFrame."""

    logging.basicConfig(level=logging.INFO)
    _set_seed(config.training.seed)

    data_root = Path(config.data_root)
    prepare_data_for_training()

    manifest_path = _resolve_manifest_path(data_root, config.manifest_path)
    segments = load_segments_from_hf_manifest(root=data_root, manifest_path=manifest_path)
    segments = filter_and_summarize(segments, min_frames=20)
    if not segments:
        logger.warning("No segments available after filtering.")
        return pd.DataFrame()

    dataset = ADTSegmentDataset(segments, target_feature=config.target_feature, n_mels=config.n_mels)
    if len(dataset) == 0:
        logger.warning("ADT dataset is empty. Exiting without results.")
        return pd.DataFrame()

    device_str = config.training.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    unique_subjects = sorted(set(dataset.subject_ids))
    n_splits = min(5, len(unique_subjects)) if len(unique_subjects) > 1 else 1
    splitter = GroupKFold(n_splits=n_splits) if n_splits > 1 else None

    all_metrics: dict[str, list[dict[str, float]]] = {}
    groups = np.array(dataset.subject_ids)
    indices = np.arange(len(dataset))
    splits = splitter.split(indices, groups=groups) if splitter else [(indices, indices)]

    for fold_idx, (train_idx, eval_idx) in enumerate(splits):
        logger.info("Fold %d/%d: train %d segments, eval %d segments", fold_idx + 1, n_splits, len(train_idx), len(eval_idx))
        sample_eeg, sample_env, _ = dataset[0]
        model = EEGToEnvelopeADT(
            n_channels=sample_eeg.shape[1],
            out_dim=sample_env.shape[1],
            config=config.adt_model,
        )
        train_loader = _build_dataloader(
            dataset, train_idx, batch_size=config.training.batch_size, shuffle=True
        )
        _train_single_fold(model, train_loader, exp_config=config, device=device)
        fold_metrics = _evaluate_model(model, dataset, eval_idx, device=device)
        for sid, records in fold_metrics.items():
            all_metrics.setdefault(sid, []).extend(records)

    results = aggregate_subject_metrics(all_metrics)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = config.results_dir / f"adt_results_{timestamp}.csv"
    results.to_csv(output_path, index=False)
    logger.info("Saved ADT results to %s", output_path)
    return results


__all__ = ["run_adt_experiment"]
