"""Transformer-based audio feature extraction for EEG encoding baselines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT

from .utils import ensure_directory, resolve_dataset_path


@dataclass
class TransformerFeatureConfig:
    """Configuration for extracting Transformer hidden states from stimuli."""

    dataset_root: str = str(LOCAL_DATA_ROOT)
    manifest_path: str = "{dataset_root}/manifest_epochs.csv"
    wav_dir: str = "{dataset_root}/raw/audio/stimuli"
    output_dir: str = "{dataset_root}/derivatives/transformer_features"
    feature_manifest: str = (
        "{dataset_root}/derivatives/transformer_features/manifest_transformer_features.csv"
    )
    model_name: str = "facebook/wav2vec2-base"
    layer: Optional[int] = None
    target_sr: int = 50          # 目标“特征时间轴”采样率(Hz)，不是 wav2vec2 的 16k
    batch_size: int = 1
    device: str = "cpu"
    log_level: str = "INFO"


def _unique_audio_files(manifest_path: Path, dataset_root: Path, wav_dir: Path) -> List[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    import pandas as pd

    manifest_df = pd.read_csv(manifest_path)
    if "audio_file" not in manifest_df.columns:
        raise ValueError("Manifest must contain an audio_file column")
    unique_audio = manifest_df["audio_file"].dropna().unique()
    audio_paths: List[Path] = []
    for entry in unique_audio:
        candidate = Path(str(entry))
        if not candidate.is_absolute():
            candidate = dataset_root / candidate
        if not candidate.exists():
            alt = wav_dir / candidate.name
            candidate = alt
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        audio_paths.append(candidate.resolve())
    return sorted(audio_paths)


def _load_transformer(model_name: str, device: str):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)

    # ---- compatibility: transformers>=? Wav2Vec2Processor has no .sampling_rate ----
    if not hasattr(processor, "sampling_rate"):
        fe = getattr(processor, "feature_extractor", None)
        sr = getattr(fe, "sampling_rate", None) if fe is not None else None
        setattr(processor, "sampling_rate", int(sr) if sr is not None else 16000)
    # ------------------------------------------------------------------------------

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return processor, model



def _select_hidden_states(outputs, layer: Optional[int]) -> np.ndarray:
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None and len(hidden_states):
        idx = layer if layer is not None else -1
        return hidden_states[idx].detach().cpu().numpy()
    last = getattr(outputs, "last_hidden_state", None)
    if last is None:
        raise ValueError("Model outputs do not contain hidden states")
    return last.detach().cpu().numpy()


def _resample_features(features: np.ndarray, duration: float, target_sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if duration <= 0:
        raise ValueError("Audio duration must be positive")
    if target_sr <= 0:
        raise ValueError("target_sr must be positive")
    if features.ndim != 3:
        raise ValueError(f"Expected features with shape (batch, frames, dim); got {features.shape}")
    seq_len = features.shape[1]
    times = np.linspace(0, duration, num=seq_len, endpoint=False, dtype=np.float32)
    target_times = np.arange(0, duration, 1.0 / float(target_sr), dtype=np.float32)
    resampled = np.vstack(
        [
            np.interp(target_times, times, features[0, :, dim], left=np.nan, right=np.nan)
            for dim in range(features.shape[2])
        ]
    ).T
    return np.nan_to_num(resampled, nan=0.0).astype(np.float32), target_times.astype(np.float32)


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _get_wav2vec_sampling_rate(processor) -> int:
    """
    兼容不同 transformers 版本：
    - 有的 processor 没有 sampling_rate
    - 采样率通常在 processor.feature_extractor.sampling_rate
    """
    sr = getattr(processor, "sampling_rate", None)
    if sr is None:
        fe = getattr(processor, "feature_extractor", None)
        sr = getattr(fe, "sampling_rate", None)
    if sr is None:
        sr = 16000
    return int(sr)


def _extract_single(
    wav_path: Path,
    processor,
    model,
    target_sr: int,
    layer: Optional[int],
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    # ---- FIX: 不再依赖 processor.sampling_rate ----
    wav2vec_sr = _get_wav2vec_sampling_rate(processor)

    # wav2vec2 输入音频需要被 resample 到 wav2vec_sr（通常 16k）
    waveform, _ = librosa.load(wav_path, sr=wav2vec_sr)

    # 显式传 sampling_rate，避免不同版本行为差异
    inputs = processor(waveform, sampling_rate=wav2vec_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = _select_hidden_states(outputs, layer=layer)

    # duration 用 wav2vec_sr 算
    duration = len(waveform) / float(wav2vec_sr)

    resampled, times = _resample_features(hidden, duration=duration, target_sr=target_sr)
    return resampled, times


def extract_transformer_features(config: TransformerFeatureConfig) -> List[Path]:
    """Extract Transformer hidden states for all unique stimuli in the manifest."""

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    dataset_root = Path(config.dataset_root)
    manifest_path = resolve_dataset_path(config.manifest_path, dataset_root)
    wav_dir = resolve_dataset_path(config.wav_dir, dataset_root) or dataset_root
    output_dir = resolve_dataset_path(config.output_dir, dataset_root)
    manifest_out = resolve_dataset_path(config.feature_manifest, dataset_root)
    if manifest_path is None or output_dir is None or manifest_out is None:
        raise ValueError("manifest_path, output_dir, and feature_manifest must be provided")

    ensure_directory(output_dir)
    audio_paths = _unique_audio_files(manifest_path, dataset_root, wav_dir)
    if not audio_paths:
        logger.warning("No audio files discovered for Transformer features")
        return []

    logger.info(
        "Extracting Transformer features with %s (layer=%s) for %d files",
        config.model_name,
        config.layer if config.layer is not None else "last",
        len(audio_paths),
    )
    processor, model = _load_transformer(config.model_name, config.device)

    outputs: List[Path] = []
    manifest_rows: List[Dict[str, object]] = []
    for wav_path in audio_paths:
        logger.info("Processing %s", wav_path.name)
        features, times = _extract_single(
            wav_path=wav_path,
            processor=processor,
            model=model,
            target_sr=config.target_sr,
            layer=config.layer,
            device=config.device,
        )
        base = wav_path.stem
        feat_path = output_dir / f"{base}_transformer_features.npy"
        time_path = output_dir / f"{base}_transformer_times.npy"
        np.save(feat_path, features.astype(np.float32))
        np.save(time_path, times.astype(np.float32))
        outputs.append(feat_path)
        manifest_rows.append(
            {
                "audio_file": _relative_to_root(wav_path, dataset_root),
                "audio_stem": base,
                "feature_path": _relative_to_root(feat_path, dataset_root),
                "time_path": _relative_to_root(time_path, dataset_root),
                "model_name": config.model_name,
                "layer": config.layer if config.layer is not None else -1,
                "target_sr": config.target_sr,
            }
        )

    import pandas as pd

    ensure_directory(manifest_out.parent)
    pd.DataFrame(manifest_rows).to_csv(manifest_out, index=False)
    logger.info("Transformer feature manifest saved to %s", manifest_out)
    return outputs


__all__ = ["TransformerFeatureConfig", "extract_transformer_features"]
