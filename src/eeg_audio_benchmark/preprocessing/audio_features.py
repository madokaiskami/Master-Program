"""Audio feature extraction utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np

from .utils import ensure_directory


@dataclass
class AudioFeatureConfig:
    wav_dir: str
    output_dir: str
    log_level: str = "INFO"
    wav_files: List[str] | None = None
    sampling_rate: int = 22050
    frame_length: float = 0.046
    hop_length: float = 0.011
    n_mfcc: int = 20
    fmin: float = 50.0
    fmax: float | None = None
    pyin_fmin: float = 50.0
    pyin_fmax: float = 600.0


def _discover_wavs(config: AudioFeatureConfig) -> List[Path]:
    if config.wav_files:
        return [Path(p) for p in config.wav_files]
    return sorted(Path(config.wav_dir).glob("*.wav"))


def _stack_features(feature_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    matrices: List[np.ndarray] = []
    names: List[str] = []
    n_frames = None
    for name, arr in feature_dict.items():
        arr = np.atleast_2d(arr)
        if n_frames is None:
            n_frames = arr.shape[1]
        elif arr.shape[1] != n_frames:
            raise ValueError("Feature length mismatch")
        for idx in range(arr.shape[0]):
            matrices.append(arr[idx])
            suffix = f"_{idx}" if arr.shape[0] > 1 else ""
            names.append(f"{name}{suffix}")
    feature_matrix = np.stack(matrices, axis=0).T.astype(np.float32)
    return feature_matrix, names


def _extract_single(path: Path, config: AudioFeatureConfig) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y, sr = librosa.load(path, sr=config.sampling_rate)
    frame_samples = int(round(config.frame_length * sr))
    hop_samples = int(round(config.hop_length * sr))
    if frame_samples <= 0 or hop_samples <= 0:
        raise ValueError("Invalid frame/hop configuration")
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.n_mfcc,
        n_fft=frame_samples,
        hop_length=hop_samples,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms = librosa.feature.rms(y=y, frame_length=frame_samples, hop_length=hop_samples)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_samples, hop_length=hop_samples)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_samples, hop_length=hop_samples)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=frame_samples, hop_length=hop_samples)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_samples, hop_length=hop_samples)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=frame_samples, hop_length=hop_samples)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=frame_samples, hop_length=hop_samples)
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=config.pyin_fmin,
        fmax=config.pyin_fmax,
        sr=sr,
        frame_length=frame_samples,
        hop_length=hop_samples,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    voiced_flag = np.nan_to_num(voiced_flag.astype(float), nan=0.0)
    voiced_prob = np.nan_to_num(voiced_prob, nan=0.0)
    feature_dict = {
        "mfcc": mfcc,
        "mfcc_delta": delta,
        "mfcc_delta2": delta2,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "spectral_contrast": contrast,
        "f0": np.atleast_2d(f0),
        "voiced_flag": np.atleast_2d(voiced_flag),
        "voiced_prob": np.atleast_2d(voiced_prob),
    }
    matrix, names = _stack_features(feature_dict)
    times = librosa.frames_to_time(
        np.arange(matrix.shape[0]), sr=sr, hop_length=hop_samples
    ).astype(np.float32)
    return matrix, times, names


def extract_audio_features(config: AudioFeatureConfig) -> List[Path]:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    ensure_directory(Path(config.output_dir))
    outputs: List[Path] = []
    for wav_path in _discover_wavs(config):
        logger.info("Extracting features for %s", wav_path)
        matrix, times, names = _extract_single(wav_path, config)
        base = wav_path.stem
        feature_path = Path(config.output_dir) / f"{base}_features.npy"
        time_path = Path(config.output_dir) / f"{base}_feature_times.npy"
        names_path = Path(config.output_dir) / f"{base}_feature_names.txt"
        np.save(feature_path, matrix.astype(np.float32))
        np.save(time_path, times)
        names_path.write_text("\n".join(names), encoding="utf-8")
        outputs.append(feature_path)
    logger.info("Saved features for %d wav files", len(outputs))
    return outputs


__all__ = ["AudioFeatureConfig", "extract_audio_features"]
