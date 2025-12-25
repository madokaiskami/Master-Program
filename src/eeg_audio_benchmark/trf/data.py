"""HF-derivative segment loading and quality control utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """Container for a single aligned EEG/audio segment.

    Attributes
    ----------
    subject_id:
        Subject identifier for the segment.
    run_id:
        Optional run/session identifier.
    stim_id:
        Optional stimulus identifier.
    eeg:
        Array of shape (T, C) containing EEG frames aligned to the audio features.
    sound:
        Array of shape (T, D) containing aligned audio feature frames.
    event_index:
        Optional event index from the manifest.
    audio_file:
        Optional audio file name associated with the stimulus.
    """

    subject_id: str
    run_id: str | None
    stim_id: str | None
    eeg: np.ndarray
    sound: np.ndarray
    event_index: int | None = None
    audio_file: str | None = None
    eeg_path: str | None = None
    audio_path: str | None = None


_PATH_CANDIDATES = ["eeg_aligned_path", "eeg_aligned", "eeg_path", "eeg"]
_AUDIO_PATH_CANDIDATES = ["audio_aligned_path", "audio_aligned", "audio_path", "audio"]


def _resolve_manifest_path(root: Path, manifest_path: Path | None) -> Path:
    if manifest_path is not None:
        return manifest_path
    default = root / "manifest_epochs.csv"
    if default.exists():
        return default
    fallback = root / "derivatives" / "epoch_manifest.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not locate an aligned manifest. Provide manifest_path explicitly or "
        "ensure manifest_epochs.csv exists under the dataset root."
    )


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"None of the candidate columns found: {candidates}")


def _resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def load_segments_from_hf_manifest(
    root: Path | str = LOCAL_DATA_ROOT,
    manifest_path: Path | None = None,
    min_columns: Iterable[str] | None = None,
    audio_column: str | None = None,
    audio_candidates: Iterable[str] | None = None,
) -> List[Segment]:
    """Load aligned EEG/audio segments from an HF manifest.

    Parameters
    ----------
    root:
        Dataset root directory (default ``LOCAL_DATA_ROOT``).
    manifest_path:
        Optional explicit manifest path. If omitted, defaults to ``manifest_epochs.csv``
        under the dataset root, falling back to ``derivatives/epoch_manifest.csv``.
    min_columns:
        Optional iterable of columns that must be present in the manifest.
    audio_column:
        Optional explicit name of the column containing aligned audio/feature paths.
    audio_candidates:
        Optional iterable of candidate columns to search (takes precedence over
        defaults if provided).

    Returns
    -------
    List[Segment]
        Loaded segments with EEG and sound arrays.
    """

    dataset_root = Path(root)
    resolved_manifest = _resolve_manifest_path(dataset_root, manifest_path)
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Manifest not found at {resolved_manifest}")

    df = pd.read_csv(resolved_manifest)
    if min_columns:
        missing = [col for col in min_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

    eeg_col = _pick_column(df, _PATH_CANDIDATES)
    candidates = list(audio_candidates) if audio_candidates is not None else list(_AUDIO_PATH_CANDIDATES)
    if audio_column:
        candidates = [audio_column] + [c for c in candidates if c != audio_column]
    audio_col = _pick_column(df, candidates)

    segments: List[Segment] = []
    for _, row in df.iterrows():
        eeg_path = _resolve_path(dataset_root, Path(str(row[eeg_col])))
        audio_path = _resolve_path(dataset_root, Path(str(row[audio_col])))
        if not eeg_path.exists() or not audio_path.exists():
            logger.warning("Skipping missing pair: EEG=%s Audio=%s", eeg_path, audio_path)
            continue
        eeg = np.load(eeg_path)
        sound = np.load(audio_path)
        segments.append(
            Segment(
                subject_id=str(row.get("subject_id", "unknown")),
                run_id=None if pd.isna(row.get("run_id")) else str(row.get("run_id")),
                stim_id=None if pd.isna(row.get("stim_id")) else str(row.get("stim_id")),
                eeg=eeg,
                sound=sound,
                event_index=None if pd.isna(row.get("event_index")) else int(row.get("event_index")),
                audio_file=None if pd.isna(row.get("audio_file")) else str(row.get("audio_file")),
                eeg_path=str(eeg_path),
                audio_path=str(audio_path),
            )
        )

    logger.info("Loaded %d segments from %s", len(segments), resolved_manifest)
    return segments


def filter_and_summarize(segments: List[Segment], min_frames: int = 20) -> List[Segment]:
    """Filter out degenerate segments and log a summary."""

    kept: List[Segment] = []
    empty_time = 0
    empty_eeg = 0
    empty_audio = 0
    too_short = 0
    bad_dims = 0

    for seg in segments:
        if seg.eeg.ndim != 2 or seg.sound.ndim != 2:
            bad_dims += 1
            continue
        if seg.eeg.size == 0 or seg.sound.size == 0:
            empty_eeg += int(seg.eeg.size == 0)
            empty_audio += int(seg.sound.size == 0)
            continue
        if seg.eeg.shape[0] == 0 or seg.sound.shape[0] == 0:
            empty_time += 1
            continue
        if min(seg.eeg.shape[0], seg.sound.shape[0]) < min_frames:
            too_short += 1
            continue
        kept.append(seg)

    logger.info(
        "Total %d segments → kept %d (empty_time=%d, empty_eeg=%d, empty_audio=%d, short=%d, bad_dims=%d)",
        len(segments),
        len(kept),
        empty_time,
        empty_eeg,
        empty_audio,
        too_short,
        bad_dims,
    )
    return kept


def nan_inf_report(segments: List[Segment]) -> None:
    """Log how many segments contain NaNs or Infs in EEG or sound arrays."""

    eeg_nan = []
    eeg_inf = []
    sound_nan = []
    sound_inf = []
    for seg in segments:
        eeg_nan.append(int(np.isnan(seg.eeg).any()))
        eeg_inf.append(int(np.isinf(seg.eeg).any()))
        sound_nan.append(int(np.isnan(seg.sound).any()))
        sound_inf.append(int(np.isinf(seg.sound).any()))

    n = len(segments) if segments else 1
    logger.info(
        "EEG: segments=%d, with_NaN=%d (%.1f%%), with_Inf=%d (%.1f%%)",
        len(segments),
        sum(eeg_nan),
        100 * sum(eeg_nan) / n,
        sum(eeg_inf),
        100 * sum(eeg_inf) / n,
    )
    logger.info(
        "Sound: segments=%d, with_NaN=%d (%.1f%%), with_Inf=%d (%.1f%%)",
        len(segments),
        sum(sound_nan),
        100 * sum(sound_nan) / n,
        sum(sound_inf),
        100 * sum(sound_inf) / n,
    )


__all__ = [
    "Segment",
    "filter_and_summarize",
    "load_segments_from_hf_manifest",
    "nan_inf_report",
]
