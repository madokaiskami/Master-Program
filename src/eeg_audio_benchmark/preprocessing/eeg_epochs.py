"""EEG epoch slicing utilities tailored for the HF dataset layout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT

from .utils import ensure_directory, gaussian_smooth, resolve_dataset_path


@dataclass
class EEGEpochConfig:
    """Configuration for slicing continuous EEG into epochs using HF metadata."""

    dataset_root: str = str(LOCAL_DATA_ROOT)
    manifest_raw_runs: str = "{dataset_root}/manifest_raw_runs.csv"
    output_dir: str = "{dataset_root}/derivatives/epochs"
    epoch_manifest: str = "{dataset_root}/derivatives/epoch_manifest.csv"
    epoch_duration_sec: float = 4.0
    anchor: str = "onset"  # "onset" or "center"
    resample_hz: Optional[float] = None
    smoothing_sigma: float = 0.0
    log_level: str = "INFO"


def _load_eeg_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["times"], data["data"].astype(np.float32)


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"subject_id", "run_id", "onset_sec", "stim_id", "audio_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events file {path} missing columns: {sorted(missing)}")
    return df


def _resample(times: np.ndarray, data: np.ndarray, target_rate: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if not target_rate:
        return times, data
    start, stop = times[0], times[-1]
    step = 1.0 / target_rate
    new_times = np.arange(start, stop, step, dtype=np.float64)
    interp_data = np.vstack(
        [np.interp(new_times, times, data[:, ch]) for ch in range(data.shape[1])]
    ).T
    return new_times, interp_data.astype(np.float32)


def _event_bounds(row: pd.Series, duration: float, anchor: str) -> Optional[Tuple[float, float]]:
    anchor = anchor.lower()
    onset = float(row["onset_sec"])
    offset = row.get("offset_sec")
    if anchor == "center" and pd.notna(offset):
        center = 0.5 * (onset + float(offset))
        start = center - duration / 2.0
    else:
        start = onset
    end = start + duration
    return start, end


def _build_epoch(times: np.ndarray, data: np.ndarray, start: float, end: float, event_index: int) -> Optional[np.ndarray]:
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    rel_times = times[mask] - start
    eeg = data[mask]
    markers = np.full(rel_times.shape, float(event_index), dtype=np.float32)
    epoch = np.column_stack((rel_times.astype(np.float32), markers, eeg))
    return epoch


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def slice_eeg_to_epochs(config: EEGEpochConfig) -> List[Path]:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    dataset_root = Path(config.dataset_root)
    manifest_path = resolve_dataset_path(config.manifest_raw_runs, dataset_root)
    output_dir = resolve_dataset_path(config.output_dir, dataset_root)
    manifest_out = resolve_dataset_path(config.epoch_manifest, dataset_root)

    if manifest_path is None or output_dir is None or manifest_out is None:
        raise ValueError("Configuration must provide manifest_raw_runs, output_dir, and epoch_manifest")

    ensure_directory(output_dir)

    runs_df = pd.read_csv(manifest_path)
    required_cols = {"subject_id", "run_id", "eeg_file", "events_file"}
    missing_cols = required_cols - set(runs_df.columns)
    if missing_cols:
        raise ValueError(f"Raw runs manifest missing columns: {sorted(missing_cols)}")

    manifest_rows: List[dict] = []
    output_files: List[Path] = []

    for _, run in runs_df.iterrows():
        eeg_path = dataset_root / run["eeg_file"]
        events_path = dataset_root / run["events_file"]
        if not eeg_path.exists():
            raise FileNotFoundError(eeg_path)
        if not events_path.exists():
            raise FileNotFoundError(events_path)

        logger.info("Processing %s | %s", run.get("subject_id"), run.get("run_id"))
        times, data = _load_eeg_npz(eeg_path)
        times, data = _resample(times, data, config.resample_hz)
        if config.smoothing_sigma > 0:
            data = gaussian_smooth(data, config.smoothing_sigma)

        events_df = _load_events(events_path)
        for event_idx, event_row in events_df.iterrows():
            start, end = _event_bounds(event_row, config.epoch_duration_sec, config.anchor)
            epoch = _build_epoch(times, data, start, end, event_idx)
            if epoch is None:
                logger.debug(
                    "Skipping event %s (no samples between %.2f-%.2f)",
                    event_row.get("stim_id", event_idx),
                    start,
                    end,
                )
                continue

            stim_id = str(event_row.get("stim_id", f"evt{event_idx:03d}"))
            base = f"{run['subject_id']}_{run['run_id']}_evt-{event_idx:03d}_{stim_id}"
            epoch_path = output_dir / f"{base}.npy"
            np.save(epoch_path, epoch.astype(np.float32))
            output_files.append(epoch_path)

            manifest_rows.append(
                {
                    "subject_id": run["subject_id"],
                    "run_id": run["run_id"],
                    "event_index": int(event_idx),
                    "stim_id": stim_id,
                    "audio_file": event_row["audio_file"],
                    "onset_sec": float(event_row["onset_sec"]),
                    "offset_sec": float(event_row["offset_sec"]) if pd.notna(event_row.get("offset_sec")) else None,
                    "epoch_path": _relative_to_root(epoch_path, dataset_root),
                    "sampling_rate_hz": float(run.get("sampling_rate_hz", np.nan)),
                    "n_channels": int(run.get("n_channels", data.shape[1])),
                    "n_samples": int(epoch.shape[0]),
                    "audio_stem": Path(str(event_row["audio_file"])).stem,
                }
            )

    if not manifest_rows:
        raise ValueError("No epochs were generated; check configuration and event bounds")

    ensure_directory(manifest_out.parent)
    pd.DataFrame(manifest_rows).to_csv(manifest_out, index=False)
    logger.info("Saved %d epochs to %s", len(output_files), output_dir)
    logger.info("Epoch manifest written to %s", manifest_out)
    return output_files


__all__ = ["EEGEpochConfig", "slice_eeg_to_epochs"]
