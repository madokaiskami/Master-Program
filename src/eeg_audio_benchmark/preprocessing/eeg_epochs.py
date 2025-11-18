"""EEG epoch slicing utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import dataframe_from_file, ensure_directory, gaussian_smooth


CBYT_DTYPE = np.dtype([
    ("time_sec", "d"),
    ("marker", "i"),
    ("data", "d", (32,)),
])


@dataclass
class EEGEpochConfig:
    output_dir: str
    epoch_duration: float
    anchor: str = "auto"
    log_level: str = "INFO"
    cbyt_files: Optional[List[str]] = None
    cbyt_dir: Optional[str] = None
    cbyt_pattern: str = "*.CBYT"
    stimulus_table: Optional[str] = None
    stimulus_sheet: Optional[str] = None
    stim_sequence_column: str = "Sequence_Number"
    stim_wav_column: str = "WAV_Filename"
    stim_start_column: str = "Start_Time"
    stim_end_column: str = "End_Time"
    marker_table: Optional[str] = None
    marker_sheet: Optional[str] = None
    marker_start_column: str = "time of beginning"
    marker_end_column: str = "time of the end"
    resample_hz: Optional[float] = None
    smoothing_sigma: float = 0.0


@dataclass
class StimulusRecord:
    sequence: int
    wav_base: str
    start: Optional[float]
    end: Optional[float]


def _discover_cbyt_files(config: EEGEpochConfig) -> List[Path]:
    if config.cbyt_files:
        return [Path(p) for p in config.cbyt_files]
    if not config.cbyt_dir:
        raise ValueError("Either cbyt_files or cbyt_dir must be provided")
    return sorted(Path(config.cbyt_dir).expanduser().glob(config.cbyt_pattern))


def _load_table_records(
    path: Optional[str],
    sheet_name: Optional[str],
    sequence_column: str,
    wav_column: Optional[str] = None,
    start_column: Optional[str] = None,
    end_column: Optional[str] = None,
) -> Dict[int, Dict[str, Optional[float]]]:
    if not path:
        return {}
    df = dataframe_from_file(Path(path), sheet_name)
    records: Dict[int, Dict[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        if sequence_column not in row or pd.isna(row[sequence_column]):
            continue
        try:
            seq = int(row[sequence_column])
        except Exception:
            continue
        rec: Dict[str, Optional[float]] = {}
        if wav_column and wav_column in row and pd.notna(row[wav_column]):
            rec["wav_base"] = Path(str(row[wav_column])).stem
        if start_column and start_column in row and pd.notna(row[start_column]):
            rec["start"] = float(row[start_column])
        if end_column and end_column in row and pd.notna(row[end_column]):
            rec["end"] = float(row[end_column])
        records[seq] = rec
    return records


def _merge_records(config: EEGEpochConfig) -> Dict[int, StimulusRecord]:
    stim_records = _load_table_records(
        config.stimulus_table,
        config.stimulus_sheet,
        config.stim_sequence_column,
        wav_column=config.stim_wav_column,
        start_column=config.stim_start_column,
        end_column=config.stim_end_column,
    )
    marker_records = _load_table_records(
        config.marker_table,
        config.marker_sheet,
        config.stim_sequence_column,
        start_column=config.marker_start_column,
        end_column=config.marker_end_column,
    )
    merged: Dict[int, StimulusRecord] = {}
    keys = set(stim_records) | set(marker_records)
    for seq in sorted(keys):
        stim = stim_records.get(seq, {})
        marker = marker_records.get(seq, {})
        wav_base = stim.get("wav_base") or f"stim_{seq:03d}"
        start = stim.get("start")
        end = stim.get("end")
        if marker.get("start") is not None:
            start = marker["start"]
        if marker.get("end") is not None:
            end = marker["end"]
        merged[seq] = StimulusRecord(sequence=seq, wav_base=wav_base, start=start, end=end)
    return merged


def _load_cbyt_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.fromfile(path, dtype=CBYT_DTYPE)
    if raw.size == 0:
        raise ValueError(f"No samples found in {path}")
    return raw["time_sec"], raw["data"].astype(np.float32)


def _resample(times: np.ndarray, data: np.ndarray, target_rate: Optional[float]) -> tuple[np.ndarray, np.ndarray]:
    if not target_rate:
        return times, data
    start = times[0]
    stop = times[-1]
    step = 1.0 / target_rate
    new_times = np.arange(start, stop, step)
    interp_data = np.vstack(
        [np.interp(new_times, times, data[:, ch]) for ch in range(data.shape[1])]
    ).T
    return new_times, interp_data.astype(np.float32)


def _epoch_bounds(record: StimulusRecord, duration: float, anchor: str) -> Optional[tuple[float, float]]:
    anchor = anchor.lower()
    start = record.start
    end = record.end
    if anchor == "start" and start is not None:
        return start, start + duration
    if anchor == "end" and end is not None:
        return end - duration, end
    if anchor == "auto":
        if start is not None:
            return start, start + duration
        if end is not None:
            return end - duration, end
    return None


def _build_epoch(times: np.ndarray, data: np.ndarray, start: float, end: float, sequence: int) -> Optional[np.ndarray]:
    mask = (times >= start) & (times <= end)
    if not mask.any():
        return None
    rel_times = times[mask] - start
    eeg = data[mask]
    markers = np.full(rel_times.shape, -1.0, dtype=np.float32)
    markers[0] = float(sequence)
    epoch = np.column_stack((rel_times.astype(np.float32), markers, eeg))
    return epoch


def slice_eeg_to_epochs(config: EEGEpochConfig) -> List[Path]:
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)
    ensure_directory(Path(config.output_dir))
    records = _merge_records(config)
    if not records:
        raise ValueError("No stimulus records available")

    output_files: List[Path] = []
    for cbyt_path in _discover_cbyt_files(config):
        logger.info("Processing %s", cbyt_path)
        times, data = _load_cbyt_file(cbyt_path)
        times, data = _resample(times, data, config.resample_hz)
        if config.smoothing_sigma > 0:
            data = gaussian_smooth(data, config.smoothing_sigma)
        base = cbyt_path.stem
        for record in records.values():
            bounds = _epoch_bounds(record, config.epoch_duration, config.anchor)
            if not bounds:
                continue
            start, end = bounds
            epoch = _build_epoch(times, data, start, end, record.sequence)
            if epoch is None:
                logger.debug("No samples for stimulus %s in %s", record.sequence, cbyt_path)
                continue
            fname = f"{base}_{record.sequence:03d}_{record.wav_base}.npy"
            out_path = Path(config.output_dir) / fname
            np.save(out_path, epoch.astype(np.float32))
            output_files.append(out_path)
    logger.info("Saved %d epochs to %s", len(output_files), config.output_dir)
    return output_files


__all__ = ["EEGEpochConfig", "slice_eeg_to_epochs"]
