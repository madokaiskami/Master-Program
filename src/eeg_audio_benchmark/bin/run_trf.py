"""Run envelope-level TRF analysis using HF derivatives."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, Sequence

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT, prepare_data_for_training
from eeg_audio_benchmark.trf.data import (
    filter_and_summarize,
    load_segments_from_hf_manifest,
    nan_inf_report,
)
from eeg_audio_benchmark.trf.eval import run_trf_analysis_per_subject
from eeg_audio_benchmark.trf.models import TRFConfig
from eeg_audio_benchmark.trf.offset import pick_best_global_offset
from eeg_audio_benchmark.trf.roi import select_roi_channels_for_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRF envelope analysis on HF data")
    parser.add_argument("--config", required=True, help="Path to TRF YAML config")
    parser.add_argument("--preproc-config", type=Path, help="Optional preprocessing config")
    parser.add_argument(
        "--force-download", action="store_true", help="Force re-download of HF dataset"
    )
    parser.add_argument(
        "--force-preproc", action="store_true", help="Force re-running preprocessing"
    )
    parser.add_argument(
        "--skip-hf-sync",
        action="store_true",
        help="Skip HF sync/preprocessing (assumes derivatives already exist)",
    )
    return parser.parse_args()


def _resolve_dataset_root(config_dict: Dict[str, object]) -> Path:
    root = config_dict.get("dataset_root", str(LOCAL_DATA_ROOT))
    return Path(str(root))


def _resolve_manifest(config_dict: Dict[str, object], dataset_root: Path) -> Path:
    manifest = config_dict.get("manifest_path")
    if manifest is not None:
        return Path(str(manifest))
    default = dataset_root / "manifest_epochs.csv"
    if default.exists():
        return default
    fallback = dataset_root / "derivatives" / "epoch_manifest.csv"
    return fallback


def _ensure_results_dir(config_dict: Dict[str, object]) -> Path:
    base = Path(config_dict.get("output_dir", "results/trf"))
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = base / f"trf_results_{timestamp}.csv"
    return out_path


def _collect_roi_channels(
    segments,
    subject_ids: Sequence[str],
    roi_config: Dict[str, object],
    n_mels: int,
    smooth_win: int,
) -> Dict[str, Sequence[int]]:
    roi_map: Dict[str, Sequence[int]] = {}
    for sid in subject_ids:
        roi = select_roi_channels_for_subject(
            segments,
            subject_id=sid,
            max_lag_frames=int(roi_config.get("max_lag_frames", 10)),
            top_k=int(roi_config.get("top_k", 3)),
            n_mels=n_mels,
            smooth_win=smooth_win,
            voicing_column=roi_config.get("voicing_column"),
        )
        roi_map[sid] = roi
    return roi_map


def _collect_offsets(
    segments,
    subject_ids: Sequence[str],
    roi_map: Dict[str, Sequence[int]],
    offset_config: Dict[str, object],
    frame_hz: float,
    n_mels: int,
    smooth_win: int,
) -> Dict[str, int]:
    offsets_ms = offset_config.get("candidate_offsets_ms", [])
    candidate_offsets_frames = [int(round(ms / 1000.0 * frame_hz)) for ms in offsets_ms]
    max_lag_frames = int(offset_config.get("max_lag_frames", offset_config.get("roi_max_lag_frames", 10)))
    offset_map: Dict[str, int] = {}
    for sid in subject_ids:
        roi = roi_map.get(sid, [])
        offset_map[sid] = pick_best_global_offset(
            segments,
            subject_id=sid,
            roi_channels=roi,
            candidate_offsets_frames=candidate_offsets_frames,
            max_lag_frames=max_lag_frames,
            n_mels=n_mels,
            smooth_win=smooth_win,
        )
    return offset_map


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config_dict = load_config(args.config)
    dataset_root = _resolve_dataset_root(config_dict)

    if not args.skip_hf_sync:
        prepare_data_for_training(
            preproc_config_path=args.preproc_config,
            force_preproc=args.force_preproc,
            force_download=args.force_download,
        )

    manifest_path = _resolve_manifest(config_dict, dataset_root)
    segments = load_segments_from_hf_manifest(dataset_root, manifest_path=manifest_path)
    qc_config = config_dict.get("qc", {}) if isinstance(config_dict.get("qc", {}), dict) else {}
    min_frames = int(qc_config.get("min_frames", 20))
    segments = filter_and_summarize(segments, min_frames=min_frames)
    nan_inf_report(segments)

    subject_ids = sorted({s.subject_id for s in segments})
    n_mels = int(config_dict.get("n_mels", 40))
    smooth_win = int(config_dict.get("smooth_win", 9))

    roi_map: Dict[str, Sequence[int]] = {}
    roi_config = config_dict.get("roi", {}) if isinstance(config_dict.get("roi", {}), dict) else {}
    if roi_config.get("enabled", False):
        roi_map = _collect_roi_channels(segments, subject_ids, roi_config, n_mels=n_mels, smooth_win=smooth_win)

    offset_map: Dict[str, int] = {}
    offset_config = config_dict.get("offset", {}) if isinstance(config_dict.get("offset", {}), dict) else {}
    frame_hz = float(config_dict.get("frame_hz", 1 / 0.011))
    if offset_config.get("enabled", False):
        offset_map = _collect_offsets(
            segments,
            subject_ids,
            roi_map,
            offset_config,
            frame_hz=frame_hz,
            n_mels=n_mels,
            smooth_win=smooth_win,
        )

    trf_section = config_dict.get("trf", {}) if isinstance(config_dict.get("trf", {}), dict) else {}
    trf_config = TRFConfig(
        alpha=float(trf_section.get("alpha", 1.0)),
        n_pre=int(trf_section.get("n_pre_frames", trf_section.get("n_pre", 5))),
        n_post=int(trf_section.get("n_post_frames", trf_section.get("n_post", 10))),
    )
    n_splits = int(trf_section.get("n_splits", 5))

    results = run_trf_analysis_per_subject(
        segments,
        trf_config=trf_config,
        n_splits=n_splits,
        roi_map=roi_map,
        offset_map=offset_map,
        n_mels=n_mels,
        smooth_win=smooth_win,
    )

    output_path = _ensure_results_dir(config_dict)
    results.to_csv(output_path, index=False)
    print(f"Saved TRF results to {output_path}")
    if not results.empty:
        mean_r2 = results["mean_r2"].median()
        print(f"Median subject mean R^2: {mean_r2:.4f}")


if __name__ == "__main__":
    main()
