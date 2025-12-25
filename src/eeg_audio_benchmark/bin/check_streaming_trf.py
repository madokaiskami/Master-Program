"""Quick streaming sanity check for TRF pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from eeg_audio_benchmark.config import load_config
from eeg_audio_benchmark.hf_data import LOCAL_DATA_ROOT
from eeg_audio_benchmark.trf.data import load_segments_from_hf_manifest
from eeg_audio_benchmark.trf.eval import eval_subject_trf_envelope
from eeg_audio_benchmark.trf.models import TRFConfig

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny streaming TRF check.")
    parser.add_argument("--config", required=True, help="Path to TRF YAML config")
    parser.add_argument("--max-segments", type=int, default=2, help="Limit segments for smoke test")
    parser.add_argument("--subject", type=str, default=None, help="Subject id to evaluate")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_config(args.config)
    trf_section = cfg.get("trf", {}) if isinstance(cfg.get("trf", {}), dict) else {}
    manifest = trf_section.get("manifest_path")
    dataset_root = Path(cfg.get("dataset_root", LOCAL_DATA_ROOT))
    manifest_path = Path(manifest) if manifest else dataset_root / "manifest_epochs.csv"

    segments = load_segments_from_hf_manifest(dataset_root, manifest_path=manifest_path)
    subject_ids = sorted({s.subject_id for s in segments})
    if not subject_ids:
        raise RuntimeError("No subjects found in manifest")
    subject = args.subject or subject_ids[0]

    trf_cfg = TRFConfig(
        ridge_alpha=float(trf_section.get("ridge_alpha", trf_section.get("alpha", 1.0))),
        n_pre=int(trf_section.get("n_pre_frames", trf_section.get("n_pre", 5))),
        n_post=int(trf_section.get("n_post_frames", trf_section.get("n_post", 10))),
        audio_representation=str(trf_section.get("audio_representation", "handcrafted")),
        mel_n_bands=int(trf_section.get("mel_n_bands", cfg.get("n_mels", 40))),
        mel_mode=str(trf_section.get("mel_mode", "envelope")),
        mel_smooth_win=int(trf_section.get("mel_smooth_win", cfg.get("smooth_win", 9))),
        acoustic_features=list(trf_section.get("acoustic_features", ["broadband_env"])),
        eeg_highpass_win=int(trf_section.get("eeg_highpass_win", 15)),
        eeg_zscore_mode=str(trf_section.get("eeg_zscore_mode", "per_segment_channel")),
        streaming=True,
        data_chunk_rows=int(trf_section.get("data_chunk_rows", trf_section.get("chunk_rows", 0)) or 0) or None,
        solver=str(trf_section.get("solver", "ridge_sklearn")),
        feature_reduce_method=str(trf_section.get("feature_reduce_method", "none")),
        feature_reduce_out_dim=trf_section.get("feature_reduce_out_dim"),
        random_state=trf_section.get("random_state"),
    )

    logger.info("Running streaming TRF check on subject=%s (max_segments=%s)", subject, args.max_segments)
    res = eval_subject_trf_envelope(
        segments,
        subject_id=subject,
        trf_config=trf_cfg,
        n_splits=1,
        max_segments=args.max_segments,
    )
    print(res)


if __name__ == "__main__":
    main()
