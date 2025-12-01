"""CLI to generate analysis figures for EEG-audio benchmarking."""
from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional

from eeg_audio_benchmark.analysis.spectral_eeg import (
    compute_band_power_per_subject,
    load_all_segments,
    plot_band_correlation_heatmap,
    plot_example_psd,
)
from eeg_audio_benchmark.analysis.trf_performance import (
    load_trf_results,
    plot_band_ablation,
    plot_subject_paired_scatter,
    plot_trf_metric_distribution,
)

logger = logging.getLogger(__name__)


def _parse_mapping(items: list[str] | None) -> Mapping[str, str]:
    mapping: dict[str, str] = {}
    if not items:
        return mapping
    for item in items:
        if "=" not in item:
            raise ValueError(f"Mapping items must be KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        mapping[key] = value
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/hf_eeg_audio/manifest_epochs.csv"),
        help="Path to the aligned epochs manifest",
    )
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate of EEG in Hz")
    parser.add_argument(
        "--trf-results",
        nargs="+",
        type=Path,
        required=True,
        help="One or more TRF results CSV files",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        help=(
            "Optional condition labels matching --trf-results, e.g.: "
            "--trf-results res_time.csv res_psd.csv res_time_psd.csv "
            "--conditions time_only psd_only time_psd"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory where figures will be saved",
    )
    parser.add_argument(
        "--example-subject",
        type=str,
        default=None,
        help="Subject ID to use for example PSD plot; defaults to the first in the manifest",
    )
    parser.add_argument(
        "--baseline-condition",
        type=str,
        default=None,
        help="Condition label for the full model used in band ablation plots",
    )
    parser.add_argument(
        "--ablated-conditions",
        nargs="*",
        default=None,
        help="Mappings of band=conditionlabel for ablation plots",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading EEG segments from %s", args.manifest)
    segments = load_all_segments(args.manifest)
    if not segments:
        raise RuntimeError("No segments loaded from manifest")

    subject_id = args.example_subject or segments[0].subject_id
    psd_path = output_dir / f"psd_example_{subject_id}.png"
    plot_example_psd(subject_id, segments, fs=args.fs, out_path=psd_path)

    band_power_df = compute_band_power_per_subject(segments, fs=args.fs)
    heatmap_path = output_dir / "band_power_correlation.png"
    plot_band_correlation_heatmap(band_power_df, out_path=heatmap_path)

    condition_map: Optional[Mapping[str, str]] = None
    if args.conditions is not None:
        if len(args.conditions) != len(args.trf_results):
            raise ValueError(
                "Number of --conditions entries must match number of --trf-results files"
            )
        condition_map = {str(path): condition for path, condition in zip(args.trf_results, args.conditions)}

    trf_df = load_trf_results(args.trf_results, condition_map)

    for metric in ["mean_r2", "median_pred_r"]:
        dist_path = output_dir / f"trf_{metric}_distribution.png"
        plot_trf_metric_distribution(trf_df, metric=metric, out_path=dist_path)

    conditions = sorted(trf_df["condition"].unique())
    for cond_x, cond_y in combinations(conditions, 2):
        paired_path = output_dir / f"paired_{cond_x}_vs_{cond_y}.png"
        plot_subject_paired_scatter(trf_df, metric="mean_r2", cond_x=cond_x, cond_y=cond_y, out_path=paired_path)

    if args.baseline_condition and args.ablated_conditions:
        ablated_map = _parse_mapping(args.ablated_conditions)
        ablation_path = output_dir / "band_ablation.png"
        plot_band_ablation(
            trf_df,
            baseline_condition=args.baseline_condition,
            ablated_conditions=ablated_map,
            metric="mean_r2",
            out_path=ablation_path,
        )


if __name__ == "__main__":
    main()
