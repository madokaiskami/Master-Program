"""Analysis utilities for EEG-audio experiments."""

from .spectral_eeg import (
    BandDef,
    DEFAULT_BANDS,
    compute_band_power_per_subject,
    compute_psd_for_segment,
    load_all_segments,
    plot_band_correlation_heatmap,
    plot_example_psd,
)
from .trf_performance import (
    load_trf_results,
    plot_band_ablation,
    plot_subject_paired_scatter,
    plot_trf_metric_distribution,
)

__all__ = [
    "BandDef",
    "DEFAULT_BANDS",
    "compute_band_power_per_subject",
    "compute_psd_for_segment",
    "load_all_segments",
    "plot_band_correlation_heatmap",
    "plot_example_psd",
    "load_trf_results",
    "plot_band_ablation",
    "plot_subject_paired_scatter",
    "plot_trf_metric_distribution",
]
