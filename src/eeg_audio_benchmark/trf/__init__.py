"""TRF-style EEG/audio analysis utilities."""

from .data import Segment, filter_and_summarize, load_segments_from_hf_manifest, nan_inf_report
from .eval import eval_subject_trf_envelope, run_trf_analysis_per_subject
from .features import build_lagged_features, causal_moving_average, envelope_for_segments, envelope_from_matrix, envelope_from_sound_matrix
from .models import TRFConfig, TRFEncoder
from .offset import pick_best_global_offset, score_offset_for_roi, shift_sound_forward
from .roi import compute_voiced_mask, select_roi_channels_for_subject

__all__ = [
    "Segment",
    "filter_and_summarize",
    "load_segments_from_hf_manifest",
    "nan_inf_report",
    "eval_subject_trf_envelope",
    "run_trf_analysis_per_subject",
    "build_lagged_features",
    "causal_moving_average",
    "envelope_for_segments",
    "envelope_from_matrix",
    "envelope_from_sound_matrix",
    "TRFConfig",
    "TRFEncoder",
    "pick_best_global_offset",
    "score_offset_for_roi",
    "shift_sound_forward",
    "compute_voiced_mask",
    "select_roi_channels_for_subject",
]
