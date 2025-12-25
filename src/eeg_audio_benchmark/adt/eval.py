"""Evaluation utilities for ADT EEG→audio decoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def _flatten(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a = a.reshape(-1)
    b = b.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_segment_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return R² and correlation metrics for a single segment."""

    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        T = min(pred.shape[0], target.shape[0])
        pred = pred[:T]
        target = target[:T]

    r2 = float(r2_score(_flatten(target), _flatten(pred)))
    pred_r = _pearson(pred, target)

    shift = max(1, int(0.33 * target.shape[0]))
    rolled = np.roll(target, shift=shift, axis=0)
    null_r2 = float(r2_score(_flatten(rolled), _flatten(pred)))
    pred_r_null = _pearson(pred, rolled)
    return {
        "r2": r2,
        "pred_r": pred_r,
        "null_r2": null_r2,
        "pred_r_null": pred_r_null,
    }


def aggregate_subject_metrics(per_subject_segments: dict[str, list[dict[str, float]]]) -> pd.DataFrame:
    """Aggregate per-segment metrics into a per-subject summary DataFrame."""

    summaries = []
    for subject_id, seg_metrics in per_subject_segments.items():
        r2_vals = [m["r2"] for m in seg_metrics if np.isfinite(m.get("r2", np.nan))]
        null_r2_vals = [m["null_r2"] for m in seg_metrics if np.isfinite(m.get("null_r2", np.nan))]
        pred_r_vals = [m["pred_r"] for m in seg_metrics if np.isfinite(m.get("pred_r", np.nan))]
        pred_r_null_vals = [
            m["pred_r_null"] for m in seg_metrics if np.isfinite(m.get("pred_r_null", np.nan))
        ]
        summaries.append(
            {
                "subject_id": subject_id,
                "mean_r2": float(np.nanmean(r2_vals)) if r2_vals else np.nan,
                "null_mean_r2": float(np.nanmean(null_r2_vals)) if null_r2_vals else np.nan,
                "median_pred_r": float(np.nanmedian(pred_r_vals)) if pred_r_vals else np.nan,
                "median_pred_r_null": float(np.nanmedian(pred_r_null_vals)) if pred_r_null_vals else np.nan,
                "n_segments": len(seg_metrics),
            }
        )
    return pd.DataFrame(summaries)


__all__ = ["aggregate_subject_metrics", "compute_segment_metrics"]
