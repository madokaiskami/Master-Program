"""Visualization helpers for TRF results."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _infer_condition(path: Path, mapping: Mapping[str, str] | None) -> str:
    if mapping is None:
        return path.stem
    # Allow matching by full path, name, or stem
    key_candidates = [str(path), path.name, path.stem]
    for key in key_candidates:
        if key in mapping:
            return mapping[key]
    return path.stem


def load_trf_results(results_paths: Sequence[Path], condition_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Load and concatenate TRF result CSVs with a condition label."""

    frames = []
    for path in results_paths:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"TRF results not found: {resolved}")
        df = pd.read_csv(resolved)
        df = df.copy()
        df["condition"] = _infer_condition(resolved, condition_map)
        frames.append(df)
        logger.info("Loaded %d rows from %s as condition '%s'", len(df), resolved, df["condition"].iloc[0])
    if not frames:
        raise ValueError("No TRF results loaded")
    return pd.concat(frames, ignore_index=True)


def plot_trf_metric_distribution(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Plot distribution of a TRF metric across conditions."""

    required = {"subject_id", "condition", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(data=df, x="condition", y=metric, inner=None, cut=0, ax=ax)
    sns.stripplot(data=df, x="condition", y=metric, hue="subject_id", ax=ax, dodge=True, legend=False)
    ax.set_title(f"Distribution of {metric}")
    ax.set_ylabel(metric)
    ax.set_xlabel("Condition")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved metric distribution plot for %s to %s", metric, out_path)


def plot_subject_paired_scatter(df: pd.DataFrame, metric: str, cond_x: str, cond_y: str, out_path: Path) -> None:
    """Plot subject-wise paired metrics between two conditions."""

    def _filter(condition: str) -> pd.DataFrame:
        return df[df["condition"] == condition][["subject_id", metric]].rename(columns={metric: f"{metric}_{condition}"})

    df_x = _filter(cond_x)
    df_y = _filter(cond_y)
    merged = pd.merge(df_x, df_y, on="subject_id", how="inner")
    if merged.empty:
        raise ValueError("No overlapping subjects between conditions for paired scatter plot")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged[f"{metric}_{cond_x}"], merged[f"{metric}_{cond_y}"], alpha=0.8)
    lims = [
        min(merged[f"{metric}_{cond_x}"].min(), merged[f"{metric}_{cond_y}"].min()),
        max(merged[f"{metric}_{cond_x}"].max(), merged[f"{metric}_{cond_y}"].max()),
    ]
    ax.plot(lims, lims, "k--", label="y=x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"{cond_x} {metric}")
    ax.set_ylabel(f"{cond_y} {metric}")
    ax.set_title(f"Subject-wise {metric}: {cond_x} vs {cond_y}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved paired scatter for %s vs %s to %s", cond_x, cond_y, out_path)


def plot_band_ablation(
    df: pd.DataFrame,
    baseline_condition: str,
    ablated_conditions: Mapping[str, str],
    metric: str,
    out_path: Path,
) -> None:
    """Plot improvement from removing individual bands relative to a baseline."""

    required = {"subject_id", "condition", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    base_df = df[df["condition"] == baseline_condition][["subject_id", metric]].rename(columns={metric: "baseline"})
    if base_df.empty:
        raise ValueError(f"No rows found for baseline condition '{baseline_condition}'")

    rows = []
    for band, cond_label in ablated_conditions.items():
        ablated_df = df[df["condition"] == cond_label][["subject_id", metric]].rename(columns={metric: "ablated"})
        merged = pd.merge(base_df, ablated_df, on="subject_id", how="inner")
        if merged.empty:
            logger.warning("Skipping band %s: no overlapping subjects for condition %s", band, cond_label)
            continue
        delta = merged["baseline"] - merged["ablated"]
        rows.append({
            "band": band,
            "delta": delta.mean(),
            "sem": delta.std(ddof=1) / (len(delta) ** 0.5),
            "n_subjects": len(delta),
        })

    if not rows:
        raise ValueError("No ablation comparisons could be computed")

    df_plot = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_plot["band"], df_plot["delta"], yerr=df_plot["sem"], capsize=5)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel(f"Δ{metric} (baseline - ablated)")
    ax.set_title(f"Band ablation relative to {baseline_condition}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved band ablation plot to %s", out_path)


__all__ = [
    "load_trf_results",
    "plot_trf_metric_distribution",
    "plot_subject_paired_scatter",
    "plot_band_ablation",
]
