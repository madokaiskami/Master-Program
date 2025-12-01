# EEG Audio Benchmark

A complete pipeline for encoding and decoding relationships between continuous EEG and natural audio. The project ships end-to-end tooling: preprocessing CBYT EEG/audio pairs, running Temporal Response Function (TRF) baselines, and producing publication-ready analysis figures.

## Repository layout

```
Master-Program/
├── src/eeg_audio_benchmark/          # Core Python package
│   ├── preprocessing/                # EEG/audio preprocessing steps
│   ├── trf/                          # TRF data loader, ROI utilities, models, offsets
│   ├── analysis/                     # Figure generation utilities (PSD + TRF results)
│   ├── bin/                          # CLI entrypoints (preprocessing, TRF)
│   ├── datasets.py / representations.py  # Feature registration and dataset assembly
│   ├── experiment.py                 # Config-driven experiment runner
│   └── models/                       # Pluggable regression models
├── configs/                          # Example YAML configs for preprocessing & TRF
├── data/                             # Expected location for HF downloads and derivatives
└── results/                          # Outputs: TRF CSVs and figures
```

## Preprocessing pipeline

The preprocessing modules revolve around the HuggingFace dataset `Aurelianous/eeg-audio` and generate aligned EEG/audio derivatives under `data/hf_eeg_audio/derivatives/`.

1. **Slice EEG into epochs** – `preprocessing/eeg_epochs.py`
   - Input: `manifest_raw_runs.csv` + `raw/eeg/*.npz` + `raw/events/*.csv`
   - Output: epoch `.npy` files and `derivatives/epoch_manifest.csv`
2. **Artifact scoring** – `preprocessing/artifacts.py`
   - Computes composite artifact scores and writes a CSV with `Is_Artifact`
3. **Audio feature extraction** – `preprocessing/audio_features.py`
   - Uses Librosa to compute features per WAV and records a feature manifest
4. **Alignment** – `preprocessing/alignment.py`
   - Aligns clean EEG epochs with audio features on a shared grid, producing `manifest_epochs.csv`

All steps accept dataclass configs and can be chained via `python -m eeg_audio_benchmark.bin.sync_hf_data --preproc-config <config.yaml>`.

## TRF baseline pipeline

* **Data loading** – `trf/data.py` loads aligned pairs into `Segment` objects and provides helpers like `filter_and_summarize`.
* **Models/configs** – `trf/models.py` defines `TRFConfig` and `TRFEncoder` plus feature lag utilities.
* **ROI and offsets** – `trf/roi.py` for channel subsets; `trf/offset.py` for offset searches.
* **Running TRF** – The CLI `python -m eeg_audio_benchmark.bin.run_trf --config <yaml>` trains subject-level TRF encoders and writes CSVs to `results/trf/trf_results_*.csv` containing columns such as `subject_id`, `mean_r2`, `median_pred_r`, `r2_per_channel`, `offset_frames`, `roi_channels`, and null-baseline metrics.

## Analysis utilities (weeks 3–7)

The new `eeg_audio_benchmark.analysis` package adds reusable plotting helpers and a one-shot CLI to generate figures.

### Spectral EEG (PSD)

* **Loading segments:** `analysis.spectral_eeg.load_all_segments(manifest_path)` wraps the TRF loader for aligned epochs.
* **PSD & band power:**
  - `compute_psd_for_segment(eeg, fs, nperseg=512)` returns Welch PSD per channel.
  - `compute_band_power_per_subject(segments, fs, bands=DEFAULT_BANDS)` aggregates per-subject/channel power across delta–gamma bands.
* **Plots:**
  - `plot_example_psd(subject_id, segments, fs, out_path)` – mean PSD with band highlights.
  - `plot_band_correlation_heatmap(df_band_power, out_path)` – correlation heatmap of band powers averaged per subject.

### TRF performance comparisons

* **Loading CSVs:** `analysis.trf_performance.load_trf_results(paths, condition_map=None)` concatenates TRF result files and tags each with a condition label (derived from filenames or an explicit mapping).
* **Distributions:** `plot_trf_metric_distribution(df, metric, out_path)` draws violin + strip plots for metrics like `mean_r2` or `median_pred_r`.
* **Subject-wise gains:** `plot_subject_paired_scatter(df, metric, cond_x, cond_y, out_path)` compares per-subject performance between two conditions.
* **Band ablations:** `plot_band_ablation(df, baseline_condition, ablated_conditions, metric, out_path)` visualizes Δmetric when individual bands are removed.

### One-shot figure generation CLI

Run all spectral and TRF plots in one command:

```bash
PYTHONPATH=src python -m eeg_audio_benchmark.analysis.cli_generate_figures \
  --manifest data/hf_eeg_audio/manifest_epochs.csv \
  --fs 250 \
  --trf-results results/trf/trf_results_time_only.csv results/trf/trf_results_psd_only.csv results/trf/trf_results_time_psd.csv \
  --output-dir results/figures
```

Optional flags:

* `--condition-labels trf_results_time_only=time trf_results_psd_only=psd ...` to override condition names
* `--example-subject <ID>` to choose the subject for the PSD example
* `--baseline-condition` and `--ablated-conditions band1=condA band2=condB ...` to enable band ablation plots

The CLI saves PSD examples, band-power heatmaps, TRF metric distributions, paired subject plots, and optional ablation bars under `results/figures/`.

## Getting started

1. Install in editable mode: `pip install -e .` (requires numpy, pandas, scipy, matplotlib, seaborn, librosa, sklearn, torch, etc.)
2. Download/prepare data via the preprocessing steps or the combined sync CLI.
3. Train TRF baselines with `python -m eeg_audio_benchmark.bin.run_trf --config <configs/trf_example.yaml>`.
4. Generate figures with the analysis CLI or import functions from `eeg_audio_benchmark.analysis` in notebooks.

## Contributing

The codebase favors typed, functional utilities with clear logging (`logging.getLogger(__name__)`). When extending the pipeline or analysis modules, keep Matplotlib figures closed after saving and prefer `Path`/`pathlib` for filesystem work.
