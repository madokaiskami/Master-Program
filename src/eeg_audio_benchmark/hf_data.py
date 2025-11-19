"""Utilities for synchronizing and preparing the HuggingFace EEG/audio dataset."""

from __future__ import annotations

import logging
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - optional dependency
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # pragma: no cover - handled downstream
    HfApi = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

HF_DATASET_ID = "Aurelianous/eeg-audio"
LOCAL_DATA_ROOT = Path("data/hf_eeg_audio")
REVISION_FILE_NAME = ".hf_revision"
DEFAULT_PREPROC_CONFIG = Path("configs/preproc_on_hf.yaml")


def _require_hf_dependency() -> None:
    if HfApi is None or snapshot_download is None:  # pragma: no cover - import guard
        raise ImportError(
            "huggingface_hub is required for dataset synchronization. "
            "Install it via `pip install huggingface_hub`."
        )


def get_remote_revision(repo_id: str = HF_DATASET_ID) -> str:
    """Return the current remote revision sha for the HF dataset."""

    _require_hf_dependency()
    try:
        api = HfApi()
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
        return info.sha
    except Exception as exc:  # pragma: no cover - network/IO
        msg = f"Failed to query remote revision for {repo_id}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc


def _revision_file(root: Path) -> Path:
    return root / REVISION_FILE_NAME


def get_local_revision(root: Path = LOCAL_DATA_ROOT) -> Optional[str]:
    """Return the locally cached revision sha if present."""

    revision_path = _revision_file(root)
    if not revision_path.exists():
        return None
    try:
        return revision_path.read_text(encoding="utf-8").strip() or None
    except Exception as exc:  # pragma: no cover - unlikely
        logger.warning("Failed to read local revision file %s: %s", revision_path, exc)
        return None


def write_local_revision(root: Path, revision: str) -> None:
    """Persist the provided revision sha alongside the dataset."""

    revision_path = _revision_file(root)
    revision_path.parent.mkdir(parents=True, exist_ok=True)
    revision_path.write_text(revision, encoding="utf-8")


def ensure_local_dataset(
    repo_id: str = HF_DATASET_ID,
    local_root: Path = LOCAL_DATA_ROOT,
    force: bool = False,
) -> Path:
    """Ensure that the HF dataset is available locally and up to date."""

    _require_hf_dependency()
    remote_revision = get_remote_revision(repo_id)
    local_revision = get_local_revision(local_root)

    if not force and local_revision == remote_revision and local_root.exists():
        logger.info("Local HF dataset is up-to-date at %s", local_root)
        return local_root

    logger.info(
        "Fetching HF dataset %s into %s (force=%s)", repo_id, local_root, force
    )
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_root),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    except Exception as exc:  # pragma: no cover - network/IO
        msg = f"Failed to download dataset {repo_id}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    write_local_revision(local_root, remote_revision)
    return local_root


def has_derivatives(root: Path) -> bool:
    """Return True if aligned derivatives appear to be present under ``root``."""

    eeg_dir = root / "derivatives" / "aligned" / "eeg"
    audio_dir = root / "derivatives" / "aligned" / "audio"
    if not eeg_dir.exists() or not audio_dir.exists():
        return False
    try:
        next(eeg_dir.glob("*_EEG_aligned.npy"))
        next(audio_dir.glob("*_Sound_aligned.npy"))
        return True
    except StopIteration:
        return False


def _run_preprocessing(config_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "eeg_audio_benchmark.bin.run_preprocessing",
        "--config",
        str(config_path),
    ]
    logger.info("Running preprocessing pipeline: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:  # pragma: no cover - subprocess failure
        logger.error("Preprocessing failed:\nSTDOUT:%s\nSTDERR:%s", result.stdout, result.stderr)
        raise RuntimeError("Preprocessing pipeline failed. See logs for details.")
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.debug(result.stderr.strip())


def prepare_derivatives_if_needed(
    root: Path = LOCAL_DATA_ROOT,
    preproc_config_path: Path | None = None,
    force: bool = False,
) -> None:
    """Ensure aligned derivatives exist under ``root``; run preprocessing if needed."""

    if not force and has_derivatives(root):
        logger.info("Aligned derivatives already present under %s", root)
        return

    config_path = Path(preproc_config_path) if preproc_config_path else DEFAULT_PREPROC_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(
            "Preprocessing config not found. Provide `preproc_config_path` "
            f"or create the default config at {config_path}."
        )

    logger.info("Generating derivatives using config %s", config_path)
    try:
        _run_preprocessing(config_path)
    except Exception as exc:  # pragma: no cover - subprocess failure
        msg = f"Failed to run preprocessing pipeline: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    if not has_derivatives(root):  # pragma: no cover - validation
        raise RuntimeError(
            "Preprocessing finished but aligned derivatives were not found."
        )


def sanity_check_derivatives(root: Path = LOCAL_DATA_ROOT) -> None:
    """Perform a light-weight sanity check that HF derivatives exist."""

    epochs_dir = root / "derivatives" / "epochs"
    aligned_eeg_dir = root / "derivatives" / "aligned" / "eeg"
    aligned_audio_dir = root / "derivatives" / "aligned" / "audio"
    manifest_path = root / "manifest_epochs.csv"

    if not epochs_dir.exists() or not any(epochs_dir.glob("*.npy")):
        raise RuntimeError(f"Missing or empty epochs directory: {epochs_dir}")
    if not aligned_eeg_dir.exists() or not any(aligned_eeg_dir.glob("*_EEG_aligned.npy")):
        raise RuntimeError(f"Missing aligned EEG outputs under {aligned_eeg_dir}")
    if not aligned_audio_dir.exists() or not any(aligned_audio_dir.glob("*_Sound_aligned.npy")):
        raise RuntimeError(f"Missing aligned audio outputs under {aligned_audio_dir}")
    if not manifest_path.exists():
        raise RuntimeError(f"Aligned manifest not found: {manifest_path}")


def prepare_data_for_training(
    preproc_config_path: Path | None = None,
    force_preproc: bool = False,
    force_download: bool = False,
) -> Path:
    """Ensure HF data (and derivatives) are available locally for training."""

    root = ensure_local_dataset(force=force_download)
    prepare_derivatives_if_needed(
        root=root,
        preproc_config_path=preproc_config_path,
        force=force_preproc,
    )
    return root


def run_preprocessing_sanity_check(
    preproc_config_path: Path | None = DEFAULT_PREPROC_CONFIG,
    force_download: bool = False,
) -> None:
    """Run preprocessing (if needed) and assert core HF derivatives exist."""

    root = prepare_data_for_training(
        preproc_config_path=preproc_config_path,
        force_preproc=True,
        force_download=force_download,
    )
    sanity_check_derivatives(root)


__all__ = [
    "HF_DATASET_ID",
    "LOCAL_DATA_ROOT",
    "get_remote_revision",
    "get_local_revision",
    "write_local_revision",
    "ensure_local_dataset",
    "has_derivatives",
    "prepare_derivatives_if_needed",
    "prepare_data_for_training",
    "sanity_check_derivatives",
    "run_preprocessing_sanity_check",
]
