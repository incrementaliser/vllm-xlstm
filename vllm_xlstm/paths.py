"""Repository and artifact path helpers for reproducible runs."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root (parent of the ``vllm_xlstm`` package)."""
    return Path(__file__).resolve().parent.parent


def artifacts_root() -> Path:
    """Return ``<repo>/artifacts``."""
    return repo_root() / "artifacts"


def ensure_artifact_dirs() -> dict[str, Path]:
    """
    Create standard artifact subdirectories if missing.

    Returns:
        Mapping of logical name to path: logs, runs.
    """
    root = artifacts_root()
    mapping = {
        "logs": root / "logs",
        "runs": root / "runs",
        "slurm": root / "logs" / "slurm",
    }
    for path in mapping.values():
        path.mkdir(parents=True, exist_ok=True)
    return mapping


def timestamp() -> str:
    """UTC ``YYYYMMDD_HHMMSS`` stamp for ``DATE_TIME_EXPERIMENT`` run names."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sanitize_experiment(name: str) -> str:
    """Return a filesystem-safe experiment slug (word chars, ``.``, ``-``)."""
    cleaned = re.sub(r"[^\w.\-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("._-")
    return cleaned or "run"


def new_run_dir(experiment: str = "vllm-speed") -> Path:
    """
    Create ``artifacts/runs/{YYYYMMDD}_{HHMMSS}_{experiment}/``.

    Also updates ``artifacts/runs/LATEST`` as a convenience symlink to this
    directory. The run identity is the ``DATE_TIME_EXPERIMENT`` folder name.

    Args:
        experiment: Short slug appended after the UTC date and time.

    Returns:
        Absolute path to the new run directory.
    """
    ensure_artifact_dirs()
    slug = sanitize_experiment(experiment)
    run = artifacts_root() / "runs" / f"{timestamp()}_{slug}"
    for sub in ("logs", "results", "plots"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    latest = artifacts_root() / "runs" / "LATEST"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    try:
        latest.symlink_to(run.resolve())
    except OSError:
        latest.write_text(str(run.resolve()) + "\n", encoding="utf-8")
    return run.resolve()


def default_hf_dir() -> Path | None:
    """
    Return a local Hub snapshot for ``NX-AI/xLSTM-7b`` when the cache has one.

    Does not hard-code a cluster path or snapshot hash. Returns ``None`` when
    the Hugging Face cache has not downloaded the model yet.
    """
    snaps = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--NX-AI--xLSTM-7b"
        / "snapshots"
    )
    if not snaps.is_dir():
        return None
    dirs = sorted(path for path in snaps.iterdir() if path.is_dir())
    return dirs[-1] if dirs else None


def speed_prompts_path() -> Path:
    """Path to the speed-plan prompt JSON shipped with the package."""
    packaged = Path(__file__).resolve().parent / "data" / "speed_prompts.json"
    if packaged.is_file():
        return packaged
    return repo_root() / "data" / "speed_prompts.json"
