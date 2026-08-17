"""JSON and logging helpers for CLI entry points."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


def write_json(path: Path, payload: Any) -> Path:
    """
    Write ``payload`` as indented JSON, creating parent directories.

    Args:
        path: Destination file.
        payload: JSON-serialisable object.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


class _Tee:
    """Write to both the original stream and a log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        """Write to both streams."""
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush both streams."""
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        """Report TTY status of the primary stream (needed by transformers)."""
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        """Delegate fileno to the primary stream when available."""
        return self.primary.fileno()

    @property
    def encoding(self) -> str | None:
        """Encoding of the primary stream."""
        return getattr(self.primary, "encoding", None)

    def __getattr__(self, name: str) -> Any:
        """Proxy remaining file attributes to the primary stream."""
        return getattr(self.primary, name)


@contextmanager
def tee_run_log(run_dir: Path, filename: str = "run.log") -> Iterator[Path]:
    """
    Tee stdout/stderr into ``run_dir/logs/<filename>``.

    Args:
        run_dir: ``DATE_TIME_EXPERIMENT`` directory from ``new_run_dir``.
        filename: Log file name under ``logs/``.

    Yields:
        Path to the log file.
    """
    path = run_dir / "logs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(old_out, fh)  # type: ignore[assignment]
        sys.stderr = _Tee(old_err, fh)  # type: ignore[assignment]
        try:
            print(f"[vllm-xlstm] run logging to {path}")
            yield path
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
