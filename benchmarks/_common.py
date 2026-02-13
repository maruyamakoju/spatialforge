"""Shared utilities for local benchmarking scripts.

These scripts are designed for reproducible local runs (for example on RTX4090)
and are intentionally not part of CI execution.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit_short() -> str:
    """Best-effort git commit hash (short)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def percentile(values_ms: list[float], q: float) -> float:
    """Return percentile in milliseconds."""
    if not values_ms:
        return 0.0
    return float(np.percentile(np.asarray(values_ms, dtype=np.float64), q))


def ensure_output_path(path_str: str | None, prefix: str) -> Path:
    """Resolve and create output path for benchmark JSON."""
    if path_str:
        out = Path(path_str)
    else:
        stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path("benchmarks") / "results" / f"{prefix}_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def system_info() -> dict[str, Any]:
    """Collect runtime environment details."""
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "gpu_name": gpu_name,
    }

