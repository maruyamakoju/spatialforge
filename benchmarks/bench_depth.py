#!/usr/bin/env python3
"""Benchmark /depth inference with reproducible JSON outputs.

Example:
    python benchmarks/bench_depth.py --backend da3 --models small,base,large \
        --resolutions 512,768,1024 --runs 20 --warmup 5
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _common import (
    ensure_output_path,
    git_commit_short,
    percentile,
    system_info,
    utc_timestamp,
    write_json,
)

from spatialforge.inference.depth_engine import DepthEngine
from spatialforge.inference.model_manager import ModelManager


def _resolve_device(device_arg: str) -> str:
    if device_arg in {"cpu", "cuda"}:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype_arg: str, device: str) -> str:
    if device == "cpu":
        return "float32"
    if dtype_arg == "auto":
        return "float16"
    return dtype_arg


def _parse_csv_ints(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def _parse_csv_str(raw: str) -> list[str]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(token)
    if not vals:
        raise ValueError("Expected at least one model name")
    return vals


def _benchmark_case(
    engine: DepthEngine,
    *,
    model: str,
    resolution: int,
    runs: int,
    warmup: int,
    seed: int,
) -> dict[str, Any]:
    h = resolution
    w = int(round(resolution * 16 / 9))
    rng = np.random.default_rng(seed + resolution)
    image_rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    for _ in range(warmup):
        engine.estimate(image_rgb=image_rgb, model_size=model)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_ms: list[float] = []
    engine_ms: list[float] = []
    sample_license = ""
    sample_repo = ""

    for _ in range(runs):
        t0 = time.perf_counter()
        result = engine.estimate(image_rgb=image_rgb, model_size=model)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000.0
        wall_ms.append(float(elapsed))
        engine_ms.append(float(result.processing_time_ms))
        sample_license = result.model_license
        sample_repo = result.model_used

    record: dict[str, Any] = {
        "model": model,
        "resolution": {"width": w, "height": h},
        "runs": runs,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(float(statistics.mean(wall_ms)), 3),
            "median": round(float(statistics.median(wall_ms)), 3),
            "p50": round(percentile(wall_ms, 50), 3),
            "p95": round(percentile(wall_ms, 95), 3),
            "min": round(float(min(wall_ms)), 3),
            "max": round(float(max(wall_ms)), 3),
            "std": round(float(statistics.pstdev(wall_ms)), 3),
        },
        "engine_processing_ms": {
            "mean": round(float(statistics.mean(engine_ms)), 3),
            "p95": round(percentile(engine_ms, 95), 3),
        },
        "fps": round(1000.0 / float(statistics.mean(wall_ms)), 3),
        "model_repo": sample_repo,
        "model_license": sample_license,
    }

    if torch.cuda.is_available():
        record["peak_vram_gb"] = round(float(torch.cuda.max_memory_allocated(0) / 1e9), 3)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="SpatialForge depth benchmark")
    parser.add_argument("--backend", choices=["hf", "da3"], default="da3")
    parser.add_argument("--models", default="small,base,large")
    parser.add_argument("--resolutions", default="512,768,1024")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Path to output JSON")
    args = parser.parse_args()

    models = _parse_csv_str(args.models)
    resolutions = _parse_csv_ints(args.resolutions)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    manager = ModelManager(
        model_dir=Path(args.model_dir),
        device=device,
        dtype=dtype,
        research_mode=False,
        depth_backend=args.backend,
    )
    engine = DepthEngine(manager)

    records: list[dict[str, Any]] = []
    for model in models:
        for res in resolutions:
            print(f"[depth] model={model} res={res} backend={args.backend} device={device} dtype={dtype}")
            try:
                rec = _benchmark_case(
                    engine,
                    model=model,
                    resolution=res,
                    runs=args.runs,
                    warmup=args.warmup,
                    seed=args.seed,
                )
                records.append(rec)
                print(
                    f"  p50={rec['latency_ms']['p50']}ms p95={rec['latency_ms']['p95']}ms "
                    f"fps={rec['fps']}"
                )
            except Exception as exc:
                err = {
                    "model": model,
                    "resolution": {"height": res, "width": int(round(res * 16 / 9))},
                    "error": str(exc),
                }
                records.append(err)
                print(f"  ERROR: {exc}")

    manager.unload_all()

    output = ensure_output_path(args.output, prefix="depth_benchmark")
    payload = {
        "kind": "depth_benchmark",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_short(),
        "system": system_info(),
        "config": {
            "backend": args.backend,
            "models": models,
            "resolutions": resolutions,
            "runs": args.runs,
            "warmup": args.warmup,
            "device": device,
            "dtype": dtype,
            "seed": args.seed,
        },
        "results": records,
    }
    write_json(output, payload)
    print(f"Wrote benchmark JSON: {output}")


if __name__ == "__main__":
    main()
