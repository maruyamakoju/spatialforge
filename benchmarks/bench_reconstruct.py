#!/usr/bin/env python3
"""Benchmark reconstruct pipeline and export JSON metrics.

This script can run with real models or with mocked depth/pose stages.
For RTX4090 research reporting, run without mock flags.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

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

from spatialforge.inference.depth_engine import DepthResult
from spatialforge.inference.model_manager import ModelManager
from spatialforge.inference.pose_engine import CameraPoseResult, PoseEstimationResult
from spatialforge.inference.reconstruct_engine import ReconstructEngine


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


def _synthetic_frames(frame_count: int, width: int, height: int, seed: int) -> list[np.ndarray]:
    """Generate deterministic frame sequence with trackable motion."""
    rng = np.random.default_rng(seed + frame_count + width + height)
    frames: list[np.ndarray] = []
    base = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    grad = ((xx / max(1, width - 1)) * 255).astype(np.uint8)
    base[..., 1] = grad
    base[..., 2] = ((yy / max(1, height - 1)) * 255).astype(np.uint8)

    for i in range(frame_count):
        frame = base.copy()
        x0 = int((i * 7) % max(1, width - 40))
        y0 = int((i * 5) % max(1, height - 40))
        frame[y0:y0 + 32, x0:x0 + 32, 0] = 255
        noise = rng.integers(0, 8, size=(height, width, 3), dtype=np.uint8)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _fake_depth(_self: Any, frame: np.ndarray, *args: Any, **kwargs: Any) -> DepthResult:
    h, w = frame.shape[:2]
    y = np.linspace(1.0, 3.0, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 0.25, w, dtype=np.float32)[None, :]
    depth = y + x
    return DepthResult(
        depth_map=depth,
        min_depth=float(depth.min()),
        max_depth=float(depth.max()),
        is_metric=True,
        confidence_mean=0.85,
        focal_length_px=float(max(h, w)),
        width=w,
        height=h,
        model_used="mock/depth",
        model_license="apache-2.0",
        processing_time_ms=2.0,
    )


def _fake_pose(_self: Any, frames: list[np.ndarray], output_pointcloud: bool = False) -> PoseEstimationResult:
    h, w = frames[0].shape[:2]
    focal = float(max(h, w) * 1.1)
    poses: list[CameraPoseResult] = []
    for i in range(len(frames)):
        poses.append(
            CameraPoseResult(
                frame_index=i,
                rotation=np.eye(3, dtype=np.float64),
                translation=np.array([0.03 * i, 0.0, 0.0], dtype=np.float64),
                fx=focal,
                fy=focal,
                cx=w / 2.0,
                cy=h / 2.0,
                width=w,
                height=h,
            )
        )
    return PoseEstimationResult(poses=poses, pointcloud=None, processing_time_ms=1.0)


def _benchmark_case(
    engine: ReconstructEngine,
    *,
    frame_count: int,
    width: int,
    height: int,
    runs: int,
    warmup: int,
    quality: str,
    output_format: str,
    seed: int,
    mock_depth: bool,
    mock_pose: bool,
) -> dict[str, Any]:
    frames = _synthetic_frames(frame_count=frame_count, width=width, height=height, seed=seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    patchers = []
    if mock_depth:
        patchers.append(patch("spatialforge.inference.depth_engine.DepthEngine.estimate", new=_fake_depth))
    if mock_pose:
        patchers.append(patch("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", new=_fake_pose))

    for p in patchers:
        p.start()

    try:
        for _ in range(warmup):
            with TemporaryDirectory(prefix="sf_recon_bench_warmup_") as tmp:
                engine.reconstruct(
                    frames=frames,
                    quality=quality,
                    output_format=output_format,
                    output_dir=Path(tmp),
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_ms: list[float] = []
        engine_ms: list[float] = []
        sample_points: int | None = None

        for _ in range(runs):
            with TemporaryDirectory(prefix="sf_recon_bench_run_") as tmp:
                t0 = time.perf_counter()
                result = engine.reconstruct(
                    frames=frames,
                    quality=quality,
                    output_format=output_format,
                    output_dir=Path(tmp),
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - t0) * 1000.0
                wall_ms.append(float(elapsed))
                engine_ms.append(float(result.processing_time_ms))
                sample_points = result.num_points or result.num_gaussians or result.num_vertices

        record: dict[str, Any] = {
            "frame_count": frame_count,
            "resolution": {"width": width, "height": height},
            "quality": quality,
            "output_format": output_format,
            "runs": runs,
            "warmup": warmup,
            "mock_depth": mock_depth,
            "mock_pose": mock_pose,
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
            "sample_output_points": sample_points,
        }
        if torch.cuda.is_available():
            record["peak_vram_gb"] = round(float(torch.cuda.max_memory_allocated(0) / 1e9), 3)
        return record
    finally:
        for p in reversed(patchers):
            p.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="SpatialForge reconstruct benchmark")
    parser.add_argument("--frame-counts", default="30,60,120")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--quality", choices=["draft", "standard", "high"], default="standard")
    parser.add_argument("--output-format", choices=["pointcloud", "gaussian", "mesh"], default="pointcloud")
    parser.add_argument(
        "--reconstruct-backend",
        choices=["legacy", "tsdf", "da3"],
        default="legacy",
        help="Reconstruction backend selector (currently legacy is implemented).",
    )
    parser.add_argument("--backend", choices=["hf", "da3"], default="da3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mock-depth", action="store_true", help="Mock depth estimation stage")
    parser.add_argument("--mock-pose", action="store_true", help="Mock pose estimation stage")
    parser.add_argument("--output", default=None, help="Path to output JSON")
    args = parser.parse_args()

    frame_counts = _parse_csv_ints(args.frame_counts)
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    manager = ModelManager(
        model_dir=Path(args.model_dir),
        device=device,
        dtype=dtype,
        research_mode=False,
        depth_backend=args.backend,
    )
    engine = ReconstructEngine(manager, backend=args.reconstruct_backend)

    records: list[dict[str, Any]] = []
    for frame_count in frame_counts:
        print(
            f"[reconstruct] frames={frame_count} res={args.width}x{args.height} "
            f"depth_backend={args.backend} reconstruct_backend={args.reconstruct_backend} "
            f"device={device} dtype={dtype} "
            f"mock_depth={args.mock_depth} mock_pose={args.mock_pose}"
        )
        try:
            rec = _benchmark_case(
                engine,
                frame_count=frame_count,
                width=args.width,
                height=args.height,
                runs=args.runs,
                warmup=args.warmup,
                quality=args.quality,
                output_format=args.output_format,
                seed=args.seed,
                mock_depth=args.mock_depth,
                mock_pose=args.mock_pose,
            )
            records.append(rec)
            print(
                f"  p50={rec['latency_ms']['p50']}ms p95={rec['latency_ms']['p95']}ms "
                f"points={rec.get('sample_output_points')}"
            )
        except Exception as exc:
            err = {
                "frame_count": frame_count,
                "resolution": {"width": args.width, "height": args.height},
                "error": str(exc),
            }
            records.append(err)
            print(f"  ERROR: {exc}")

    manager.unload_all()

    output = ensure_output_path(args.output, prefix="reconstruct_benchmark")
    payload = {
        "kind": "reconstruct_benchmark",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_short(),
        "system": system_info(),
        "config": {
            "frame_counts": frame_counts,
            "width": args.width,
            "height": args.height,
            "runs": args.runs,
            "warmup": args.warmup,
            "quality": args.quality,
            "output_format": args.output_format,
            "backend": args.backend,
            "reconstruct_backend": args.reconstruct_backend,
            "device": device,
            "dtype": dtype,
            "seed": args.seed,
            "mock_depth": args.mock_depth,
            "mock_pose": args.mock_pose,
        },
        "results": records,
    }
    write_json(output, payload)
    print(f"Wrote benchmark JSON: {output}")


if __name__ == "__main__":
    main()
