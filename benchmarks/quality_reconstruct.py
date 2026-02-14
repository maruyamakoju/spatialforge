#!/usr/bin/env python3
"""Quality-evaluation skeleton for reconstruction backends.

This is intentionally lightweight for PR-A:
- keeps legacy behavior unchanged
- provides reproducible JSON output for future geometry/GT metrics
- supports synthetic fixtures by default and optional real frame folders
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from _common import ensure_output_path, git_commit_short, system_info, utc_timestamp, write_json

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


def _load_sequence_frames(sequence_dir: Path) -> list[np.ndarray]:
    """Load RGB frames from a sequence folder (sorted by filename)."""
    from PIL import Image

    frame_paths = sorted(
        p for p in sequence_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    frames: list[np.ndarray] = []
    for fp in frame_paths:
        with Image.open(fp) as img:
            frames.append(np.asarray(img.convert("RGB"), dtype=np.uint8))
    return frames


def _synthetic_sequence(
    name: str,
    frame_count: int,
    width: int,
    height: int,
    seed: int,
) -> tuple[str, list[np.ndarray]]:
    rng = np.random.default_rng(seed + frame_count + width + height)
    frames: list[np.ndarray] = []
    yy, xx = np.mgrid[0:height, 0:width]
    grad_r = ((xx / max(1, width - 1)) * 255).astype(np.uint8)
    grad_g = ((yy / max(1, height - 1)) * 255).astype(np.uint8)

    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = grad_r
        frame[..., 1] = grad_g
        x0 = int((i * 9) % max(1, width - 48))
        y0 = int((i * 7) % max(1, height - 48))
        frame[y0:y0 + 36, x0:x0 + 36, 2] = 255
        noise = rng.integers(0, 8, size=(height, width, 3), dtype=np.uint8)
        frames.append(np.clip(frame + noise, 0, 255).astype(np.uint8))
    return name, frames


def _fake_depth(_self: Any, frame: np.ndarray, *args: Any, **kwargs: Any) -> DepthResult:
    h, w = frame.shape[:2]
    y = np.linspace(1.0, 3.5, h, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 0.3, w, dtype=np.float32)[None, :]
    depth = y + x
    return DepthResult(
        depth_map=depth,
        min_depth=float(depth.min()),
        max_depth=float(depth.max()),
        is_metric=True,
        confidence_mean=0.9,
        focal_length_px=float(max(h, w)),
        width=w,
        height=h,
        model_used="mock/depth",
        model_license="apache-2.0",
        processing_time_ms=1.0,
    )


def _fake_pose(_self: Any, frames: list[np.ndarray], output_pointcloud: bool = False) -> PoseEstimationResult:
    h, w = frames[0].shape[:2]
    focal = float(max(h, w) * 1.2)
    poses: list[CameraPoseResult] = []
    for i in range(len(frames)):
        poses.append(
            CameraPoseResult(
                frame_index=i,
                rotation=np.eye(3, dtype=np.float64),
                translation=np.array([0.04 * i, 0.0, 0.0], dtype=np.float64),
                fx=focal,
                fy=focal,
                cx=w / 2.0,
                cy=h / 2.0,
                width=w,
                height=h,
            )
        )
    return PoseEstimationResult(poses=poses, pointcloud=None, processing_time_ms=1.0)


def _sequence_metrics(name: str, frame_count: int, result, output_size_bytes: int) -> dict[str, Any]:
    bbox = result.bounding_box or [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    min_pt = np.asarray(bbox[0], dtype=np.float64)
    max_pt = np.asarray(bbox[1], dtype=np.float64)
    extents = np.maximum(max_pt - min_pt, 0.0)
    bbox_volume = float(extents[0] * extents[1] * extents[2])

    point_like = result.num_points or result.num_gaussians or result.num_vertices or 0
    pose_count = 0
    camera_path_length = 0.0
    if result.camera_poses_json:
        try:
            poses_data = json.loads(result.camera_poses_json)
            pose_count = len(poses_data)
            if pose_count >= 2:
                translations = [np.asarray(item["translation"], dtype=np.float64) for item in poses_data]
                camera_path_length = float(
                    sum(np.linalg.norm(translations[i + 1] - translations[i]) for i in range(len(translations) - 1))
                )
        except Exception:
            pose_count = 0
            camera_path_length = 0.0

    return {
        "sequence": name,
        "frame_count": frame_count,
        "requested_backend": result.requested_backend,
        "backend_used": result.backend_used,
        "output_format": result.output_format,
        "output_size_bytes": output_size_bytes,
        "processing_time_ms": round(float(result.processing_time_ms), 3),
        "point_like_count": int(point_like),
        "bbox_volume_m3_like": round(bbox_volume, 6),
        "camera_pose_count": pose_count,
        "camera_path_length_like": round(camera_path_length, 6),
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "error" not in r]
    if not ok:
        return {
            "num_sequences": 0,
            "num_success": 0,
            "num_failed": len(results),
        }

    processing = np.asarray([float(r["processing_time_ms"]) for r in ok], dtype=np.float64)
    points = np.asarray([float(r["point_like_count"]) for r in ok], dtype=np.float64)
    volume = np.asarray([float(r["bbox_volume_m3_like"]) for r in ok], dtype=np.float64)

    return {
        "num_sequences": len(results),
        "num_success": len(ok),
        "num_failed": len(results) - len(ok),
        "processing_time_ms": {
            "mean": round(float(np.mean(processing)), 3),
            "p50": round(float(np.percentile(processing, 50)), 3),
            "p95": round(float(np.percentile(processing, 95)), 3),
        },
        "point_like_count": {
            "mean": round(float(np.mean(points)), 3),
            "min": int(np.min(points)),
            "max": int(np.max(points)),
        },
        "bbox_volume_m3_like": {
            "mean": round(float(np.mean(volume)), 6),
        },
        "notes": [
            "PR-A skeleton metric set; GT-based geometry metrics are next PR.",
            "Use this JSON format as the stable contract for longitudinal comparisons.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SpatialForge reconstruct quality-evaluation skeleton")
    parser.add_argument("--dataset-dir", default=None, help="Optional dataset root with sequence folders")
    parser.add_argument("--quality", choices=["draft", "standard", "high"], default="standard")
    parser.add_argument("--output-format", choices=["pointcloud", "gaussian", "mesh"], default="pointcloud")
    parser.add_argument("--reconstruct-backend", choices=["legacy", "tsdf", "da3"], default="legacy")
    parser.add_argument("--depth-backend", choices=["hf", "da3"], default="da3")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--real-depth", action="store_true", help="Run real depth model instead of mock")
    parser.add_argument("--real-pose", action="store_true", help="Run real pose model instead of mock")
    parser.add_argument("--synthetic-frame-count", type=int, default=12)
    parser.add_argument("--synthetic-width", type=int, default=640)
    parser.add_argument("--synthetic-height", type=int, default=384)
    parser.add_argument("--output", default=None, help="Path to output JSON")
    args = parser.parse_args()

    mock_depth = not args.real_depth
    mock_pose = not args.real_pose
    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)

    manager = ModelManager(
        model_dir=Path(args.model_dir),
        device=device,
        dtype=dtype,
        research_mode=False,
        depth_backend=args.depth_backend,
    )
    engine = ReconstructEngine(manager, backend=args.reconstruct_backend)

    sequences: list[tuple[str, list[np.ndarray]]] = []
    if args.dataset_dir:
        root = Path(args.dataset_dir)
        if root.exists() and root.is_dir():
            for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                frames = _load_sequence_frames(seq_dir)
                if len(frames) >= 3:
                    sequences.append((seq_dir.name, frames))

    if not sequences:
        sequences.append(
            _synthetic_sequence(
                name="synthetic_baseline",
                frame_count=args.synthetic_frame_count,
                width=args.synthetic_width,
                height=args.synthetic_height,
                seed=args.seed,
            )
        )

    patchers = []
    if mock_depth:
        patchers.append(patch("spatialforge.inference.depth_engine.DepthEngine.estimate", new=_fake_depth))
    if mock_pose:
        patchers.append(patch("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", new=_fake_pose))
    for p in patchers:
        p.start()

    per_sequence: list[dict[str, Any]] = []
    try:
        for name, frames in sequences:
            try:
                with TemporaryDirectory(prefix="sf_reconstruct_quality_") as tmp:
                    result = engine.reconstruct(
                        frames=frames,
                        quality=args.quality,
                        output_format=args.output_format,
                        output_dir=Path(tmp),
                    )
                    output_size = result.output_path.stat().st_size if result.output_path.exists() else 0
                    per_sequence.append(
                        _sequence_metrics(
                            name=name,
                            frame_count=len(frames),
                            result=result,
                            output_size_bytes=output_size,
                        )
                    )
            except Exception as exc:
                per_sequence.append({"sequence": name, "frame_count": len(frames), "error": str(exc)})
    finally:
        for p in reversed(patchers):
            p.stop()
        manager.unload_all()

    payload = {
        "kind": "reconstruct_quality_eval",
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit_short(),
        "system": system_info(),
        "config": {
            "dataset_dir": args.dataset_dir,
            "quality": args.quality,
            "output_format": args.output_format,
            "reconstruct_backend": args.reconstruct_backend,
            "depth_backend": args.depth_backend,
            "device": device,
            "dtype": dtype,
            "seed": args.seed,
            "mock_depth": mock_depth,
            "mock_pose": mock_pose,
            "synthetic_frame_count": args.synthetic_frame_count,
            "synthetic_width": args.synthetic_width,
            "synthetic_height": args.synthetic_height,
        },
        "results": per_sequence,
        "summary": _aggregate(per_sequence),
    }

    output = ensure_output_path(args.output, prefix="reconstruct_quality")
    write_json(output, payload)
    print(f"Wrote quality-eval JSON: {output}")


if __name__ == "__main__":
    main()
