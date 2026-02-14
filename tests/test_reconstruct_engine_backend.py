"""ReconstructEngine backend selection tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
import pytest

from spatialforge.inference.depth_engine import DepthResult
from spatialforge.inference.pose_engine import CameraPoseResult, PoseEstimationResult
from spatialforge.inference.reconstruct_engine import ReconstructEngine


def _fake_depth(_self, frame: np.ndarray, *args, **kwargs) -> DepthResult:
    h, w = frame.shape[:2]
    depth = np.full((h, w), 2.0, dtype=np.float32)
    return DepthResult(
        depth_map=depth,
        min_depth=2.0,
        max_depth=2.0,
        is_metric=True,
        confidence_mean=0.8,
        focal_length_px=float(max(h, w)),
        width=w,
        height=h,
        model_used="mock/depth",
        model_license="apache-2.0",
        processing_time_ms=1.0,
    )


def _fake_pose(_self, frames: list[np.ndarray], output_pointcloud: bool = False) -> PoseEstimationResult:
    h, w = frames[0].shape[:2]
    focal = float(max(h, w))
    poses = [
        CameraPoseResult(
            frame_index=i,
            rotation=np.eye(3, dtype=np.float64),
            translation=np.array([0.02 * i, 0.0, 0.0], dtype=np.float64),
            fx=focal,
            fy=focal,
            cx=w / 2.0,
            cy=h / 2.0,
            width=w,
            height=h,
        )
        for i in range(len(frames))
    ]
    return PoseEstimationResult(poses=poses, pointcloud=None, processing_time_ms=1.0)


def _sample_frames() -> list[np.ndarray]:
    return [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]


def test_reconstruct_engine_invalid_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported reconstruct backend"):
        ReconstructEngine(MagicMock(), backend="invalid")


@pytest.mark.parametrize("backend", ["tsdf", "da3"])
def test_reconstruct_engine_unimplemented_backend_falls_back_to_legacy(monkeypatch, backend: str):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)

    engine = ReconstructEngine(MagicMock(), backend=backend)
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp:
        result = engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="pointcloud",
            output_dir=Path(tmp),
        )

    assert result.requested_backend == backend
    assert result.backend_used == "legacy"
    assert result.num_points is not None
    assert result.num_points > 0


def test_reconstruct_engine_legacy_backend_used(monkeypatch):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)

    engine = ReconstructEngine(MagicMock(), backend="legacy")
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp:
        result = engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="pointcloud",
            output_dir=Path(tmp),
        )

    assert result.requested_backend == "legacy"
    assert result.backend_used == "legacy"
