"""ReconstructEngine backend selection tests."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
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


def _fake_depth_non_metric(_self, frame: np.ndarray, *args, **kwargs) -> DepthResult:
    h, w = frame.shape[:2]
    depth = np.full((h, w), 0.5, dtype=np.float32)
    return DepthResult(
        depth_map=depth,
        min_depth=0.0,
        max_depth=1.0,
        is_metric=False,
        confidence_mean=0.7,
        focal_length_px=float(max(h, w)),
        width=w,
        height=h,
        model_used="mock/depth-relative",
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


def _fake_open3d_module():
    class FakeImage:
        def __init__(self, data):
            self.data = np.asarray(data)

    class FakeRGBDImage:
        @staticmethod
        def create_from_color_and_depth(color, depth, **kwargs):
            return {"color": color, "depth": depth, "kwargs": kwargs}

    class FakePinholeCameraIntrinsic:
        def __init__(self, width, height, fx, fy, cx, cy):
            self.width = width
            self.height = height
            self.fx = fx
            self.fy = fy
            self.cx = cx
            self.cy = cy

    class FakeMesh:
        def __init__(self, points):
            self.vertices = np.asarray(points, dtype=np.float32)
            self.vertex_colors = np.tile(np.array([[0.2, 0.4, 0.8]], dtype=np.float32), (len(points), 1))

        def compute_vertex_normals(self):
            return None

    class FakePointCloud:
        def __init__(self, points):
            self.points = np.asarray(points, dtype=np.float32)
            self.colors = np.tile(np.array([[0.7, 0.2, 0.1]], dtype=np.float32), (len(points), 1))

    class FakeVolume:
        def __init__(self, **kwargs):
            self._points: list[np.ndarray] = []
            self.kwargs = kwargs

        def integrate(self, rgbd, intrinsic, extrinsic):
            del rgbd, intrinsic
            extrinsic = np.asarray(extrinsic, dtype=np.float64)
            offset = float(len(self._points)) * 0.1
            self._points.append(np.array([extrinsic[0, 3] + offset, extrinsic[1, 3], 1.0], dtype=np.float32))

        def extract_triangle_mesh(self):
            points = np.asarray(self._points, dtype=np.float32)
            if points.size == 0:
                points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            return FakeMesh(points)

        def extract_point_cloud(self):
            points = np.asarray(self._points, dtype=np.float32)
            if points.size == 0:
                points = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
            return FakePointCloud(points)

    def _write_triangle_mesh(path: str, mesh):
        Path(path).write_bytes(b"fake-mesh")
        return bool(len(mesh.vertices) >= 0)

    def _write_point_cloud(path: str, pointcloud):
        Path(path).write_bytes(b"fake-pointcloud")
        return bool(len(pointcloud.points) >= 0)

    fake_o3d = SimpleNamespace()
    fake_o3d.geometry = SimpleNamespace(
        Image=FakeImage,
        RGBDImage=FakeRGBDImage,
    )
    fake_o3d.camera = SimpleNamespace(PinholeCameraIntrinsic=FakePinholeCameraIntrinsic)
    fake_o3d.pipelines = SimpleNamespace(
        integration=SimpleNamespace(
            ScalableTSDFVolume=FakeVolume,
            TSDFVolumeColorType=SimpleNamespace(RGB8="RGB8"),
        )
    )
    fake_o3d.io = SimpleNamespace(
        write_triangle_mesh=_write_triangle_mesh,
        write_point_cloud=_write_point_cloud,
    )
    return fake_o3d


def test_reconstruct_engine_invalid_backend_rejected():
    with pytest.raises(ValueError, match="Unsupported reconstruct backend"):
        ReconstructEngine(MagicMock(), backend="invalid")


def test_reconstruct_engine_da3_backend_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)

    engine = ReconstructEngine(MagicMock(), backend="da3")
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp:
        result = engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="pointcloud",
            output_dir=Path(tmp),
        )

    assert result.requested_backend == "da3"
    assert result.backend_used == "legacy"
    assert result.num_points is not None
    assert result.num_points > 0


def test_reconstruct_engine_tsdf_missing_dependency_is_explicit(monkeypatch):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)

    def _raise_missing():
        raise RuntimeError("TSDF backend requires optional dependency 'open3d'.")

    monkeypatch.setattr(ReconstructEngine, "_import_open3d", staticmethod(_raise_missing))

    engine = ReconstructEngine(MagicMock(), backend="tsdf")
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp, pytest.raises(RuntimeError, match="open3d"):
        engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="mesh",
            output_dir=Path(tmp),
        )


def test_reconstruct_engine_tsdf_backend_used_when_dependency_available(monkeypatch):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)
    monkeypatch.setattr(ReconstructEngine, "_import_open3d", staticmethod(_fake_open3d_module))

    engine = ReconstructEngine(MagicMock(), backend="tsdf")
    output_exists = False
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp:
        result = engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="mesh",
            output_dir=Path(tmp),
        )
        output_exists = result.output_path.exists()

    assert result.requested_backend == "tsdf"
    assert result.backend_used == "tsdf"
    assert result.num_vertices is not None
    assert result.num_vertices > 0
    assert output_exists is True


def test_reconstruct_engine_tsdf_requires_metric_depth(monkeypatch):
    monkeypatch.setattr("spatialforge.inference.depth_engine.DepthEngine.estimate", _fake_depth_non_metric)
    monkeypatch.setattr("spatialforge.inference.pose_engine.PoseEngine.estimate_poses", _fake_pose)
    monkeypatch.setattr(ReconstructEngine, "_import_open3d", staticmethod(_fake_open3d_module))

    engine = ReconstructEngine(MagicMock(), backend="tsdf")
    with TemporaryDirectory(prefix="sf_reconstruct_test_") as tmp, pytest.raises(RuntimeError, match="metric depth"):
        engine.reconstruct(
            frames=_sample_frames(),
            quality="draft",
            output_format="mesh",
            output_dir=Path(tmp),
        )


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
