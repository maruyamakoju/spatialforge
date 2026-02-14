"""3D reconstruction engine — video to 3D Gaussian Splatting / point cloud / mesh."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model_manager import ModelManager
    from .pose_engine import PoseEstimationResult

logger = logging.getLogger(__name__)
_SUPPORTED_RECONSTRUCT_BACKENDS = {"legacy", "tsdf", "da3"}
_QUALITY_CONFIG = {
    "draft": {"depth_model": "small", "max_frames": 30},
    "standard": {"depth_model": "large", "max_frames": 100},
    "high": {"depth_model": "giant", "max_frames": 200},
}
_TSDF_CONFIG = {
    "draft": {"voxel_length": 0.03, "sdf_trunc": 0.12, "depth_trunc": 8.0},
    "standard": {"voxel_length": 0.02, "sdf_trunc": 0.08, "depth_trunc": 10.0},
    "high": {"voxel_length": 0.01, "sdf_trunc": 0.04, "depth_trunc": 12.0},
}

ProgressCallback = Callable[[str, dict[str, Any] | None], None]
EvalHook = Callable[[np.ndarray], None]


@dataclass
class ReconstructResult:
    """Result from 3D reconstruction."""

    output_path: Path  # Path to .ply / .obj / .pcd file
    output_format: str  # "gaussian" | "pointcloud" | "mesh"
    num_gaussians: int | None = None
    num_points: int | None = None
    num_vertices: int | None = None
    bounding_box: list[list[float]] | None = None  # [[min_x,min_y,min_z],[max_x,max_y,max_z]]
    camera_poses_json: str | None = None
    requested_backend: str = "legacy"
    backend_used: str = "legacy"
    processing_time_ms: float = 0.0


class ReconstructEngine:
    """Reconstructs 3D scenes from video using depth estimation + multi-view geometry."""

    def __init__(self, model_manager: ModelManager, backend: str = "legacy") -> None:
        self._mm = model_manager
        normalized = backend.lower()
        if normalized not in _SUPPORTED_RECONSTRUCT_BACKENDS:
            raise ValueError(
                f"Unsupported reconstruct backend: '{backend}'. "
                f"Supported values: {sorted(_SUPPORTED_RECONSTRUCT_BACKENDS)}"
            )
        self._backend = normalized

    @property
    def backend(self) -> str:
        return self._backend

    def reconstruct(
        self,
        frames: list[NDArray[np.uint8]],
        quality: str = "standard",
        output_format: str = "pointcloud",
        output_dir: Path | None = None,
        progress_callback: ProgressCallback | None = None,
        eval_hook: EvalHook | None = None,
        eval_max_points: int = 200_000,
    ) -> ReconstructResult:
        """Reconstruct a 3D scene from a sequence of frames."""
        if len(frames) < 3:
            raise ValueError("At least 3 frames are required for reconstruction")

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="sf_recon_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        quality_cfg = _QUALITY_CONFIG.get(quality, _QUALITY_CONFIG["standard"])
        limited_frames = frames[: quality_cfg["max_frames"]]

        if self._backend == "legacy":
            return self._reconstruct_legacy(
                limited_frames,
                quality=quality,
                output_format=output_format,
                output_dir=output_dir,
                depth_model=quality_cfg["depth_model"],
                requested_backend="legacy",
                backend_used="legacy",
                progress_callback=progress_callback,
                eval_hook=eval_hook,
                eval_max_points=eval_max_points,
            )

        if self._backend == "tsdf":
            return self._reconstruct_tsdf(
                limited_frames,
                quality=quality,
                output_format=output_format,
                output_dir=output_dir,
                depth_model=quality_cfg["depth_model"],
                requested_backend="tsdf",
                progress_callback=progress_callback,
                eval_hook=eval_hook,
                eval_max_points=eval_max_points,
            )

        logger.warning(
            "Reconstruct backend '%s' is not implemented yet; falling back to legacy pipeline",
            self._backend,
        )
        return self._reconstruct_legacy(
            limited_frames,
            quality=quality,
            output_format=output_format,
            output_dir=output_dir,
            depth_model=quality_cfg["depth_model"],
            requested_backend=self._backend,
            backend_used="legacy",
            progress_callback=progress_callback,
            eval_hook=eval_hook,
            eval_max_points=eval_max_points,
        )

    def _reconstruct_legacy(
        self,
        frames: list[NDArray[np.uint8]],
        *,
        quality: str,
        output_format: str,
        output_dir: Path,
        depth_model: str,
        requested_backend: str,
        backend_used: str,
        progress_callback: ProgressCallback | None,
        eval_hook: EvalHook | None,
        eval_max_points: int,
    ) -> ReconstructResult:
        t0 = time.perf_counter()

        self._emit_progress(
            progress_callback,
            "estimating_depth",
            num_frames=len(frames),
            depth_model=depth_model,
            reconstruct_backend=requested_backend,
        )
        depth_maps = self._estimate_depth_maps(
            frames,
            depth_model=depth_model,
            require_metric=False,
            backend=requested_backend,
        )

        self._emit_progress(progress_callback, "estimating_pose", num_frames=len(frames))
        pose_result = self._estimate_poses(frames)

        self._emit_progress(progress_callback, "integrating_legacy", num_frames=len(frames))
        all_points, all_colors = self._depth_to_pointcloud(frames, depth_maps, pose_result.poses)
        if len(all_points) == 0:
            raise RuntimeError("Failed to generate point cloud: no valid 3D points")

        bbox = self._compute_bbox(all_points)
        if output_format == "pointcloud":
            self._emit_progress(progress_callback, "extracting_pointcloud")
            out_path = output_dir / "scene_legacy_pointcloud.ply"
            self._save_ply(out_path, all_points, all_colors)
            result = ReconstructResult(
                output_path=out_path,
                output_format="pointcloud",
                num_points=len(all_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used=backend_used,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        elif output_format == "gaussian":
            self._emit_progress(progress_callback, "extracting_mesh")
            out_path = output_dir / "scene_legacy_gaussian.ply"
            self._save_gaussian_ply(out_path, all_points, all_colors)
            result = ReconstructResult(
                output_path=out_path,
                output_format="gaussian",
                num_gaussians=len(all_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used=backend_used,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            self._emit_progress(progress_callback, "extracting_mesh")
            out_path = output_dir / "scene_legacy_mesh.ply"
            self._save_ply(out_path, all_points, all_colors)
            result = ReconstructResult(
                output_path=out_path,
                output_format="mesh",
                num_vertices=len(all_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used=backend_used,
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

        self._attach_camera_poses(result, pose_result, output_dir)
        self._emit_eval_hook(eval_hook, all_points, max_points=eval_max_points)
        return result

    def _reconstruct_tsdf(
        self,
        frames: list[NDArray[np.uint8]],
        *,
        quality: str,
        output_format: str,
        output_dir: Path,
        depth_model: str,
        requested_backend: str,
        progress_callback: ProgressCallback | None,
        eval_hook: EvalHook | None,
        eval_max_points: int,
    ) -> ReconstructResult:
        t0 = time.perf_counter()
        o3d = self._import_open3d()
        tsdf_cfg = _TSDF_CONFIG.get(quality, _TSDF_CONFIG["standard"])

        self._emit_progress(
            progress_callback,
            "estimating_depth",
            num_frames=len(frames),
            depth_model=depth_model,
            reconstruct_backend=requested_backend,
        )
        depth_maps = self._estimate_depth_maps(
            frames,
            depth_model=depth_model,
            require_metric=True,
            backend=requested_backend,
        )

        self._emit_progress(progress_callback, "estimating_pose", num_frames=len(frames))
        pose_result = self._estimate_poses(frames)

        self._emit_progress(progress_callback, "tsdf_integrating", num_frames=len(frames))
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=tsdf_cfg["voxel_length"],
            sdf_trunc=tsdf_cfg["sdf_trunc"],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )
        for frame, depth_map, pose in zip(frames, depth_maps, pose_result.poses, strict=False):
            color_image = o3d.geometry.Image(np.ascontiguousarray(frame.astype(np.uint8)))
            depth_image = o3d.geometry.Image(np.ascontiguousarray(depth_map.astype(np.float32)))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_image,
                depth_image,
                depth_scale=1.0,
                depth_trunc=tsdf_cfg["depth_trunc"],
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                int(pose.width),
                int(pose.height),
                float(pose.fx),
                float(pose.fy),
                float(pose.cx),
                float(pose.cy),
            )
            extrinsic = self._world_to_camera_extrinsic(pose.rotation, pose.translation)
            volume.integrate(rgbd, intrinsic, extrinsic)

        self._emit_progress(progress_callback, "extracting_mesh")
        mesh = volume.extract_triangle_mesh()
        if hasattr(mesh, "compute_vertex_normals"):
            mesh.compute_vertex_normals()
        pointcloud = volume.extract_point_cloud()

        mesh_vertices = self._as_points(np.asarray(mesh.vertices))
        mesh_colors = self._as_colors(np.asarray(mesh.vertex_colors), fallback_count=len(mesh_vertices))
        pcd_points = self._as_points(np.asarray(pointcloud.points))
        pcd_colors = self._as_colors(np.asarray(pointcloud.colors), fallback_count=len(pcd_points))
        eval_points = pcd_points if len(pcd_points) > 0 else mesh_vertices

        bbox_points = pcd_points if len(pcd_points) > 0 else mesh_vertices
        if len(bbox_points) == 0:
            raise RuntimeError("TSDF integration produced no geometry")
        bbox = self._compute_bbox(bbox_points)

        if output_format == "mesh":
            self._emit_progress(progress_callback, "saving_mesh")
            out_path = output_dir / "scene_tsdf_mesh.ply"
            wrote = bool(o3d.io.write_triangle_mesh(str(out_path), mesh))
            if not wrote or not out_path.exists():
                self._save_ply(out_path, bbox_points, self._fallback_colors(len(bbox_points)))
            result = ReconstructResult(
                output_path=out_path,
                output_format="mesh",
                num_vertices=len(mesh_vertices) or len(bbox_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used="tsdf",
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        elif output_format == "pointcloud":
            self._emit_progress(progress_callback, "extracting_pointcloud")
            out_path = output_dir / "scene_tsdf_pointcloud.ply"
            point_points = pcd_points if len(pcd_points) > 0 else mesh_vertices
            point_colors = pcd_colors if len(pcd_colors) > 0 else mesh_colors
            wrote = bool(o3d.io.write_point_cloud(str(out_path), pointcloud))
            if not wrote or not out_path.exists():
                self._save_ply(out_path, point_points, point_colors)
            result = ReconstructResult(
                output_path=out_path,
                output_format="pointcloud",
                num_points=len(point_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used="tsdf",
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        else:
            self._emit_progress(progress_callback, "saving_gaussians")
            gaussian_points = mesh_vertices if len(mesh_vertices) > 0 else pcd_points
            gaussian_colors = mesh_colors if len(mesh_colors) > 0 else pcd_colors
            if len(gaussian_points) == 0:
                raise RuntimeError("TSDF gaussian export requires non-empty geometry")
            out_path = output_dir / "scene_tsdf_gaussian.ply"
            self._save_gaussian_ply(out_path, gaussian_points, gaussian_colors)
            result = ReconstructResult(
                output_path=out_path,
                output_format="gaussian",
                num_gaussians=len(gaussian_points),
                bounding_box=bbox,
                requested_backend=requested_backend,
                backend_used="tsdf",
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

        self._attach_camera_poses(result, pose_result, output_dir)
        self._emit_eval_hook(eval_hook, eval_points, max_points=eval_max_points)
        return result

    def _estimate_depth_maps(
        self,
        frames: list[NDArray[np.uint8]],
        *,
        depth_model: str,
        require_metric: bool,
        backend: str,
    ) -> list[NDArray[np.float32]]:
        from .depth_engine import DepthEngine

        depth_engine = DepthEngine(self._mm)
        depth_maps: list[NDArray[np.float32]] = []
        for i, frame in enumerate(frames):
            depth_result = depth_engine.estimate(frame, model_size=depth_model)
            if require_metric and not depth_result.is_metric:
                raise RuntimeError(
                    f"{backend.upper()} backend requires metric depth maps, but frame {i} "
                    f"from model '{depth_result.model_used}' was non-metric"
                )
            depth_maps.append(depth_result.depth_map)
        return depth_maps

    def _estimate_poses(self, frames: list[NDArray[np.uint8]]) -> PoseEstimationResult:
        from .pose_engine import PoseEngine

        pose_engine = PoseEngine(self._mm)
        return pose_engine.estimate_poses(frames, output_pointcloud=True)

    @staticmethod
    def _import_open3d() -> Any:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "TSDF backend requires optional dependency 'open3d'. "
                'Install with: pip install -e ".[tsdf]"'
            ) from exc
        return o3d

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        step: str,
        **meta: Any,
    ) -> None:
        if callback is None:
            return
        callback(step, meta or None)

    @staticmethod
    def _emit_eval_hook(
        callback: EvalHook | None,
        points_world: NDArray[np.float32],
        *,
        max_points: int,
    ) -> None:
        if callback is None:
            return
        points = np.asarray(points_world, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            callback(np.zeros((0, 3), dtype=np.float32))
            return
        points = points[:, :3]
        if max_points > 0 and len(points) > max_points:
            idx = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
            points = points[idx]
        callback(np.ascontiguousarray(points, dtype=np.float32))

    @staticmethod
    def _world_to_camera_extrinsic(
        rotation: NDArray[np.float64],
        translation: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        transform_wc = np.eye(4, dtype=np.float64)
        transform_wc[:3, :3] = np.asarray(rotation, dtype=np.float64)
        transform_wc[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        return np.linalg.inv(transform_wc)

    @staticmethod
    def _as_points(values: NDArray[Any]) -> NDArray[np.float32]:
        if values.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        return arr[:, :3]

    @staticmethod
    def _fallback_colors(count: int) -> NDArray[np.uint8]:
        if count == 0:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.full((count, 3), 200, dtype=np.uint8)

    def _as_colors(self, values: NDArray[Any], *, fallback_count: int) -> NDArray[np.uint8]:
        if values.size == 0:
            return self._fallback_colors(fallback_count)
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        arr = arr[:, :3]
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
        return arr.astype(np.uint8)

    @staticmethod
    def _compute_bbox(points: NDArray[np.float32]) -> list[list[float]]:
        bbox_min = points.min(axis=0).tolist()
        bbox_max = points.max(axis=0).tolist()
        return [bbox_min, bbox_max]

    @staticmethod
    def _attach_camera_poses(result: ReconstructResult, pose_result: PoseEstimationResult, output_dir: Path) -> None:
        poses_json = []
        for pose in pose_result.poses:
            poses_json.append(
                {
                    "frame_index": pose.frame_index,
                    "rotation": pose.rotation.tolist(),
                    "translation": pose.translation.tolist(),
                    "intrinsics": {"fx": pose.fx, "fy": pose.fy, "cx": pose.cx, "cy": pose.cy},
                }
            )
        result.camera_poses_json = json.dumps(poses_json)
        poses_path = output_dir / "cameras.json"
        poses_path.write_text(result.camera_poses_json)

    def _depth_to_pointcloud(
        self,
        frames: list[NDArray[np.uint8]],
        depth_maps: list[NDArray[np.float32]],
        poses: list[Any],
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """Back-project depth maps into 3D using estimated camera poses."""
        all_points = []
        all_colors = []

        for _, (frame, depth, pose) in enumerate(zip(frames, depth_maps, poses, strict=False)):
            h, w = frame.shape[:2]
            focal = pose.fx
            cx, cy = pose.cx, pose.cy

            u, v = np.meshgrid(np.arange(w), np.arange(h))

            step = 4
            u = u[::step, ::step].flatten()
            v = v[::step, ::step].flatten()
            d = depth[::step, ::step].flatten()
            colors = frame[::step, ::step].reshape(-1, 3)

            valid = d > 0.01
            u, v, d, colors = u[valid], v[valid], d[valid], colors[valid]

            if len(u) == 0:
                continue

            x_coord = (u - cx) * d / focal
            y_coord = (v - cy) * d / focal
            z_coord = d

            points_camera = np.stack([x_coord, y_coord, z_coord], axis=-1).astype(np.float32)
            rotation = pose.rotation
            translation = pose.translation
            points_world = (rotation @ points_camera.T).T + translation

            all_points.append(points_world)
            all_colors.append(colors)

        if not all_points:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        return np.vstack(all_points), np.vstack(all_colors).astype(np.uint8)

    def _save_ply(self, path: Path, points: NDArray[np.float32], colors: NDArray[np.uint8]) -> None:
        """Save point cloud as PLY file."""
        n = len(points)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        with open(path, "wb") as file_obj:
            file_obj.write(header.encode())
            for i in range(n):
                file_obj.write(points[i].tobytes())
                file_obj.write(colors[i].tobytes())

    def _save_gaussian_ply(self, path: Path, points: NDArray[np.float32], colors: NDArray[np.uint8]) -> None:
        """Save as PLY with Gaussian splat attributes (position, color, scale, rotation, opacity)."""
        n = len(points)
        scales = np.full((n, 3), 0.01, dtype=np.float32)
        rotations = np.zeros((n, 4), dtype=np.float32)
        rotations[:, 0] = 1.0
        opacities = np.full((n, 1), 0.8, dtype=np.float32)

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "property float opacity\n"
            "end_header\n"
        )

        with open(path, "wb") as file_obj:
            file_obj.write(header.encode())
            for i in range(n):
                file_obj.write(points[i].tobytes())
                file_obj.write(colors[i].tobytes())
                file_obj.write(scales[i].tobytes())
                file_obj.write(rotations[i].tobytes())
                file_obj.write(opacities[i].tobytes())
