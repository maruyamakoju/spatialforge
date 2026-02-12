"""3D reconstruction engine — video to 3D Gaussian Splatting / point cloud / mesh."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


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
    processing_time_ms: float = 0.0


class ReconstructEngine:
    """Reconstructs 3D scenes from video using depth estimation + multi-view geometry.

    Pipeline:
    1. Extract keyframes from video
    2. Estimate depth for each keyframe
    3. Estimate camera poses
    4. Generate dense point cloud from depth-projected frames
    5. (Optional) Train 3D Gaussian Splatting / convert to mesh
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def reconstruct(
        self,
        frames: list[NDArray[np.uint8]],
        quality: str = "standard",
        output_format: str = "pointcloud",
        output_dir: Path | None = None,
    ) -> ReconstructResult:
        """Reconstruct a 3D scene from a sequence of frames.

        Args:
            frames: List of RGB images (keyframes from video).
            quality: 'draft', 'standard', or 'high'.
            output_format: 'gaussian', 'pointcloud', or 'mesh'.
            output_dir: Where to save results. Uses temp dir if None.
        """
        t0 = time.perf_counter()

        if len(frames) < 3:
            raise ValueError("At least 3 frames are required for reconstruction")

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="sf_recon_"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Quality settings
        quality_config = {
            "draft": {"depth_model": "small", "max_frames": 30},
            "standard": {"depth_model": "large", "max_frames": 100},
            "high": {"depth_model": "giant", "max_frames": 200},
        }
        cfg = quality_config.get(quality, quality_config["standard"])
        frames = frames[: cfg["max_frames"]]

        # Step 1: Estimate depth for each frame
        from .depth_engine import DepthEngine

        depth_engine = DepthEngine(self._mm)
        depth_maps = []
        for frame in frames:
            depth_result = depth_engine.estimate(frame, model_size=cfg["depth_model"])
            depth_maps.append(depth_result.depth_map)

        # Step 2: Estimate camera poses
        from .pose_engine import PoseEngine

        pose_engine = PoseEngine(self._mm)
        pose_result = pose_engine.estimate_poses(frames, output_pointcloud=True)

        # Step 3: Generate dense point cloud
        all_points, all_colors = self._depth_to_pointcloud(frames, depth_maps, pose_result.poses)

        if len(all_points) == 0:
            raise RuntimeError("Failed to generate point cloud: no valid 3D points")

        # Compute bounding box
        bbox_min = all_points.min(axis=0).tolist()
        bbox_max = all_points.max(axis=0).tolist()

        # Save results
        if output_format == "pointcloud":
            out_path = output_dir / "scene.ply"
            self._save_ply(out_path, all_points, all_colors)
            recon_result = ReconstructResult(
                output_path=out_path,
                output_format="pointcloud",
                num_points=len(all_points),
                bounding_box=[bbox_min, bbox_max],
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        elif output_format == "gaussian":
            # For MVP, save as PLY with Gaussian-compatible format
            # Full 3DGS training can be added with gsplat/nerfstudio integration
            out_path = output_dir / "scene_gaussian.ply"
            self._save_gaussian_ply(out_path, all_points, all_colors)
            recon_result = ReconstructResult(
                output_path=out_path,
                output_format="gaussian",
                num_gaussians=len(all_points),
                bounding_box=[bbox_min, bbox_max],
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )
        else:  # mesh
            out_path = output_dir / "scene.ply"
            self._save_ply(out_path, all_points, all_colors)
            recon_result = ReconstructResult(
                output_path=out_path,
                output_format="mesh",
                num_vertices=len(all_points),
                bounding_box=[bbox_min, bbox_max],
                processing_time_ms=(time.perf_counter() - t0) * 1000,
            )

        # Save camera poses
        poses_json = []
        for p in pose_result.poses:
            poses_json.append({
                "frame_index": p.frame_index,
                "rotation": p.rotation.tolist(),
                "translation": p.translation.tolist(),
                "intrinsics": {"fx": p.fx, "fy": p.fy, "cx": p.cx, "cy": p.cy},
            })
        recon_result.camera_poses_json = json.dumps(poses_json)
        poses_path = output_dir / "cameras.json"
        poses_path.write_text(recon_result.camera_poses_json)

        return recon_result

    def _depth_to_pointcloud(
        self,
        frames: list[NDArray[np.uint8]],
        depth_maps: list[NDArray[np.float32]],
        poses: list,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
        """Back-project depth maps into 3D using estimated camera poses."""
        all_points = []
        all_colors = []

        for frame, depth, pose in zip(frames, depth_maps, poses, strict=False):
            h, w = frame.shape[:2]
            focal = pose.fx
            cx, cy = pose.cx, pose.cy

            # Create pixel grid
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Subsample for efficiency (every 4th pixel)
            step = 4
            u = u[::step, ::step].flatten()
            v = v[::step, ::step].flatten()
            d = depth[::step, ::step].flatten()
            colors = frame[::step, ::step].reshape(-1, 3)

            # Filter invalid depths
            valid = d > 0.01
            u, v, d, colors = u[valid], v[valid], d[valid], colors[valid]

            if len(u) == 0:
                continue

            # Back-project to 3D (camera coordinates)
            X = (u - cx) * d / focal
            Y = (v - cy) * d / focal
            Z = d

            pts_cam = np.stack([X, Y, Z], axis=-1).astype(np.float32)

            # Transform to world coordinates
            R = pose.rotation
            t = pose.translation
            pts_world = (R @ pts_cam.T).T + t

            all_points.append(pts_world)
            all_colors.append(colors)

        if not all_points:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        return np.vstack(all_points), np.vstack(all_colors).astype(np.uint8)

    def _save_ply(self, path: Path, points: NDArray[np.float32], colors: NDArray[np.uint8]) -> None:
        """Save point cloud as PLY file (vectorized binary write)."""
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

        # Pack xyz (float32) + rgb (uint8) into a structured array for single write
        vertex_dtype = np.dtype([("xyz", np.float32, 3), ("rgb", np.uint8, 3)])
        vertices = np.empty(n, dtype=vertex_dtype)
        vertices["xyz"] = points
        vertices["rgb"] = colors

        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(vertices.tobytes())

    def _save_gaussian_ply(self, path: Path, points: NDArray[np.float32], colors: NDArray[np.uint8]) -> None:
        """Save as PLY with Gaussian splat attributes (vectorized binary write).

        This is a simplified version; full 3DGS training would produce optimized splats.
        """
        n = len(points)

        # Initialize Gaussian attributes
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

        # Pack all fields into a structured array for single write
        gs_dtype = np.dtype([
            ("xyz", np.float32, 3),
            ("rgb", np.uint8, 3),
            ("scale", np.float32, 3),
            ("rot", np.float32, 4),
            ("opacity", np.float32, 1),
        ])
        vertices = np.empty(n, dtype=gs_dtype)
        vertices["xyz"] = points
        vertices["rgb"] = colors
        vertices["scale"] = scales
        vertices["rot"] = rotations
        vertices["opacity"] = opacities

        with open(path, "wb") as f:
            f.write(header.encode())
            f.write(vertices.tobytes())
