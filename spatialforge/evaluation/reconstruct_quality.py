"""Quality metrics for comparing reconstruction backends without GT meshes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def render_depth_from_points(
    points_world: NDArray[np.float32],
    *,
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Render depth from world-space points using a z-buffer projection."""
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be positive")

    depth = np.full((height, width), np.nan, dtype=np.float32)
    if points_world.size == 0:
        return depth

    points = np.asarray(points_world, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return depth
    points = points[:, :3]

    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64).reshape(3)

    points_cam = (rot.T @ (points - trans).T).T
    z = points_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return depth

    points_cam = points_cam[valid]
    z = z[valid]
    u = np.rint((points_cam[:, 0] * fx / z) + cx).astype(np.int32)
    v = np.rint((points_cam[:, 1] * fy / z) + cy).astype(np.int32)

    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(in_frame):
        return depth

    u = u[in_frame]
    v = v[in_frame]
    z = z[in_frame].astype(np.float32)
    flat_index = (v.astype(np.int64) * int(width)) + u.astype(np.int64)

    zbuffer = np.full(int(width) * int(height), np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, flat_index, z)
    zbuffer[np.isinf(zbuffer)] = np.nan
    return zbuffer.reshape((height, width))


def compute_rendered_depth_fit(
    *,
    points_world: NDArray[np.float32],
    observed_depth_maps: list[NDArray[np.float32]],
    poses: list[Any],
    downscale: float = 0.5,
    inlier_threshold_m: float = 0.05,
) -> dict[str, Any]:
    """Compute GT-free rendered-depth fit metrics for reconstructed geometry."""
    if downscale <= 0.0:
        raise ValueError("downscale must be > 0")
    if inlier_threshold_m <= 0:
        raise ValueError("inlier_threshold_m must be > 0")

    n = min(len(observed_depth_maps), len(poses))
    if n == 0:
        return {
            "frames_evaluated": 0,
            "frames_with_overlap": 0,
            "downscale": downscale,
            "inlier_threshold_m": inlier_threshold_m,
            "coverage": _empty_stats(),
            "abs_depth_error_m": _empty_stats(),
            "rel_depth_error": _empty_stats(),
            "inlier_ratio": _empty_stats(),
        }

    coverage_values: list[float] = []
    inlier_values: list[float] = []
    abs_errors: list[np.ndarray] = []
    rel_errors: list[np.ndarray] = []
    overlap_pixels_total = 0
    valid_pixels_total = 0
    frames_with_overlap = 0

    for depth_map, pose in zip(observed_depth_maps[:n], poses[:n], strict=False):
        obs = np.asarray(depth_map, dtype=np.float32)
        h, w = obs.shape[:2]
        out_h = max(1, int(round(h * downscale)))
        out_w = max(1, int(round(w * downscale)))
        if out_h != h or out_w != w:
            obs = cv2.resize(obs, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        sx = float(out_w) / float(w)
        sy = float(out_h) / float(h)
        fx = float(pose.fx) * sx
        fy = float(pose.fy) * sy
        cx = float(pose.cx) * sx
        cy = float(pose.cy) * sy

        rendered = render_depth_from_points(
            points_world,
            rotation=np.asarray(pose.rotation, dtype=np.float64),
            translation=np.asarray(pose.translation, dtype=np.float64),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=out_w,
            height=out_h,
        )

        valid_obs = np.isfinite(obs) & (obs > 1e-6)
        valid_render = np.isfinite(rendered) & (rendered > 1e-6)
        overlap = valid_obs & valid_render
        valid_obs_count = int(np.sum(valid_obs))
        overlap_count = int(np.sum(overlap))
        valid_pixels_total += valid_obs_count
        overlap_pixels_total += overlap_count

        if valid_obs_count > 0:
            coverage_values.append(float(overlap_count) / float(valid_obs_count))

        if overlap_count == 0:
            continue

        frames_with_overlap += 1
        err_abs = np.abs(rendered[overlap] - obs[overlap]).astype(np.float64)
        err_rel = err_abs / np.maximum(obs[overlap].astype(np.float64), 1e-6)
        abs_errors.append(err_abs)
        rel_errors.append(err_rel)
        inlier_values.append(float(np.mean(err_abs <= float(inlier_threshold_m))))

    abs_all = np.concatenate(abs_errors) if abs_errors else np.asarray([], dtype=np.float64)
    rel_all = np.concatenate(rel_errors) if rel_errors else np.asarray([], dtype=np.float64)

    return {
        "frames_evaluated": n,
        "frames_with_overlap": frames_with_overlap,
        "downscale": downscale,
        "inlier_threshold_m": inlier_threshold_m,
        "valid_pixels_total": valid_pixels_total,
        "overlap_pixels_total": overlap_pixels_total,
        "coverage": _stats(coverage_values),
        "abs_depth_error_m": _stats(abs_all),
        "rel_depth_error": _stats(rel_all),
        "inlier_ratio": _stats(inlier_values),
    }


def _empty_stats() -> dict[str, float | None]:
    return {"mean": None, "median": None, "p95": None}


def _stats(values: list[float] | NDArray[np.float64]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return _empty_stats()
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }
