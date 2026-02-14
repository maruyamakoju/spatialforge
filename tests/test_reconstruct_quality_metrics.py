"""Tests for GT-free reconstruction quality metrics."""

from __future__ import annotations

import numpy as np

from spatialforge.evaluation.reconstruct_quality import compute_rendered_depth_fit
from spatialforge.inference.pose_engine import CameraPoseResult


def _pose(width: int, height: int) -> CameraPoseResult:
    fx = 40.0
    fy = 40.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return CameraPoseResult(
        frame_index=0,
        rotation=np.eye(3, dtype=np.float64),
        translation=np.zeros(3, dtype=np.float64),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
    )


def _points_from_depth(depth: np.ndarray, pose: CameraPoseResult) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.reshape(-1).astype(np.float32)
    x = ((u.reshape(-1) - pose.cx) * z / pose.fx).astype(np.float32)
    y = ((v.reshape(-1) - pose.cy) * z / pose.fy).astype(np.float32)
    points_cam = np.stack([x, y, z], axis=-1)
    points_world = (pose.rotation @ points_cam.T).T + pose.translation
    return points_world.astype(np.float32)


def test_rendered_depth_fit_perfect_projection():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    pose = _pose(width=8, height=8)
    points = _points_from_depth(depth, pose)

    metrics = compute_rendered_depth_fit(
        points_world=points,
        observed_depth_maps=[depth],
        poses=[pose],
        downscale=1.0,
        inlier_threshold_m=0.05,
    )

    assert metrics["frames_evaluated"] == 1
    assert metrics["frames_with_overlap"] == 1
    assert metrics["coverage"]["mean"] is not None
    assert metrics["coverage"]["mean"] > 0.99
    assert metrics["abs_depth_error_m"]["mean"] is not None
    assert metrics["abs_depth_error_m"]["mean"] < 1e-4
    assert metrics["inlier_ratio"]["mean"] is not None
    assert metrics["inlier_ratio"]["mean"] > 0.99


def test_rendered_depth_fit_detects_depth_bias():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    pose = _pose(width=8, height=8)
    points = _points_from_depth(depth, pose)
    points[:, 2] += 0.5  # Add depth bias

    metrics = compute_rendered_depth_fit(
        points_world=points,
        observed_depth_maps=[depth],
        poses=[pose],
        downscale=1.0,
        inlier_threshold_m=0.05,
    )

    assert metrics["abs_depth_error_m"]["mean"] is not None
    assert metrics["abs_depth_error_m"]["mean"] > 0.3
    assert metrics["inlier_ratio"]["mean"] is not None
    assert metrics["inlier_ratio"]["mean"] < 0.5


def test_rendered_depth_fit_empty_input_is_handled():
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    pose = _pose(width=8, height=8)

    metrics = compute_rendered_depth_fit(
        points_world=np.zeros((0, 3), dtype=np.float32),
        observed_depth_maps=[depth],
        poses=[pose],
        downscale=1.0,
        inlier_threshold_m=0.05,
    )

    assert metrics["frames_evaluated"] == 1
    assert metrics["frames_with_overlap"] == 0
    assert metrics["coverage"]["mean"] == 0.0
    assert metrics["abs_depth_error_m"]["mean"] is None
