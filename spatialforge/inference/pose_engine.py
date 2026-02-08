"""Camera pose estimation engine — multi-view geometry from DA3."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class CameraPoseResult:
    """Camera intrinsics and extrinsics for a single frame."""

    frame_index: int
    rotation: NDArray[np.float64]  # 3x3
    translation: NDArray[np.float64]  # 3
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class PoseEstimationResult:
    """Result of multi-view pose estimation."""

    poses: list[CameraPoseResult]
    pointcloud: NDArray[np.float32] | None = None  # N x 3
    processing_time_ms: float = 0.0


class PoseEngine:
    """Estimates camera poses from multiple views using depth + geometric reasoning.

    Strategy:
    1. Run depth estimation on each frame
    2. Extract feature correspondences
    3. Solve relative poses via essential matrix decomposition
    4. Refine with bundle adjustment (when available)
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def estimate_poses(
        self,
        frames: list[NDArray[np.uint8]],
        output_pointcloud: bool = False,
    ) -> PoseEstimationResult:
        """Estimate camera poses from a sequence of RGB frames.

        Args:
            frames: List of H x W x 3 uint8 RGB images.
            output_pointcloud: Whether to generate a sparse point cloud.

        Returns:
            PoseEstimationResult with camera poses and optional point cloud.
        """
        import cv2

        t0 = time.perf_counter()

        if len(frames) < 2:
            raise ValueError("At least 2 frames are required for pose estimation")

        h, w = frames[0].shape[:2]
        focal = max(w, h) * 1.2  # estimated focal length
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)

        # Feature detection (ORB for speed, can upgrade to SuperPoint)
        orb = cv2.ORB_create(nfeatures=2000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        poses: list[CameraPoseResult] = []
        all_3d_points: list[NDArray[np.float32]] = []

        # First frame is at origin
        poses.append(
            CameraPoseResult(
                frame_index=0,
                rotation=np.eye(3),
                translation=np.zeros(3),
                fx=focal,
                fy=focal,
                cx=cx,
                cy=cy,
                width=w,
                height=h,
            )
        )

        cumulative_R = np.eye(3)
        cumulative_t = np.zeros(3)

        for i in range(1, len(frames)):
            gray_prev = cv2.cvtColor(frames[i - 1], cv2.COLOR_RGB2GRAY)
            gray_curr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)

            kp1, des1 = orb.detectAndCompute(gray_prev, None)
            kp2, des2 = orb.detectAndCompute(gray_curr, None)

            if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
                # Not enough features, copy previous pose
                poses.append(
                    CameraPoseResult(
                        frame_index=i,
                        rotation=cumulative_R.copy(),
                        translation=cumulative_t.copy(),
                        fx=focal,
                        fy=focal,
                        cx=cx,
                        cy=cy,
                        width=w,
                        height=h,
                    )
                )
                continue

            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:500]

            if len(matches) < 8:
                poses.append(
                    CameraPoseResult(
                        frame_index=i,
                        rotation=cumulative_R.copy(),
                        translation=cumulative_t.copy(),
                        fx=focal,
                        fy=focal,
                        cx=cx,
                        cy=cy,
                        width=w,
                        height=h,
                    )
                )
                continue

            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                poses.append(
                    CameraPoseResult(
                        frame_index=i,
                        rotation=cumulative_R.copy(),
                        translation=cumulative_t.copy(),
                        fx=focal,
                        fy=focal,
                        cx=cx,
                        cy=cy,
                        width=w,
                        height=h,
                    )
                )
                continue

            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

            cumulative_R = R @ cumulative_R
            cumulative_t = cumulative_t + cumulative_R @ t.flatten()

            poses.append(
                CameraPoseResult(
                    frame_index=i,
                    rotation=cumulative_R.copy(),
                    translation=cumulative_t.copy(),
                    fx=focal,
                    fy=focal,
                    cx=cx,
                    cy=cy,
                    width=w,
                    height=h,
                )
            )

            # Triangulate points for point cloud
            if output_pointcloud and mask_pose is not None:
                inlier_pts1 = pts1[mask.ravel() == 1][:100]
                inlier_pts2 = pts2[mask.ravel() == 1][:100]
                if len(inlier_pts1) >= 4:
                    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
                    P2 = K @ np.hstack([R, t])
                    points_4d = cv2.triangulatePoints(P1, P2, inlier_pts1.T, inlier_pts2.T)
                    points_3d = (points_4d[:3] / points_4d[3:]).T.astype(np.float32)
                    all_3d_points.append(points_3d)

        pointcloud = None
        if output_pointcloud and all_3d_points:
            pointcloud = np.vstack(all_3d_points)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PoseEstimationResult(
            poses=poses,
            pointcloud=pointcloud,
            processing_time_ms=elapsed_ms,
        )
