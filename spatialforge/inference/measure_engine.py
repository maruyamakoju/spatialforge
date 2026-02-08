"""Measurement engine — extract real-world distances from depth maps."""

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

# Known reference object dimensions (meters)
REFERENCE_SIZES: dict[str, tuple[float, float]] = {
    "a4_paper": (0.210, 0.297),  # 210mm x 297mm
    "credit_card": (0.0856, 0.0539),  # 85.6mm x 53.98mm
}


@dataclass
class MeasureResult:
    """Result from distance measurement."""

    distance_m: float
    confidence: float
    depth_at_points: list[float]
    calibration_method: str
    processing_time_ms: float


class MeasureEngine:
    """Measures real-world distances using depth estimation + optional reference objects."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def measure(
        self,
        image_rgb: NDArray[np.uint8],
        point1: tuple[float, float],
        point2: tuple[float, float],
        reference_type: str | None = None,
        reference_bbox: tuple[float, float, float, float] | None = None,
    ) -> MeasureResult:
        """Measure distance between two points in an image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.
            point1: (x, y) coordinates of first point.
            point2: (x, y) coordinates of second point.
            reference_type: Optional known object for scale calibration.
            reference_bbox: (x, y, w, h) bounding box of reference object.

        Returns:
            MeasureResult with real-world distance estimate.
        """
        from .depth_engine import DepthEngine

        t0 = time.perf_counter()

        h, w = image_rgb.shape[:2]

        # Validate points
        x1, y1 = int(point1[0]), int(point1[1])
        x2, y2 = int(point2[0]), int(point2[1])
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        # Run depth estimation (use 'large' for speed/accuracy tradeoff in measurements)
        depth_engine = DepthEngine(self._mm)
        depth_result = depth_engine.estimate(image_rgb, model_size="large")
        depth_map = depth_result.depth_map

        # Get depths at measurement points (average 3x3 patch for stability)
        d1 = self._sample_depth(depth_map, x1, y1)
        d2 = self._sample_depth(depth_map, x2, y2)

        # Estimate focal length
        focal = depth_result.focal_length_px or (max(w, h) * 1.2)

        # Determine calibration method and scale factor
        if reference_type and reference_bbox:
            scale, calibration_method = self._calibrate_with_reference(
                depth_map, reference_type, reference_bbox, focal, w, h
            )
            confidence = 0.90  # Reference-calibrated
        else:
            scale = 1.0
            calibration_method = "monocular_estimated"
            confidence = 0.70  # Uncalibrated

        # Apply scale to depths
        d1_metric = d1 * scale
        d2_metric = d2 * scale

        # 3D distance calculation using pinhole camera model
        X1 = (x1 - w / 2) * d1_metric / focal
        Y1 = (y1 - h / 2) * d1_metric / focal
        Z1 = d1_metric

        X2 = (x2 - w / 2) * d2_metric / focal
        Y2 = (y2 - h / 2) * d2_metric / focal
        Z2 = d2_metric

        distance_3d = float(np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return MeasureResult(
            distance_m=round(distance_3d, 4),
            confidence=confidence,
            depth_at_points=[round(d1_metric, 4), round(d2_metric, 4)],
            calibration_method=calibration_method,
            processing_time_ms=elapsed_ms,
        )

    def _sample_depth(self, depth_map: NDArray[np.float32], x: int, y: int, patch: int = 3) -> float:
        """Sample depth at a point using a patch average for stability."""
        h, w = depth_map.shape
        half = patch // 2
        y_min = max(0, y - half)
        y_max = min(h, y + half + 1)
        x_min = max(0, x - half)
        x_max = min(w, x + half + 1)
        patch_vals = depth_map[y_min:y_max, x_min:x_max]
        valid = patch_vals[patch_vals > 0]
        if len(valid) == 0:
            return float(depth_map[y, x])
        return float(np.median(valid))

    def _calibrate_with_reference(
        self,
        depth_map: NDArray[np.float32],
        ref_type: str,
        ref_bbox: tuple[float, float, float, float],
        focal: float,
        img_w: int,
        img_h: int,
    ) -> tuple[float, str]:
        """Calibrate metric scale using a known reference object.

        Returns (scale_factor, method_name).
        """
        if ref_type not in REFERENCE_SIZES:
            return 1.0, "unknown_reference"

        real_w, real_h = REFERENCE_SIZES[ref_type]
        bx, by, bw, bh = ref_bbox

        # Get average depth at reference object center
        cx = int(bx + bw / 2)
        cy = int(by + bh / 2)
        ref_depth = self._sample_depth(depth_map, cx, cy, patch=5)

        if ref_depth <= 0:
            return 1.0, f"{ref_type}_failed"

        # Calculate expected pixel size at this depth
        expected_pixel_w = real_w * focal / ref_depth
        expected_pixel_h = real_h * focal / ref_depth

        # Compare with actual bbox pixel size
        scale_w = expected_pixel_w / bw if bw > 0 else 1.0
        scale_h = expected_pixel_h / bh if bh > 0 else 1.0
        scale = (scale_w + scale_h) / 2.0

        return scale, f"{ref_type}_calibrated"
