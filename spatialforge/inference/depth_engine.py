"""Depth estimation inference engine — core of SpatialForge."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Result from depth estimation."""

    depth_map: NDArray[np.float32]  # H x W float32, metric depth in meters
    min_depth_m: float
    max_depth_m: float
    confidence_mean: float
    focal_length_px: float | None
    width: int
    height: int
    processing_time_ms: float


class DepthEngine:
    """Runs depth estimation using DA3/DAv2 models."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def estimate(
        self,
        image_rgb: NDArray[np.uint8],
        model_size: str = "giant",
    ) -> DepthResult:
        """Run monocular depth estimation on a single RGB image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.
            model_size: One of 'giant', 'large', 'base', 'small'.

        Returns:
            DepthResult with metric depth map and metadata.
        """
        t0 = time.perf_counter()

        h, w = image_rgb.shape[:2]
        pil_image = Image.fromarray(image_rgb)

        # Get model pipeline
        pipe = self._mm.get_depth_model(model_size)

        # Run inference
        result = pipe(pil_image)

        # Extract depth map
        depth_map = self._extract_depth(result, h, w)

        # Compute stats
        valid_mask = depth_map > 0
        min_depth = float(np.min(depth_map[valid_mask])) if np.any(valid_mask) else 0.0
        max_depth = float(np.max(depth_map[valid_mask])) if np.any(valid_mask) else 0.0

        # Confidence: higher values for areas with more structure
        confidence = self._estimate_confidence(depth_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DepthResult(
            depth_map=depth_map,
            min_depth_m=min_depth,
            max_depth_m=max_depth,
            confidence_mean=confidence,
            focal_length_px=self._estimate_focal_length(w, h),
            width=w,
            height=h,
            processing_time_ms=elapsed_ms,
        )

    def _extract_depth(self, result: Any, target_h: int, target_w: int) -> NDArray[np.float32]:
        """Extract and normalize depth map from pipeline output."""
        if isinstance(result, dict):
            # transformers pipeline returns {"depth": PIL.Image, "predicted_depth": Tensor}
            if "predicted_depth" in result:
                import torch

                depth_tensor = result["predicted_depth"]
                if isinstance(depth_tensor, torch.Tensor):
                    depth = depth_tensor.squeeze().cpu().numpy().astype(np.float32)
                else:
                    depth = np.array(depth_tensor, dtype=np.float32)
            elif "depth" in result:
                depth_pil = result["depth"]
                depth = np.array(depth_pil, dtype=np.float32)
            else:
                raise RuntimeError(f"Unexpected pipeline output keys: {result.keys()}")
        else:
            depth = np.array(result, dtype=np.float32)

        # Resize to target resolution if needed
        if depth.shape[0] != target_h or depth.shape[1] != target_w:
            import cv2

            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return depth

    def _estimate_confidence(self, depth: NDArray[np.float32]) -> float:
        """Estimate overall depth confidence based on gradient structure."""
        import cv2

        # Compute Laplacian to detect depth edges / structure
        laplacian = cv2.Laplacian(depth, cv2.CV_32F)
        structure = np.abs(laplacian)

        # Higher structure = more confident (more geometric detail)
        # Normalize to [0, 1] range
        max_struct = float(np.percentile(structure, 99)) if structure.size > 0 else 1.0
        if max_struct < 1e-6:
            return 0.5  # flat scene, medium confidence

        mean_struct = float(np.mean(structure))
        confidence = min(1.0, mean_struct / max_struct * 2)
        return round(max(0.1, confidence), 3)

    def _estimate_focal_length(self, w: int, h: int) -> float:
        """Estimate focal length in pixels from image dimensions.

        Uses a common heuristic: focal_length ≈ max(w, h) * 1.2
        This is a rough estimate for typical smartphone cameras (~26-28mm equivalent).
        """
        return round(max(w, h) * 1.2, 1)
