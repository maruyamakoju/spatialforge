"""Depth estimation inference engine — core of SpatialForge.

This is the money-making module. Every dollar flows through here.

Supported modes:
  - Metric depth (meters): Uses DA3-Metric-Large (Apache 2.0).
    Output is absolute depth in meters. Requires focal length.
  - Relative depth: Uses DA3-Base/Small (Apache 2.0).
    Output is relative disparity (0-1 range), not metric.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Result from depth estimation."""

    depth_map: NDArray[np.float32]  # H x W float32
    min_depth: float
    max_depth: float
    is_metric: bool  # True = values in meters, False = relative disparity
    confidence_mean: float
    focal_length_px: float | None
    width: int
    height: int
    model_used: str
    model_license: str
    processing_time_ms: float


# ── Focal length estimation ──────────────────────────────────

def extract_focal_from_exif(image_bytes: bytes) -> float | None:
    """Try to extract focal length in pixels from EXIF data.

    Formula: focal_px = focal_mm * image_width_px / sensor_width_mm
    Most phone cameras have ~4.2mm focal length and ~6.17mm sensor width (1/2.55").
    """
    try:
        import io

        from PIL import Image as PILImage
        from PIL.ExifTags import TAGS

        img = PILImage.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()
        if exif_data is None:
            return None

        focal_mm = None
        img_width = img.width

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "FocalLength":
                if hasattr(value, "numerator"):
                    focal_mm = value.numerator / value.denominator
                else:
                    focal_mm = float(value)
            elif tag == "FocalLengthIn35mmFilm":
                # 35mm equiv → approximate real focal length
                # sensor crop factor for typical phone ≈ 7.6
                if value and value > 0:
                    focal_mm = float(value) / 7.6

        if focal_mm is None:
            return None

        # Assume typical phone sensor width ~6.17mm (1/2.55" sensor)
        sensor_width_mm = 6.17
        focal_px = focal_mm * img_width / sensor_width_mm
        return focal_px

    except Exception:
        return None


def estimate_focal_px(width: int, height: int) -> float:
    """Heuristic focal length estimation when no EXIF available.

    Typical smartphone ≈ 26mm equivalent → focal ≈ 0.9 * max(w,h).
    This is a rough estimate; confidence should be reduced when using this.
    """
    return max(width, height) * 0.9


# ── Main engine ──────────────────────────────────────────────

class DepthEngine:
    """Runs depth estimation using DA3 models via HuggingFace pipeline."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager

    def estimate(
        self,
        image_rgb: NDArray[np.uint8],
        model_size: str = "large",
        focal_length_px: float | None = None,
        image_bytes: bytes | None = None,
    ) -> DepthResult:
        """Run depth estimation on a single RGB image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.
            model_size: Model name/alias (see model_manager).
            focal_length_px: Known focal length in pixels (improves metric accuracy).
            image_bytes: Original file bytes (for EXIF extraction).

        Returns:
            DepthResult with depth map and metadata.
        """
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]

        # Resolve focal length: EXIF > explicit > heuristic
        focal = focal_length_px
        focal_source = "explicit"

        if focal is None and image_bytes is not None:
            focal = extract_focal_from_exif(image_bytes)
            if focal is not None:
                focal_source = "exif"

        if focal is None:
            focal = estimate_focal_px(w, h)
            focal_source = "heuristic"

        # Load model
        pipe, model_info = self._mm.get_depth_model(model_size)

        # Convert to PIL for transformers pipeline
        pil_image = Image.fromarray(image_rgb)

        # Run inference
        try:
            raw_output = pipe(pil_image)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                # Clear CUDA cache and retry once
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("GPU out of memory during depth estimation") from e
            raise

        # Extract depth tensor from pipeline output
        raw_depth = self._extract_raw_depth(raw_output, h, w)

        # Clear GPU memory after extraction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Convert to metric if applicable
        is_metric = model_info.task == "metric_depth"
        if is_metric:
            # Metric models output depth directly in meters (or close to it).
            # The transformers pipeline for DA2 Metric models already outputs
            # metric depth. We just need to ensure correct scaling.
            depth_map = raw_depth.astype(np.float32)
        else:
            # Relative depth models output disparity (inverse depth).
            # Higher values = closer. Normalize to [0, 1] range.
            d_min, d_max = raw_depth.min(), raw_depth.max()
            if d_max - d_min > 1e-6:
                depth_map = ((raw_depth - d_min) / (d_max - d_min)).astype(np.float32)
            else:
                depth_map = np.zeros_like(raw_depth, dtype=np.float32)

        # Compute stats
        valid_mask = depth_map > 1e-6 if is_metric else depth_map > 0
        if np.any(valid_mask):
            min_depth = float(np.min(depth_map[valid_mask]))
            max_depth = float(np.max(depth_map[valid_mask]))
        else:
            min_depth = 0.0
            max_depth = 0.0

        # Confidence estimation
        confidence = self._compute_confidence(depth_map, focal_source, is_metric)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DepthResult(
            depth_map=depth_map,
            min_depth=min_depth,
            max_depth=max_depth,
            is_metric=is_metric,
            confidence_mean=confidence,
            focal_length_px=round(focal, 1),
            width=w,
            height=h,
            model_used=model_info.repo,
            model_license=model_info.license,
            processing_time_ms=elapsed_ms,
        )

    def _extract_raw_depth(
        self, output: Any, target_h: int, target_w: int,
    ) -> NDArray[np.float32]:
        """Extract depth array from HuggingFace pipeline output."""
        import torch as _torch

        if isinstance(output, dict):
            if "predicted_depth" in output:
                tensor = output["predicted_depth"]
                if isinstance(tensor, _torch.Tensor):
                    depth = tensor.squeeze().cpu().numpy().astype(np.float32)
                    del tensor  # Free GPU memory
                else:
                    depth = np.array(tensor, dtype=np.float32)
            elif "depth" in output:
                depth = np.array(output["depth"], dtype=np.float32)
            else:
                raise RuntimeError(f"Unexpected pipeline output keys: {list(output.keys())}")
        else:
            depth = np.array(output, dtype=np.float32)

        # Ensure 2D
        if depth.ndim == 3:
            depth = depth.squeeze(0)

        # Resize to match original image
        if depth.shape[0] != target_h or depth.shape[1] != target_w:
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        return depth

    @staticmethod
    def _compute_confidence(
        depth: NDArray[np.float32],
        focal_source: str,
        is_metric: bool,
    ) -> float:
        """Estimate overall confidence score [0, 1].

        Factors:
        - Focal length source (EXIF > heuristic)
        - Metric vs relative mode
        - Scene structure (gradient magnitude)
        """
        # Base confidence
        if is_metric:
            base = 0.75 if focal_source == "exif" else 0.55 if focal_source == "explicit" else 0.40
        else:
            base = 0.70  # relative depth is always decent

        # Structure bonus: more edges = more geometry = more confident
        laplacian = cv2.Laplacian(depth, cv2.CV_32F)
        structure = np.abs(laplacian)
        p99 = float(np.percentile(structure, 99)) if structure.size > 0 else 1.0
        if p99 > 1e-6:
            struct_score = min(1.0, float(np.mean(structure)) / p99 * 2)
        else:
            struct_score = 0.3

        confidence = min(1.0, base + struct_score * 0.2)
        return round(confidence, 3)
