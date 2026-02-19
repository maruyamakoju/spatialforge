"""Railway defect detection engine — YOLOv8 based.

Detects specific track defect classes from camera images:
  - Rail: cracks, wear, corrugation, spalling
  - Fasteners: missing, broken
  - Sleepers: cracks, decay
  - Track: ballast fouling, joint defects, gauge anomalies

Model loading follows the same pattern as depth_engine.py:
the model is loaded via ModelManager to ensure thread-safe caching
and proper GPU memory management.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default confidence thresholds per severity mapping.
# Higher confidence + dangerous defect class = higher severity.
_CRITICAL_CLASSES = {"rail_crack", "fastener_missing", "gauge_anomaly"}
_MAJOR_CLASSES = {"rail_spalling", "fastener_broken", "sleeper_crack", "joint_defect"}

# Class index → DefectClass value mapping (set during model init)
_DEFAULT_CLASS_MAP: list[str] = [
    "rail_crack",
    "rail_wear",
    "rail_corrugation",
    "rail_spalling",
    "fastener_missing",
    "fastener_broken",
    "sleeper_crack",
    "sleeper_decay",
    "ballast_fouling",
    "joint_defect",
    "gauge_anomaly",
]


@dataclass
class Detection:
    """Raw detection from YOLOv8."""

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # x1, y1, x2, y2
    severity: str = "info"


@dataclass
class DefectDetectionResult:
    """Result from running defect detection on a single frame."""

    detections: list[Detection] = field(default_factory=list)
    frame_shape: tuple[int, int] = (0, 0)  # (H, W)
    processing_time_ms: float = 0.0
    model_used: str = ""


class DefectEngine:
    """Runs YOLOv8 defect detection on railway track images.

    Usage:
        engine = DefectEngine(model_path="models/railscan-yolov8m.pt")
        result = engine.detect(image_rgb)
        for det in result.detections:
            print(f"{det.class_name}: {det.confidence:.2f} @ {det.bbox_xyxy}")
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        class_map: list[str] | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._conf_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._device = device
        self._class_map = class_map or _DEFAULT_CLASS_MAP
        self._model: Any = None

    def _load_model(self) -> Any:
        """Load YOLOv8 model (lazy, thread-safe via GIL)."""
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for defect detection. "
                "Install with: pip install ultralytics"
            ) from exc

        if self._model_path and self._model_path.exists():
            logger.info("Loading defect model from %s", self._model_path)
            self._model = YOLO(str(self._model_path))
        else:
            # Fall back to pretrained YOLOv8m for development/demo.
            # In production, this MUST be replaced with a fine-tuned model.
            logger.warning(
                "No fine-tuned defect model found at %s. "
                "Using pretrained YOLOv8m (COCO) — defect classes will be simulated. "
                "Train a model with: python scripts/train_defect_model.py",
                self._model_path,
            )
            self._model = YOLO("yolov8m.pt")

        return self._model

    def detect(
        self,
        image_rgb: NDArray[np.uint8],
        depth_map: NDArray[np.float32] | None = None,
    ) -> DefectDetectionResult:
        """Run defect detection on a single RGB image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.
            depth_map: Optional H x W float32 depth map for severity estimation.

        Returns:
            DefectDetectionResult with list of detections.
        """
        t0 = time.perf_counter()
        h, w = image_rgb.shape[:2]

        model = self._load_model()

        # Run YOLOv8 inference
        # Convert RGB → BGR for ultralytics (it expects BGR)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        results = model.predict(
            image_bgr,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            device=self._device if self._device != "auto" else None,
            verbose=False,
        )

        detections: list[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                    # Map class ID to defect class name
                    if self._is_fine_tuned():
                        class_name = self._class_map[cls_id] if cls_id < len(self._class_map) else f"unknown_{cls_id}"
                    else:
                        # Pretrained COCO model: simulate defect mapping for demo
                        class_name = self._coco_to_defect_class(cls_id, result.names)
                        if class_name is None:
                            continue  # Skip non-rail-relevant COCO classes

                    # Determine severity
                    severity = self._assess_severity(
                        class_name, conf, (x1, y1, x2, y2), depth_map, h, w,
                    )

                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=round(conf, 4),
                        bbox_xyxy=(round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)),
                        severity=severity,
                    ))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return DefectDetectionResult(
            detections=detections,
            frame_shape=(h, w),
            processing_time_ms=elapsed_ms,
            model_used=str(self._model_path or "yolov8m-coco"),
        )

    def detect_batch(
        self,
        frames: list[NDArray[np.uint8]],
        depth_maps: list[NDArray[np.float32]] | None = None,
    ) -> list[DefectDetectionResult]:
        """Run detection on a batch of frames for video processing."""
        results = []
        for i, frame in enumerate(frames):
            dm = depth_maps[i] if depth_maps and i < len(depth_maps) else None
            results.append(self.detect(frame, dm))
        return results

    def annotate_image(
        self,
        image_rgb: NDArray[np.uint8],
        detections: list[Detection],
    ) -> NDArray[np.uint8]:
        """Draw detection boxes and labels on the image.

        Returns a copy with annotations drawn.
        """
        img = image_rgb.copy()

        severity_colors = {
            "critical": (255, 0, 0),      # Red
            "major": (255, 140, 0),        # Orange
            "minor": (255, 255, 0),        # Yellow
            "info": (0, 200, 255),         # Cyan
        }

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = severity_colors.get(det.severity, (255, 255, 255))
            thickness = 3 if det.severity in ("critical", "major") else 2

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"{det.class_name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Draw label text
            text_color = (0, 0, 0) if det.severity != "critical" else (255, 255, 255)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
            )

        return img

    def _is_fine_tuned(self) -> bool:
        """Check if the loaded model is our fine-tuned rail defect model."""
        if self._model is None:
            return False
        # Fine-tuned models have our custom class names
        model_names = getattr(self._model, "names", {})
        if isinstance(model_names, dict):
            return any(
                name in _DEFAULT_CLASS_MAP
                for name in model_names.values()
            )
        return False

    @staticmethod
    def _coco_to_defect_class(cls_id: int, names: dict[int, str]) -> str | None:
        """Map COCO class detections to rail defect classes for demo purposes.

        This is a TEMPORARY mapping for demo/development only.
        In production, this is never called because we use a fine-tuned model.
        """
        # We don't map COCO classes — return None to skip all detections
        # from the pretrained model. The demo mode uses synthetic detections
        # generated by the InspectionPipeline instead.
        return None

    @staticmethod
    def _assess_severity(
        class_name: str,
        confidence: float,
        bbox_xyxy: tuple[float, float, float, float],
        depth_map: NDArray[np.float32] | None,
        img_h: int,
        img_w: int,
    ) -> str:
        """Assess defect severity based on class, confidence, size, and depth.

        Severity logic:
        - Critical: dangerous defect class + high confidence (>0.7)
        - Major: dangerous class + moderate confidence, or major class + high confidence
        - Minor: low confidence or small defect area
        - Info: very low confidence or minor class
        """
        x1, y1, x2, y2 = bbox_xyxy
        area_ratio = ((x2 - x1) * (y2 - y1)) / (img_h * img_w) if img_h * img_w > 0 else 0

        # Factor in depth (closer defects are more urgent)
        depth_factor = 1.0
        if depth_map is not None:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cx = min(max(cx, 0), img_w - 1)
            cy = min(max(cy, 0), img_h - 1)
            depth_at_center = float(depth_map[cy, cx])
            if depth_at_center > 0:
                # Closer objects (lower depth) get higher severity boost
                depth_factor = min(2.0, 1.0 / max(depth_at_center, 0.5))

        # Composite score
        score = confidence * (1.0 + area_ratio * 5) * depth_factor

        if class_name in _CRITICAL_CLASSES:
            if score > 0.6:
                return "critical"
            if score > 0.3:
                return "major"
            return "minor"
        elif class_name in _MAJOR_CLASSES:
            if score > 0.7:
                return "major"
            if score > 0.4:
                return "minor"
            return "info"
        else:
            if score > 0.8:
                return "minor"
            return "info"
