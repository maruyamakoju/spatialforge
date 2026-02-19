"""Inspection pipeline — combines depth estimation + defect detection.

This is the main entry point for railway track inspection.
It orchestrates:
  1. Frame extraction from video
  2. Depth estimation (DepthEngine)
  3. Defect detection (DefectEngine)
  4. Cross-frame defect tracking and deduplication
  5. Severity assessment with depth context
  6. Structured result aggregation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..models.inspection import (
    BBox2D,
    DefectClass,
    DetectedDefect,
    FrameInspection,
    InspectionReportSummary,
    Severity,
)
from .defect_engine import DefectEngine, DefectDetectionResult, Detection

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .depth_engine import DepthEngine

logger = logging.getLogger(__name__)


@dataclass
class InspectionConfig:
    """Configuration for the inspection pipeline."""

    # Frame sampling
    sample_fps: float = 3.0           # Frames per second to analyze
    max_frames: int = 500             # Maximum frames to process

    # Detection
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    enable_depth: bool = True         # Run depth estimation for each frame

    # Deduplication
    dedup_iou_threshold: float = 0.5  # IoU threshold for cross-frame dedup
    dedup_time_window_s: float = 2.0  # Time window for dedup

    # Output
    annotate_frames: bool = True      # Draw detections on frames
    save_annotated: bool = False      # Save annotated frames to disk


@dataclass
class InspectionResult:
    """Complete inspection result for a video or image set."""

    frames: list[FrameInspection] = field(default_factory=list)
    unique_defects: list[DetectedDefect] = field(default_factory=list)
    annotated_frames: list[NDArray[np.uint8]] = field(default_factory=list)
    total_processing_time_ms: float = 0.0
    video_duration_s: float = 0.0
    video_fps: float = 0.0
    total_frames_in_video: int = 0
    frames_analyzed: int = 0

    @property
    def summary(self) -> InspectionReportSummary:
        """Generate summary statistics."""
        severity_breakdown: dict[str, int] = {s.value: 0 for s in Severity}
        class_breakdown: dict[str, int] = {c.value: 0 for c in DefectClass}

        for defect in self.unique_defects:
            severity_breakdown[defect.severity.value] = (
                severity_breakdown.get(defect.severity.value, 0) + 1
            )
            class_breakdown[defect.defect_class.value] = (
                class_breakdown.get(defect.defect_class.value, 0) + 1
            )

        # Remove zero-count entries for cleaner output
        severity_breakdown = {k: v for k, v in severity_breakdown.items() if v > 0}
        class_breakdown = {k: v for k, v in class_breakdown.items() if v > 0}

        total_defects = sum(len(f.defects) for f in self.frames)

        return InspectionReportSummary(
            total_frames=self.frames_analyzed,
            total_defects=total_defects,
            unique_defects=len(self.unique_defects),
            severity_breakdown=severity_breakdown,
            class_breakdown=class_breakdown,
            coverage_km=None,  # Requires GPS data
            inspection_duration_s=self.total_processing_time_ms / 1000,
        )


class InspectionPipeline:
    """End-to-end railway track inspection pipeline.

    Usage:
        pipeline = InspectionPipeline(
            defect_engine=DefectEngine("models/railscan-yolov8m.pt"),
            depth_engine=depth_engine,
        )

        # Single image
        result = pipeline.inspect_image(image_rgb)

        # Video
        result = pipeline.inspect_video("track_footage.mp4")
    """

    def __init__(
        self,
        defect_engine: DefectEngine,
        depth_engine: DepthEngine | None = None,
        config: InspectionConfig | None = None,
    ) -> None:
        self._defect = defect_engine
        self._depth = depth_engine
        self._config = config or InspectionConfig()

    def inspect_image(
        self,
        image_rgb: NDArray[np.uint8],
        km_marker: float | None = None,
    ) -> InspectionResult:
        """Run inspection on a single image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.
            km_marker: Optional kilometer marker for location tracking.

        Returns:
            InspectionResult with one frame.
        """
        t0 = time.perf_counter()

        # Run depth estimation
        depth_map = None
        depth_min = None
        depth_max = None
        if self._depth and self._config.enable_depth:
            depth_result = self._depth.estimate(image_rgb)
            depth_map = depth_result.depth_map
            depth_min = depth_result.min_depth
            depth_max = depth_result.max_depth

        # Run defect detection
        det_result = self._defect.detect(image_rgb, depth_map)

        # Convert to API models
        defects = self._detections_to_models(det_result.detections, depth_map)

        frame = FrameInspection(
            frame_index=0,
            timestamp_s=0.0,
            km_marker=km_marker,
            defects=defects,
            depth_min_m=depth_min,
            depth_max_m=depth_max,
        )

        # Annotate
        annotated = []
        if self._config.annotate_frames and det_result.detections:
            annotated_img = self._defect.annotate_image(image_rgb, det_result.detections)
            annotated.append(annotated_img)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InspectionResult(
            frames=[frame],
            unique_defects=defects,  # Single frame, no dedup needed
            annotated_frames=annotated,
            total_processing_time_ms=elapsed_ms,
            frames_analyzed=1,
        )

    def inspect_video(
        self,
        video_path: str | Path,
        km_start: float | None = None,
        km_end: float | None = None,
        progress_callback: callable | None = None,
    ) -> InspectionResult:
        """Run inspection on a video file.

        Samples frames at the configured FPS rate, runs detection + depth
        on each, deduplicates detections across frames.

        Args:
            video_path: Path to video file.
            km_start: Starting kilometer marker.
            km_end: Ending kilometer marker.
            progress_callback: Optional callback(current_frame, total_frames).

        Returns:
            InspectionResult with all frames and deduplicated defects.
        """
        t0 = time.perf_counter()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_s = total_frames / video_fps if video_fps > 0 else 0

            # Calculate frame sampling interval
            sample_interval = max(1, int(video_fps / self._config.sample_fps))
            frames_to_analyze = min(
                total_frames // sample_interval,
                self._config.max_frames,
            )

            # KM interpolation
            km_per_frame = None
            if km_start is not None and km_end is not None and frames_to_analyze > 1:
                km_per_frame = (km_end - km_start) / (frames_to_analyze - 1)

            all_frames: list[FrameInspection] = []
            all_annotated: list[NDArray[np.uint8]] = []
            all_raw_detections: list[tuple[float, list[Detection]]] = []

            frame_idx = 0
            analyzed = 0

            while cap.isOpened() and analyzed < self._config.max_frames:
                ret, bgr_frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    timestamp_s = frame_idx / video_fps

                    # Depth estimation
                    depth_map = None
                    depth_min = None
                    depth_max = None
                    if self._depth and self._config.enable_depth:
                        depth_result = self._depth.estimate(rgb_frame)
                        depth_map = depth_result.depth_map
                        depth_min = depth_result.min_depth
                        depth_max = depth_result.max_depth

                    # Defect detection
                    det_result = self._defect.detect(rgb_frame, depth_map)

                    # Convert to models
                    defects = self._detections_to_models(det_result.detections, depth_map)

                    # KM marker
                    km = None
                    if km_start is not None and km_per_frame is not None:
                        km = km_start + analyzed * km_per_frame
                    elif km_start is not None:
                        km = km_start

                    frame_inspection = FrameInspection(
                        frame_index=frame_idx,
                        timestamp_s=round(timestamp_s, 3),
                        km_marker=round(km, 3) if km is not None else None,
                        defects=defects,
                        depth_min_m=depth_min,
                        depth_max_m=depth_max,
                    )
                    all_frames.append(frame_inspection)

                    # Track raw detections for dedup
                    all_raw_detections.append((timestamp_s, det_result.detections))

                    # Annotate
                    if self._config.annotate_frames and det_result.detections:
                        annotated_img = self._defect.annotate_image(
                            rgb_frame, det_result.detections,
                        )
                        all_annotated.append(annotated_img)

                    analyzed += 1

                    if progress_callback:
                        progress_callback(analyzed, frames_to_analyze)

                frame_idx += 1

        finally:
            cap.release()

        # Deduplicate detections across frames
        unique_defects = self._deduplicate_detections(all_frames)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return InspectionResult(
            frames=all_frames,
            unique_defects=unique_defects,
            annotated_frames=all_annotated,
            total_processing_time_ms=elapsed_ms,
            video_duration_s=duration_s,
            video_fps=video_fps,
            total_frames_in_video=total_frames,
            frames_analyzed=analyzed,
        )

    def _detections_to_models(
        self,
        detections: list[Detection],
        depth_map: NDArray[np.float32] | None,
    ) -> list[DetectedDefect]:
        """Convert raw Detection objects to API DetectedDefect models."""
        result = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1

            # Get depth at defect center
            depth_m = None
            if depth_map is not None:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                dh, dw = depth_map.shape[:2]
                cx = min(max(cx, 0), dw - 1)
                cy = min(max(cy, 0), dh - 1)
                depth_m = round(float(depth_map[cy, cx]), 3)

            # Map severity string to enum
            try:
                severity = Severity(det.severity)
            except ValueError:
                severity = Severity.INFO

            # Map class name to enum
            try:
                defect_class = DefectClass(det.class_name)
            except ValueError:
                logger.warning("Unknown defect class: %s", det.class_name)
                continue

            result.append(DetectedDefect(
                defect_class=defect_class,
                confidence=det.confidence,
                severity=severity,
                bbox=BBox2D(x=round(x1, 1), y=round(y1, 1), w=round(w, 1), h=round(h, 1)),
                depth_m=depth_m,
                description=self._generate_description(defect_class, severity, det.confidence),
            ))

        return result

    @staticmethod
    def _generate_description(
        defect_class: DefectClass,
        severity: Severity,
        confidence: float,
    ) -> str:
        """Generate human-readable defect description."""
        from ..models.inspection import DEFECT_LABELS_JA, SEVERITY_LABELS_JA

        ja_class = DEFECT_LABELS_JA.get(defect_class, defect_class.value)
        ja_severity = SEVERITY_LABELS_JA.get(severity, severity.value)

        return f"{ja_class}を検知（{ja_severity}・信頼度{confidence:.0%}）"

    def _deduplicate_detections(
        self,
        frames: list[FrameInspection],
    ) -> list[DetectedDefect]:
        """Deduplicate detections across frames using IoU and time window.

        Defects that appear in consecutive frames at similar positions
        are merged into a single unique defect with the highest confidence.
        """
        if not frames:
            return []

        # Collect all detections with their timestamps
        all_defects: list[tuple[float, DetectedDefect]] = []
        for frame in frames:
            for defect in frame.defects:
                all_defects.append((frame.timestamp_s, defect))

        if not all_defects:
            return []

        # Group by class, then merge overlapping detections within time window
        from collections import defaultdict
        by_class: dict[DefectClass, list[tuple[float, DetectedDefect]]] = defaultdict(list)
        for ts, d in all_defects:
            by_class[d.defect_class].append((ts, d))

        unique: list[DetectedDefect] = []

        for cls, detections in by_class.items():
            # Sort by timestamp
            detections.sort(key=lambda x: x[0])

            merged: list[DetectedDefect] = []
            for ts, det in detections:
                # Check if this detection overlaps with any existing merged detection
                found_match = False
                for i, existing in enumerate(merged):
                    if self._bbox_iou(det.bbox, existing.bbox) > self._config.dedup_iou_threshold:
                        # Keep the one with higher confidence
                        if det.confidence > existing.confidence:
                            merged[i] = det
                        found_match = True
                        break

                if not found_match:
                    merged.append(det)

            unique.extend(merged)

        # Sort by severity (critical first) then confidence
        severity_order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2, Severity.INFO: 3}
        unique.sort(key=lambda d: (severity_order.get(d.severity, 9), -d.confidence))

        return unique

    @staticmethod
    def _bbox_iou(a: BBox2D, b: BBox2D) -> float:
        """Calculate IoU between two bounding boxes."""
        ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
        bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = a.w * a.h
        area_b = b.w * b.h
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
