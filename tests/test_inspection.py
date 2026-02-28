"""Tests for the railway track inspection system.

Tests:
  - DefectEngine: model loading, detection, annotation, severity
  - InspectionPipeline: single image + video processing, dedup
  - /v1/inspect API endpoint
  - Report generator
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from spatialforge.inference.defect_engine import (
    _DEFAULT_CLASS_MAP,
    DefectDetectionResult,
    DefectEngine,
    Detection,
)
from spatialforge.inference.inspector import (
    InspectionConfig,
    InspectionPipeline,
    InspectionResult,
)
from spatialforge.models.inspection import (
    DEFECT_LABELS_JA,
    SEVERITY_LABELS_JA,
    BBox2D,
    DefectClass,
    DetectedDefect,
    FrameInspection,
    InspectionReportSummary,
    Severity,
)

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_rgb():
    """100x100 random RGB image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_depth():
    """100x100 random depth map."""
    return np.random.rand(100, 100).astype(np.float32) * 10


@pytest.fixture
def mock_detections():
    """Sample list of Detection objects."""
    return [
        Detection(
            class_id=0,
            class_name="rail_crack",
            confidence=0.85,
            bbox_xyxy=(10.0, 20.0, 50.0, 60.0),
            severity="critical",
        ),
        Detection(
            class_id=4,
            class_name="fastener_missing",
            confidence=0.72,
            bbox_xyxy=(60.0, 30.0, 80.0, 50.0),
            severity="major",
        ),
        Detection(
            class_id=1,
            class_name="rail_wear",
            confidence=0.45,
            bbox_xyxy=(20.0, 70.0, 40.0, 90.0),
            severity="info",
        ),
    ]


# ── DefectEngine unit tests ──────────────────────────────────────


class TestDefectEngine:
    """Tests for DefectEngine."""

    def test_init_default(self):
        """Engine initializes with default settings."""
        engine = DefectEngine()
        assert engine._conf_threshold == 0.25
        assert engine._iou_threshold == 0.45
        assert engine._class_map == _DEFAULT_CLASS_MAP

    def test_init_custom(self):
        """Engine initializes with custom settings."""
        engine = DefectEngine(
            model_path="custom.pt",
            confidence_threshold=0.5,
            iou_threshold=0.6,
            class_map=["crack", "wear"],
        )
        assert engine._conf_threshold == 0.5
        assert engine._class_map == ["crack", "wear"]

    def test_severity_critical_class_high_confidence(self):
        """Critical class + high confidence → critical severity."""
        severity = DefectEngine._assess_severity(
            "rail_crack", 0.9, (10, 10, 50, 50), None, 100, 100,
        )
        assert severity == "critical"

    def test_severity_critical_class_low_confidence(self):
        """Critical class + low confidence → minor severity."""
        severity = DefectEngine._assess_severity(
            "rail_crack", 0.15, (10, 10, 20, 20), None, 100, 100,
        )
        assert severity == "minor"

    def test_severity_major_class(self):
        """Major class + high confidence → major severity."""
        severity = DefectEngine._assess_severity(
            "rail_spalling", 0.85, (10, 10, 60, 60), None, 100, 100,
        )
        assert severity == "major"

    def test_severity_with_depth(self, sample_depth):
        """Depth at defect center affects severity."""
        # Set a very close depth at the defect center
        sample_depth[35, 30] = 0.3  # Very close
        severity = DefectEngine._assess_severity(
            "rail_crack", 0.7, (10, 10, 50, 60), sample_depth, 100, 100,
        )
        assert severity == "critical"

    def test_annotate_image(self, sample_rgb, mock_detections):
        """Annotation draws boxes on image."""
        engine = DefectEngine()
        annotated = engine.annotate_image(sample_rgb, mock_detections)
        # Must be same shape, but different pixels
        assert annotated.shape == sample_rgb.shape
        assert not np.array_equal(annotated, sample_rgb)

    def test_annotate_empty_detections(self, sample_rgb):
        """Annotation with no detections returns copy of original."""
        engine = DefectEngine()
        annotated = engine.annotate_image(sample_rgb, [])
        assert annotated.shape == sample_rgb.shape
        np.testing.assert_array_equal(annotated, sample_rgb)

    def test_is_fine_tuned_false_when_no_model(self):
        """_is_fine_tuned returns False when model not loaded."""
        engine = DefectEngine()
        assert not engine._is_fine_tuned()

    def test_coco_to_defect_returns_none(self):
        """COCO mapping returns None (skip all COCO classes)."""
        result = DefectEngine._coco_to_defect_class(0, {0: "person"})
        assert result is None


# ── Data model tests ───────────────────────────────────────────


class TestInspectionModels:
    """Tests for inspection data models."""

    def test_defect_class_enum(self):
        """All 12 defect classes are defined (11 surface + 1 clearance)."""
        assert len(DefectClass) == 12
        assert DefectClass.RAIL_CRACK.value == "rail_crack"
        assert DefectClass.GAUGE_ANOMALY.value == "gauge_anomaly"
        assert DefectClass.CLEARANCE_VIOLATION.value == "clearance_violation"

    def test_severity_enum(self):
        """All 4 severity levels are defined."""
        assert len(Severity) == 4
        assert Severity.CRITICAL.value == "critical"

    def test_japanese_labels(self):
        """Japanese labels exist for all classes and severities."""
        for cls in DefectClass:
            assert cls in DEFECT_LABELS_JA
        for sev in Severity:
            assert sev in SEVERITY_LABELS_JA

    def test_detected_defect_model(self):
        """DetectedDefect creates with valid data."""
        d = DetectedDefect(
            defect_class=DefectClass.RAIL_CRACK,
            confidence=0.95,
            severity=Severity.CRITICAL,
            bbox=BBox2D(x=10, y=20, w=40, h=30),
            depth_m=2.5,
            description="レールき裂を検知",
        )
        assert d.defect_class == DefectClass.RAIL_CRACK
        assert d.confidence == 0.95

    def test_detected_defect_confidence_bounds(self):
        """Confidence must be 0-1."""
        with pytest.raises(ValidationError):
            DetectedDefect(
                defect_class=DefectClass.RAIL_CRACK,
                confidence=1.5,  # Invalid
                severity=Severity.CRITICAL,
                bbox=BBox2D(x=0, y=0, w=10, h=10),
            )

    def test_frame_inspection(self):
        """FrameInspection aggregates defects."""
        fi = FrameInspection(
            frame_index=42,
            timestamp_s=1.4,
            km_marker=123.5,
            defects=[
                DetectedDefect(
                    defect_class=DefectClass.FASTENER_MISSING,
                    confidence=0.8,
                    severity=Severity.MAJOR,
                    bbox=BBox2D(x=0, y=0, w=10, h=10),
                ),
            ],
        )
        assert fi.frame_index == 42
        assert len(fi.defects) == 1

    def test_inspection_report_summary(self):
        """Summary model validates correctly."""
        s = InspectionReportSummary(
            total_frames=100,
            total_defects=15,
            unique_defects=8,
            severity_breakdown={"critical": 2, "major": 3, "minor": 3},
            class_breakdown={"rail_crack": 2, "fastener_missing": 3},
            coverage_km=5.2,
            inspection_duration_s=45.3,
        )
        assert s.total_frames == 100
        assert s.unique_defects == 8


# ── InspectionPipeline tests ────────────────────────────────────


class TestInspectionPipeline:
    """Tests for InspectionPipeline."""

    def test_inspect_image_no_detections(self, sample_rgb):
        """Pipeline handles images with no detections."""
        mock_defect = MagicMock()
        mock_defect.detect.return_value = DefectDetectionResult(
            detections=[],
            frame_shape=(100, 100),
            processing_time_ms=10.0,
        )

        pipeline = InspectionPipeline(
            defect_engine=mock_defect,
            depth_engine=None,
            config=InspectionConfig(enable_depth=False),
        )

        result = pipeline.inspect_image(sample_rgb)
        assert isinstance(result, InspectionResult)
        assert len(result.frames) == 1
        assert len(result.frames[0].defects) == 0
        assert len(result.unique_defects) == 0

    def test_inspect_image_with_detections(self, sample_rgb, mock_detections):
        """Pipeline processes detections correctly."""
        mock_defect = MagicMock()
        mock_defect.detect.return_value = DefectDetectionResult(
            detections=mock_detections,
            frame_shape=(100, 100),
            processing_time_ms=50.0,
        )
        mock_defect.annotate_image.return_value = sample_rgb.copy()

        pipeline = InspectionPipeline(
            defect_engine=mock_defect,
            depth_engine=None,
            config=InspectionConfig(enable_depth=False),
        )

        result = pipeline.inspect_image(sample_rgb, km_marker=42.5)
        assert len(result.frames) == 1
        assert len(result.unique_defects) == 3
        assert result.frames[0].km_marker == 42.5

    def test_inspect_image_with_depth(self, sample_rgb, sample_depth):
        """Pipeline uses depth engine when available."""
        mock_defect = MagicMock()
        mock_defect.detect.return_value = DefectDetectionResult(
            detections=[], frame_shape=(100, 100), processing_time_ms=10.0,
        )

        mock_depth_engine = MagicMock()
        mock_depth_result = MagicMock()
        mock_depth_result.depth_map = sample_depth
        mock_depth_result.min_depth = 0.5
        mock_depth_result.max_depth = 10.0
        mock_depth_engine.estimate.return_value = mock_depth_result

        pipeline = InspectionPipeline(
            defect_engine=mock_defect,
            depth_engine=mock_depth_engine,
            config=InspectionConfig(enable_depth=True),
        )

        result = pipeline.inspect_image(sample_rgb)
        assert result.frames[0].depth_min_m == 0.5
        assert result.frames[0].depth_max_m == 10.0
        assert result.preview_depth_map is not None
        assert result.preview_depth_map.shape == sample_depth.shape
        mock_depth_engine.estimate.assert_called_once()

    def test_summary_generation(self, sample_rgb, mock_detections):
        """Summary statistics are computed correctly."""
        mock_defect = MagicMock()
        mock_defect.detect.return_value = DefectDetectionResult(
            detections=mock_detections, frame_shape=(100, 100), processing_time_ms=50.0,
        )
        mock_defect.annotate_image.return_value = sample_rgb.copy()

        pipeline = InspectionPipeline(
            defect_engine=mock_defect,
            config=InspectionConfig(enable_depth=False),
        )

        result = pipeline.inspect_image(sample_rgb)
        summary = result.summary
        assert summary.total_frames == 1
        assert summary.total_defects == 3
        assert summary.unique_defects == 3

    def test_bbox_iou_identical(self):
        """IoU of identical boxes is 1.0."""
        a = BBox2D(x=10, y=10, w=50, h=50)
        iou = InspectionPipeline._bbox_iou(a, a)
        assert abs(iou - 1.0) < 1e-6

    def test_bbox_iou_no_overlap(self):
        """IoU of non-overlapping boxes is 0.0."""
        a = BBox2D(x=0, y=0, w=10, h=10)
        b = BBox2D(x=100, y=100, w=10, h=10)
        iou = InspectionPipeline._bbox_iou(a, b)
        assert iou == 0.0

    def test_bbox_iou_partial(self):
        """IoU of partially overlapping boxes is between 0 and 1."""
        a = BBox2D(x=0, y=0, w=20, h=20)
        b = BBox2D(x=10, y=10, w=20, h=20)
        iou = InspectionPipeline._bbox_iou(a, b)
        assert 0 < iou < 1

    def test_dedup_identical_detections(self):
        """Identical detections across frames are deduplicated."""
        defect = DetectedDefect(
            defect_class=DefectClass.RAIL_CRACK,
            confidence=0.85,
            severity=Severity.CRITICAL,
            bbox=BBox2D(x=10, y=20, w=40, h=30),
        )

        frames = [
            FrameInspection(frame_index=0, timestamp_s=0.0, defects=[defect]),
            FrameInspection(frame_index=1, timestamp_s=0.33, defects=[defect]),
            FrameInspection(frame_index=2, timestamp_s=0.67, defects=[defect]),
        ]

        pipeline = InspectionPipeline(
            defect_engine=MagicMock(),
            config=InspectionConfig(enable_depth=False),
        )

        unique = pipeline._deduplicate_detections(frames)
        assert len(unique) == 1
        assert unique[0].defect_class == DefectClass.RAIL_CRACK

    def test_dedup_different_locations(self):
        """Detections at different locations are NOT deduplicated."""
        d1 = DetectedDefect(
            defect_class=DefectClass.RAIL_CRACK,
            confidence=0.85,
            severity=Severity.CRITICAL,
            bbox=BBox2D(x=10, y=20, w=40, h=30),
        )
        d2 = DetectedDefect(
            defect_class=DefectClass.RAIL_CRACK,
            confidence=0.75,
            severity=Severity.MAJOR,
            bbox=BBox2D(x=200, y=300, w=40, h=30),
        )

        frames = [
            FrameInspection(frame_index=0, timestamp_s=0.0, defects=[d1]),
            FrameInspection(frame_index=1, timestamp_s=0.33, defects=[d2]),
        ]

        pipeline = InspectionPipeline(
            defect_engine=MagicMock(),
            config=InspectionConfig(enable_depth=False),
        )

        unique = pipeline._deduplicate_detections(frames)
        assert len(unique) == 2


# ── Report generator tests ────────────────────────────────────


class TestReportGenerator:
    """Tests for HTML report generation."""

    def test_generate_empty_report(self):
        """Report generates for inspection with no defects."""
        from spatialforge.report.generator import generate_html_report

        result = InspectionResult(
            frames=[FrameInspection(frame_index=0, timestamp_s=0.0, defects=[])],
            unique_defects=[],
            total_processing_time_ms=100.0,
            frames_analyzed=1,
        )

        html = generate_html_report(result, route_name="テスト路線")
        assert "軌道点検レポート" in html
        assert "テスト路線" in html
        assert "異常は検知されませんでした" in html

    def test_generate_report_with_defects(self):
        """Report includes defect details."""
        from spatialforge.report.generator import generate_html_report

        defect = DetectedDefect(
            defect_class=DefectClass.RAIL_CRACK,
            confidence=0.92,
            severity=Severity.CRITICAL,
            bbox=BBox2D(x=10, y=20, w=40, h=30),
            depth_m=2.5,
            description="レールき裂を検知（緊急・信頼度92%）",
        )

        result = InspectionResult(
            frames=[FrameInspection(frame_index=0, timestamp_s=0.0, defects=[defect])],
            unique_defects=[defect],
            total_processing_time_ms=500.0,
            frames_analyzed=10,
        )

        html = generate_html_report(result)
        assert "レールき裂" in html
        assert "緊急" in html
        assert "92%" in html

    def test_report_is_valid_html(self):
        """Report is valid HTML with required elements."""
        from spatialforge.report.generator import generate_html_report

        result = InspectionResult(
            frames=[], unique_defects=[],
            total_processing_time_ms=0, frames_analyzed=0,
        )

        html = generate_html_report(result)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<style>" in html
        assert "@media print" in html

    def test_report_compliance_note(self):
        """Report includes compliance/disclaimer note."""
        from spatialforge.report.generator import generate_html_report

        result = InspectionResult(
            frames=[], unique_defects=[],
            total_processing_time_ms=0, frames_analyzed=0,
        )

        html = generate_html_report(result)
        assert "資格を有する保線担当者" in html
        assert "鉄道事業法" in html
