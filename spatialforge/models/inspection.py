"""Data models for railway track inspection (defect detection + depth)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ── Defect taxonomy ──────────────────────────────────────────
# Aligned with Japanese railway inspection standards (保線規程).
# These are the primary visual defect classes detectable from
# track-level camera footage.


class DefectClass(str, Enum):
    """Railway track defect categories."""

    RAIL_CRACK = "rail_crack"              # レールき裂
    RAIL_WEAR = "rail_wear"                # レール摩耗
    RAIL_CORRUGATION = "rail_corrugation"  # 波状磨耗
    RAIL_SPALLING = "rail_spalling"        # レール剥離
    FASTENER_MISSING = "fastener_missing"  # 締結装置欠損
    FASTENER_BROKEN = "fastener_broken"    # 締結装置破損
    SLEEPER_CRACK = "sleeper_crack"        # まくら木き裂
    SLEEPER_DECAY = "sleeper_decay"        # まくら木腐食
    BALLAST_FOULING = "ballast_fouling"    # 道床汚損
    JOINT_DEFECT = "joint_defect"          # 継目板不良
    GAUGE_ANOMALY = "gauge_anomaly"        # 軌間異常


# Japanese display names (for reports)
DEFECT_LABELS_JA: dict[DefectClass, str] = {
    DefectClass.RAIL_CRACK: "レールき裂",
    DefectClass.RAIL_WEAR: "レール摩耗",
    DefectClass.RAIL_CORRUGATION: "波状磨耗",
    DefectClass.RAIL_SPALLING: "レール剥離",
    DefectClass.FASTENER_MISSING: "締結装置欠損",
    DefectClass.FASTENER_BROKEN: "締結装置破損",
    DefectClass.SLEEPER_CRACK: "まくら木き裂",
    DefectClass.SLEEPER_DECAY: "まくら木腐食",
    DefectClass.BALLAST_FOULING: "道床汚損",
    DefectClass.JOINT_DEFECT: "継目板不良",
    DefectClass.GAUGE_ANOMALY: "軌間異常",
}


class Severity(str, Enum):
    """Defect severity level (maps to maintenance urgency)."""

    CRITICAL = "critical"    # 緊急: 即時対応が必要
    MAJOR = "major"          # 重要: 次回定期点検までに対応
    MINOR = "minor"          # 軽微: 経過観察
    INFO = "info"            # 参考: 記録のみ


SEVERITY_LABELS_JA: dict[Severity, str] = {
    Severity.CRITICAL: "緊急",
    Severity.MAJOR: "重要",
    Severity.MINOR: "軽微",
    Severity.INFO: "参考",
}


# ── Detection result (single defect instance) ────────────────


class BBox2D(BaseModel):
    """Bounding box in pixel coordinates."""

    x: float = Field(..., description="Top-left X")
    y: float = Field(..., description="Top-left Y")
    w: float = Field(..., description="Width")
    h: float = Field(..., description="Height")


class DetectedDefect(BaseModel):
    """Single detected defect instance."""

    defect_class: DefectClass = Field(..., description="Defect category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    severity: Severity = Field(..., description="Assessed severity")
    bbox: BBox2D = Field(..., description="Bounding box in the source frame")
    depth_m: float | None = Field(None, description="Estimated depth at defect center (meters)")
    description: str = Field("", description="Human-readable description")


# ── Per-frame inspection result ──────────────────────────────


class FrameInspection(BaseModel):
    """Inspection result for a single video frame."""

    frame_index: int = Field(..., description="Frame number in the video")
    timestamp_s: float = Field(..., description="Timestamp in seconds")
    km_marker: float | None = Field(None, description="Kilometer post (if GPS available)")
    defects: list[DetectedDefect] = Field(default_factory=list)
    depth_min_m: float | None = Field(None, description="Min depth in frame")
    depth_max_m: float | None = Field(None, description="Max depth in frame")


# ── API request / response models ────────────────────────────


class InspectModel(str, Enum):
    """Defect detection model size."""

    STANDARD = "standard"  # YOLOv8m — balanced
    FAST = "fast"          # YOLOv8n — edge/real-time
    PRECISE = "precise"    # YOLOv8x — maximum accuracy


class InspectResponse(BaseModel):
    """Single-image inspection result."""

    defects: list[DetectedDefect] = Field(..., description="Detected defects")
    defect_count: int = Field(..., description="Total defects found")
    severity_summary: dict[str, int] = Field(
        ..., description="Count per severity level",
        examples=[{"critical": 0, "major": 1, "minor": 2, "info": 0}],
    )
    annotated_image_url: str | None = Field(
        None, description="URL to image with detection overlays",
    )
    depth_map_url: str | None = Field(
        None, description="URL to depth visualization",
    )
    processing_time_ms: float = Field(..., description="Total processing time (ms)")


class InspectVideoJobResponse(BaseModel):
    """Async video inspection job submission."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field("processing", description="Job status")
    estimated_time_s: float | None = Field(None, description="Estimated completion time")


class InspectionReportSummary(BaseModel):
    """Summary statistics for a full inspection run."""

    total_frames: int = Field(..., description="Total frames analyzed")
    total_defects: int = Field(..., description="Total defects detected")
    unique_defects: int = Field(
        ..., description="Unique defects (deduplicated across frames)",
    )
    severity_breakdown: dict[str, int] = Field(
        ..., description="Defect count per severity",
    )
    class_breakdown: dict[str, int] = Field(
        ..., description="Defect count per class",
    )
    coverage_km: float | None = Field(None, description="Distance covered (km)")
    inspection_duration_s: float = Field(..., description="Total inspection time")


class InspectionReportResponse(BaseModel):
    """Full inspection report for a video."""

    job_id: str
    status: str
    summary: InspectionReportSummary | None = None
    frames: list[FrameInspection] | None = Field(
        None, description="Per-frame details (paginated)",
    )
    report_url: str | None = Field(
        None, description="URL to downloadable HTML/PDF report",
    )
    processing_time_ms: float | None = None
    error: str | None = None
