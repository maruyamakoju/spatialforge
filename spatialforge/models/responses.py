"""Response models for all API endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── /depth ──────────────────────────────────────────────────

class DepthMetadata(BaseModel):
    width: int
    height: int
    min_depth_m: float
    max_depth_m: float
    focal_length_px: float | None = None
    confidence_mean: float


class DepthResponse(BaseModel):
    depth_map_url: str
    colormap_url: str | None = None
    metadata: DepthMetadata
    processing_time_ms: float


# ── /pose ───────────────────────────────────────────────────

class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


class CameraPose(BaseModel):
    frame_index: int
    rotation: list[list[float]] = Field(..., description="3x3 rotation matrix")
    translation: list[float] = Field(..., description="3D translation vector")
    intrinsics: CameraIntrinsics


class PoseResponse(BaseModel):
    camera_poses: list[CameraPose]
    pointcloud_url: str | None = None
    num_frames: int
    processing_time_ms: float


# ── /reconstruct ────────────────────────────────────────────

class ReconstructStats(BaseModel):
    num_gaussians: int | None = None
    num_points: int | None = None
    num_vertices: int | None = None
    bounding_box: list[list[float]] | None = None
    camera_poses: list[CameraPose] | None = None


class ReconstructJobResponse(BaseModel):
    job_id: str
    status: str = "processing"
    estimated_time_s: float | None = None


class ReconstructResultResponse(BaseModel):
    job_id: str
    status: str
    scene_url: str | None = None
    viewer_url: str | None = None
    stats: ReconstructStats | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ── /measure ────────────────────────────────────────────────

class MeasureResponse(BaseModel):
    distance_m: float
    confidence: float
    depth_at_points: list[float]
    calibration_method: str
    processing_time_ms: float


# ── /floorplan ──────────────────────────────────────────────

class FloorplanJobResponse(BaseModel):
    job_id: str
    status: str = "processing"
    estimated_time_s: float | None = None


class FloorplanResultResponse(BaseModel):
    job_id: str
    status: str
    floorplan_url: str | None = None
    floor_area_m2: float | None = None
    room_count: int | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ── /segment-3d ─────────────────────────────────────────────

class BBox3D(BaseModel):
    min_point: list[float] = Field(..., description="[x, y, z] min corner")
    max_point: list[float] = Field(..., description="[x, y, z] max corner")


class SegmentedObject(BaseModel):
    label: str
    confidence: float
    mask_url: str | None = None
    bbox_3d: BBox3D | None = None


class Segment3DJobResponse(BaseModel):
    job_id: str
    status: str = "processing"
    estimated_time_s: float | None = None


class Segment3DResultResponse(BaseModel):
    job_id: str
    status: str
    objects: list[SegmentedObject] | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ── Common ──────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    gpu_available: bool
    models_loaded: list[str]


class ErrorResponse(BaseModel):
    detail: str
    error_code: str | None = None
