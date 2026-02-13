"""Response models for all API endpoints."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ── /depth ──────────────────────────────────────────────────


class DepthMetadata(BaseModel):
    width: int = Field(..., description="Image width in pixels", examples=[1920])
    height: int = Field(..., description="Image height in pixels", examples=[1080])
    min_depth_m: float = Field(
        ..., description="Minimum depth in meters", examples=[0.35]
    )
    max_depth_m: float = Field(
        ..., description="Maximum depth in meters", examples=[12.8]
    )
    focal_length_px: float | None = Field(
        None, description="Estimated focal length in pixels", examples=[525.0]
    )
    confidence_mean: float = Field(
        ..., description="Mean prediction confidence (0-1)", examples=[0.94]
    )


class DepthResponse(BaseModel):
    """Depth estimation result with downloadable depth map."""

    depth_map_url: str = Field(
        ...,
        description="URL to download the depth map (16-bit PNG, EXR, or NumPy)",
        examples=["https://storage.example.com/depth/abc123.png"],
    )
    colormap_url: str | None = Field(
        None,
        description="URL to colorized depth visualization (JPEG)",
        examples=["https://storage.example.com/depth_vis/abc123.jpg"],
    )
    metadata: DepthMetadata
    processing_time_ms: float = Field(
        ..., description="Server-side processing time in milliseconds", examples=[87.3]
    )


# ── /pose ───────────────────────────────────────────────────


class CameraIntrinsics(BaseModel):
    fx: float = Field(..., description="Focal length X (pixels)", examples=[525.0])
    fy: float = Field(..., description="Focal length Y (pixels)", examples=[525.0])
    cx: float = Field(
        ..., description="Principal point X (pixels)", examples=[320.0]
    )
    cy: float = Field(
        ..., description="Principal point Y (pixels)", examples=[240.0]
    )
    width: int = Field(..., description="Image width", examples=[640])
    height: int = Field(..., description="Image height", examples=[480])


class CameraPose(BaseModel):
    frame_index: int = Field(..., description="Frame number in the sequence")
    rotation: list[list[float]] = Field(
        ..., description="3x3 rotation matrix (world-to-camera)"
    )
    translation: list[float] = Field(
        ..., description="3D translation vector [x, y, z] in meters"
    )
    intrinsics: CameraIntrinsics


class PoseResponse(BaseModel):
    """Camera pose estimation result with per-frame poses."""

    camera_poses: list[CameraPose] = Field(
        ..., description="List of estimated camera poses"
    )
    pointcloud_url: str | None = Field(
        None, description="URL to sparse point cloud (if requested)"
    )
    num_frames: int = Field(..., description="Total number of frames processed")
    processing_time_ms: float = Field(
        ..., description="Server-side processing time in milliseconds"
    )


# ── /reconstruct ────────────────────────────────────────────


class ReconstructStats(BaseModel):
    num_gaussians: int | None = Field(
        None, description="Number of 3D Gaussians (for Gaussian splatting output)"
    )
    num_points: int | None = Field(
        None, description="Number of points (for point cloud output)"
    )
    num_vertices: int | None = Field(
        None, description="Number of vertices (for mesh output)"
    )
    bounding_box: list[list[float]] | None = Field(
        None, description="Scene bounding box [[min_x,min_y,min_z],[max_x,max_y,max_z]]"
    )
    camera_poses: list[CameraPose] | None = Field(
        None, description="Recovered camera poses"
    )


class AsyncJobState(str, Enum):
    """Stable async job lifecycle state."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class ReconstructJobResponse(BaseModel):
    """Async reconstruction job submission response."""

    job_id: str = Field(..., description="Unique job identifier for polling")
    status: str = Field(
        "processing", description="Job status: processing, complete, failed"
    )
    state: AsyncJobState = Field(
        AsyncJobState.PROCESSING,
        description="Stable job state enum for clients",
    )
    step: str | None = Field(
        None,
        description="Optional processing step when state is processing",
    )
    estimated_time_s: float | None = Field(
        None, description="Estimated completion time in seconds"
    )


class ReconstructResultResponse(BaseModel):
    """Reconstruction job result (returned when polling a complete job)."""

    job_id: str
    status: str
    state: AsyncJobState
    step: str | None = None
    scene_url: str | None = Field(
        None, description="URL to download the reconstructed 3D scene"
    )
    viewer_url: str | None = Field(
        None, description="URL to interactive 3D viewer"
    )
    stats: ReconstructStats | None = None
    processing_time_ms: float | None = None
    error: str | None = None


# ── /measure ────────────────────────────────────────────────


class MeasureResponse(BaseModel):
    """Real-world distance measurement between two image points."""

    distance_m: float = Field(
        ..., description="Measured distance in meters", examples=[2.45]
    )
    confidence: float = Field(
        ...,
        description="Measurement confidence (0-1). Higher is more reliable.",
        examples=[0.87],
    )
    depth_at_points: list[float] = Field(
        ..., description="Estimated depth at each point in meters", examples=[[1.2, 3.7]]
    )
    calibration_method: str = Field(
        ...,
        description="Calibration method used: 'default', 'reference_object', 'known_focal'",
        examples=["reference_object"],
    )
    processing_time_ms: float = Field(
        ..., description="Server-side processing time in milliseconds", examples=[890.1]
    )


# ── /floorplan ──────────────────────────────────────────────


class FloorplanJobResponse(BaseModel):
    """Async floorplan generation job submission response."""

    job_id: str = Field(..., description="Unique job identifier for polling")
    status: str = Field(
        "processing", description="Job status: processing, complete, failed"
    )
    state: AsyncJobState = Field(
        AsyncJobState.PROCESSING,
        description="Stable job state enum for clients",
    )
    step: str | None = Field(
        None,
        description="Optional processing step when state is processing",
    )
    estimated_time_s: float | None = Field(
        None, description="Estimated completion time in seconds"
    )


class FloorplanResultResponse(BaseModel):
    """Floorplan job result."""

    job_id: str
    status: str
    state: AsyncJobState
    step: str | None = None
    floorplan_url: str | None = Field(
        None, description="URL to download the floor plan (SVG/DXF/JSON)"
    )
    floor_area_m2: float | None = Field(
        None, description="Total floor area in square meters"
    )
    room_count: int | None = Field(
        None, description="Number of detected rooms"
    )
    processing_time_ms: float | None = None
    error: str | None = None


# ── /segment-3d ─────────────────────────────────────────────


class BBox3D(BaseModel):
    min_point: list[float] = Field(
        ..., description="Minimum corner [x, y, z] in meters"
    )
    max_point: list[float] = Field(
        ..., description="Maximum corner [x, y, z] in meters"
    )


class SegmentedObject(BaseModel):
    label: str = Field(..., description="Object label", examples=["chair"])
    confidence: float = Field(
        ..., description="Detection confidence (0-1)", examples=[0.95]
    )
    mask_url: str | None = Field(
        None, description="URL to 3D segmentation mask"
    )
    bbox_3d: BBox3D | None = Field(
        None, description="3D bounding box in world coordinates"
    )


class Segment3DJobResponse(BaseModel):
    """Async 3D segmentation job submission response."""

    job_id: str = Field(..., description="Unique job identifier for polling")
    status: str = Field(
        "processing", description="Job status: processing, complete, failed"
    )
    state: AsyncJobState = Field(
        AsyncJobState.PROCESSING,
        description="Stable job state enum for clients",
    )
    step: str | None = Field(
        None,
        description="Optional processing step when state is processing",
    )
    estimated_time_s: float | None = Field(
        None, description="Estimated completion time in seconds"
    )


class Segment3DResultResponse(BaseModel):
    """3D segmentation job result."""

    job_id: str
    status: str
    state: AsyncJobState
    step: str | None = None
    objects: list[SegmentedObject] | None = Field(
        None, description="List of segmented 3D objects"
    )
    processing_time_ms: float | None = None
    error: str | None = None


# ── Common ──────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """API health check response."""

    status: str = Field("ok", description="Service status", examples=["ok"])
    version: str = Field(..., description="API version", examples=["0.1.0"])
    gpu_available: bool = Field(
        ..., description="Whether GPU is available for inference"
    )
    models_loaded: list[str] = Field(
        ..., description="List of currently loaded model names"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Human-readable error message")
    error_code: str | None = Field(
        None, description="Machine-readable error code"
    )
