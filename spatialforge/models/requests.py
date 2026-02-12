"""Request models for all API endpoints."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# ── /depth ──────────────────────────────────────────────────

class DepthModel(str, Enum):
    LARGE = "large"       # DA3-Metric-Large (Apache 2.0) — default
    BASE = "base"         # DA3-Base (Apache 2.0) — fast
    SMALL = "small"       # DA3-Small (Apache 2.0) — fastest


class DepthOutputFormat(str, Enum):
    PNG16 = "png16"
    EXR = "exr"
    NPY = "npy"


class DepthRequest(BaseModel):
    model: DepthModel = DepthModel.LARGE
    output_format: DepthOutputFormat = DepthOutputFormat.PNG16
    metric: bool = True


# ── /pose ───────────────────────────────────────────────────

class PoseRequest(BaseModel):
    output_pointcloud: bool = False


# ── /reconstruct ────────────────────────────────────────────

class ReconstructQuality(str, Enum):
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"


class ReconstructOutput(str, Enum):
    GAUSSIAN = "gaussian"
    POINTCLOUD = "pointcloud"
    MESH = "mesh"


class ReconstructRequest(BaseModel):
    quality: ReconstructQuality = ReconstructQuality.STANDARD
    output: ReconstructOutput = ReconstructOutput.GAUSSIAN


# ── /measure ────────────────────────────────────────────────

class Point2D(BaseModel):
    x: float
    y: float


class ReferenceObjectType(str, Enum):
    A4_PAPER = "a4_paper"
    CREDIT_CARD = "credit_card"


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ReferenceObject(BaseModel):
    type: ReferenceObjectType
    bbox: BBox


class MeasureRequest(BaseModel):
    points: list[Point2D] = Field(..., min_length=2, max_length=2)
    reference_object: ReferenceObject | None = None


# ── /floorplan ──────────────────────────────────────────────

class FloorplanOutputFormat(str, Enum):
    SVG = "svg"
    DXF = "dxf"
    JSON = "json"


class FloorplanRequest(BaseModel):
    output_format: FloorplanOutputFormat = FloorplanOutputFormat.SVG


# ── /segment-3d ─────────────────────────────────────────────

class Segment3DRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt describing objects to segment")
    output_3d_mask: bool = True
    output_bbox: bool = True


# ── /billing ───────────────────────────────────────────────

class CheckoutRequest(BaseModel):
    plan: str  # "builder", "pro", "enterprise"
    email: str
    success_url: str = "https://spatialforge-demo.fly.dev/docs"
    cancel_url: str = "https://spatialforge-demo.fly.dev/docs"


class PortalRequest(BaseModel):
    email: str
