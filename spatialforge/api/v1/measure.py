"""/v1/measure — Real-world distance measurement from images."""

from __future__ import annotations

import json
import math

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...config import MAX_IMAGE_FILE_SIZE
from ...models.responses import MeasureResponse
from ._video_job_utils import build_error_responses

router = APIRouter()

_ERROR_RESPONSES = build_error_responses({
    400: "Invalid input (bad points format, coordinates out of bounds, NaN/Inf)",
    413: "Image exceeds 20MB size limit",
    504: "Inference timed out",
})


@router.post("/measure", response_model=MeasureResponse, responses=_ERROR_RESPONSES)
async def measure_distance(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WebP)"),
    points: str = Form(..., description='JSON: [{"x": 100, "y": 200}, {"x": 300, "y": 400}]'),
    reference_object: str | None = Form(
        None,
        description='Optional JSON: {"type": "a4_paper", "bbox": {"x": 0, "y": 0, "w": 100, "h": 140}}',
    ),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Measure real-world distance between two points in an image.

    Optionally provide a reference object (A4 paper or credit card) with its
    bounding box for improved accuracy (±2% with reference, ±5% without).
    """
    # Parse points
    try:
        points_data = json.loads(points)
        if not isinstance(points_data, list) or len(points_data) != 2:
            raise ValueError("Exactly 2 points required")
        p1 = (float(points_data[0]["x"]), float(points_data[0]["y"]))
        p2 = (float(points_data[1]["x"]), float(points_data[1]["y"]))
        for pt in [p1, p2]:
            if any(math.isnan(v) or math.isinf(v) for v in pt):
                raise ValueError("Coordinates must be finite numbers")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid points format: {e}") from None

    # Parse reference object
    ref_type = None
    ref_bbox = None
    if reference_object:
        try:
            ref_data = json.loads(reference_object)
            ref_type = ref_data["type"]
            bbox = ref_data["bbox"]
            ref_bbox = (float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"]))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid reference_object format: {e}") from None

    # Load image
    content = await image.read()
    if len(content) > MAX_IMAGE_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 20MB limit")

    from ...utils.image import load_image_rgb, resize_if_needed

    try:
        rgb = load_image_rgb(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    rgb = resize_if_needed(rgb, max_size=4096)

    # Validate point coordinates are within image bounds
    h, w = rgb.shape[:2]
    for i, pt in enumerate([p1, p2], 1):
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            raise HTTPException(
                status_code=400,
                detail=f"Point {i} ({pt[0]}, {pt[1]}) is outside image bounds ({w}x{h})",
            )

    # Run measurement
    from ...inference.measure_engine import MeasureEngine

    engine = MeasureEngine(request.app.state.model_manager)
    result = engine.measure(
        image_rgb=rgb,
        point1=p1,
        point2=p2,
        reference_type=ref_type,
        reference_bbox=ref_bbox,
    )

    return MeasureResponse(
        distance_m=result.distance_m,
        confidence=result.confidence,
        depth_at_points=result.depth_at_points,
        calibration_method=result.calibration_method,
        processing_time_ms=round(result.processing_time_ms, 2),
    )
