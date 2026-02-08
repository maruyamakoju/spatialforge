"""/v1/measure — Real-world distance measurement from images."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.responses import MeasureResponse

router = APIRouter()


@router.post("/measure", response_model=MeasureResponse)
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
    except (json.JSONDecodeError, KeyError, TypeError) as e:
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
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20MB limit")

    from ...utils.image import load_image_rgb, resize_if_needed

    try:
        rgb = load_image_rgb(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    rgb = resize_if_needed(rgb, max_size=4096)

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
