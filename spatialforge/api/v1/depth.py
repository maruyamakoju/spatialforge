"""/v1/depth — Monocular depth estimation endpoint."""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.requests import DepthModel, DepthOutputFormat
from ...models.responses import DepthMetadata, DepthResponse

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/depth", response_model=DepthResponse)
async def estimate_depth(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WebP)"),
    model: DepthModel = Form(DepthModel.GIANT),
    output_format: DepthOutputFormat = Form(DepthOutputFormat.PNG16),
    metric: bool = Form(True),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Estimate depth from a single image.

    Returns a 16-bit PNG depth map (default), EXR, or NumPy array,
    along with metadata including min/max depth in meters, focal length,
    and confidence score.

    **Latency target**: <100ms for 1080p on RTX 5090 (TensorRT FP16).
    """
    content, rgb = await _load_and_validate(image)

    # Run inference
    from ...inference.depth_engine import DepthEngine

    engine = DepthEngine(request.app.state.model_manager)
    result = engine.estimate(rgb, model_size=model.value)

    # Encode output
    from ...utils.image import depth_to_npy, depth_to_png16

    if output_format == DepthOutputFormat.PNG16:
        encoded = depth_to_png16(result.depth_map, max_depth=result.max_depth_m)
        ext, ctype = "png", "image/png"
    elif output_format == DepthOutputFormat.NPY:
        encoded = depth_to_npy(result.depth_map)
        ext, ctype = "npy", "application/octet-stream"
    else:  # EXR — fallback to NPY for MVP
        encoded = depth_to_npy(result.depth_map)
        ext, ctype = "npy", "application/octet-stream"

    # Also generate colorized visualization for API response
    from ...utils.image import depth_to_colormap

    colormap_bytes = depth_to_colormap(result.depth_map)

    # Upload both depth data and colorized version
    obj_store = request.app.state.object_store
    if obj_store is not None:
        key = obj_store.upload_bytes(encoded, content_type=ctype, prefix="depth", extension=ext)
        depth_map_url = obj_store.get_presigned_url(key)

        color_key = obj_store.upload_bytes(colormap_bytes, content_type="image/jpeg", prefix="depth_vis", extension="jpg")
        colormap_url = obj_store.get_presigned_url(color_key)
    else:
        b64 = base64.b64encode(encoded).decode()
        depth_map_url = f"data:{ctype};base64,{b64}"

        b64_color = base64.b64encode(colormap_bytes).decode()
        colormap_url = f"data:image/jpeg;base64,{b64_color}"

    return DepthResponse(
        depth_map_url=depth_map_url,
        colormap_url=colormap_url,
        metadata=DepthMetadata(
            width=result.width,
            height=result.height,
            min_depth_m=result.min_depth_m,
            max_depth_m=result.max_depth_m,
            focal_length_px=result.focal_length_px,
            confidence_mean=result.confidence_mean,
        ),
        processing_time_ms=round(result.processing_time_ms, 2),
    )


@router.post("/depth/visualize")
async def depth_visualize(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WebP)"),
    model: DepthModel = Form(DepthModel.GIANT),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Return a colorized depth visualization as JPEG (for browser display).

    This is a convenience endpoint — returns the colorized INFERNO colormap
    directly as an image response, ideal for demos and previews.
    """
    content, rgb = await _load_and_validate(image)

    from ...inference.depth_engine import DepthEngine

    engine = DepthEngine(request.app.state.model_manager)
    result = engine.estimate(rgb, model_size=model.value)

    from ...utils.image import depth_to_colormap

    colormap_bytes = depth_to_colormap(result.depth_map)

    return Response(
        content=colormap_bytes,
        media_type="image/jpeg",
        headers={
            "X-Depth-Min-M": str(round(result.min_depth_m, 4)),
            "X-Depth-Max-M": str(round(result.max_depth_m, 4)),
            "X-Processing-Ms": str(round(result.processing_time_ms, 2)),
            "X-Confidence": str(round(result.confidence_mean, 3)),
        },
    )


async def _load_and_validate(image: UploadFile):
    """Read, validate, and return (raw_bytes, rgb_array) from an uploaded image."""
    content = await image.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 20MB limit")

    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG/WebP)")

    from ...utils.image import load_image_rgb, resize_if_needed

    try:
        rgb = load_image_rgb(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    rgb = resize_if_needed(rgb, max_size=4096)
    return content, rgb
