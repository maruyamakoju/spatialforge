"""/v1/depth — Monocular depth estimation endpoint.

THE core revenue endpoint. Every pixel flows through here.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.requests import DepthModel, DepthOutputFormat
from ...models.responses import DepthMetadata, DepthResponse

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

_ERROR_RESPONSES = {
    400: {"description": "Invalid input (bad image format, unsupported content type)"},
    401: {"description": "Missing or invalid API key"},
    413: {"description": "Image exceeds 20MB size limit"},
    429: {"description": "Monthly rate limit exceeded"},
    503: {"description": "Auth service unavailable (Redis down)"},
    504: {"description": "Inference timed out (try a smaller image or faster model)"},
}


@router.post("/depth", response_model=DepthResponse, responses=_ERROR_RESPONSES)
async def estimate_depth(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WebP)"),
    model: DepthModel = Form(DepthModel.LARGE),
    output_format: DepthOutputFormat = Form(DepthOutputFormat.PNG16),
    metric: bool = Form(True),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Estimate depth from a single image.

    Returns a depth map (16-bit PNG, EXR, or NumPy) plus a colorized
    visualization URL for browser display.

    **Models** (all Apache 2.0):
    - `large` (default): DA3-Metric-Large — best quality, metric depth in meters
    - `base`: DA3-Base — fast, relative depth
    - `small`: DA3-Small — fastest, edge/mobile

    **Latency target**: <100ms for 1080p on RTX 5090 (TensorRT FP16).
    """
    raw_bytes, rgb = await _load_and_validate(image)

    # Choose model based on metric flag
    model_name = model.value
    if metric and model_name in ("small", "base"):
        # small/base don't have metric versions; use metric-large instead
        model_name = "large"

    # Run inference
    from ...inference.depth_engine import DepthEngine

    engine = DepthEngine(request.app.state.model_manager)
    result = engine.estimate(
        rgb,
        model_size=model_name,
        image_bytes=raw_bytes,
    )

    # Encode depth data
    from ...utils.image import depth_to_colormap, depth_to_npy, depth_to_png16

    if output_format == DepthOutputFormat.PNG16:
        max_val = result.max_depth if result.is_metric else 1.0
        encoded = depth_to_png16(result.depth_map, max_depth=max_val)
        ext, ctype = "png", "image/png"
    elif output_format == DepthOutputFormat.NPY:
        encoded = depth_to_npy(result.depth_map)
        ext, ctype = "npy", "application/octet-stream"
    else:  # EXR or unknown — fallback to NPY
        encoded = depth_to_npy(result.depth_map)
        ext, ctype = "npy", "application/octet-stream"
        # TODO: Add native EXR support via OpenEXR library

    # Colorized visualization (always generated)
    colormap_bytes = depth_to_colormap(result.depth_map)

    # Upload to object store (or fallback to base64)
    obj_store = request.app.state.object_store
    if obj_store is not None:
        key = obj_store.upload_bytes(encoded, content_type=ctype, prefix="depth", extension=ext)
        depth_map_url = obj_store.get_presigned_url(key)

        color_key = obj_store.upload_bytes(
            colormap_bytes, content_type="image/jpeg", prefix="depth_vis", extension="jpg",
        )
        colormap_url = obj_store.get_presigned_url(color_key)
    else:
        b64 = base64.b64encode(encoded).decode()
        depth_map_url = f"data:{ctype};base64,{b64}"

        b64_color = base64.b64encode(colormap_bytes).decode()
        colormap_url = f"data:image/jpeg;base64,{b64_color}"

    # Response
    min_m = result.min_depth if result.is_metric else 0.0
    max_m = result.max_depth if result.is_metric else 1.0

    return DepthResponse(
        depth_map_url=depth_map_url,
        colormap_url=colormap_url,
        metadata=DepthMetadata(
            width=result.width,
            height=result.height,
            min_depth_m=min_m,
            max_depth_m=max_m,
            focal_length_px=result.focal_length_px,
            confidence_mean=result.confidence_mean,
        ),
        processing_time_ms=round(result.processing_time_ms, 2),
    )


@router.post("/depth/visualize")
async def depth_visualize(
    request: Request,
    image: UploadFile = File(..., description="Image file (JPEG/PNG/WebP)"),
    model: DepthModel = Form(DepthModel.LARGE),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Return a colorized depth visualization as JPEG.

    Convenience endpoint for demos — returns an INFERNO colormap image
    directly renderable in the browser.
    """
    raw_bytes, rgb = await _load_and_validate(image)

    from ...inference.depth_engine import DepthEngine

    engine = DepthEngine(request.app.state.model_manager)
    result = engine.estimate(rgb, model_size=model.value, image_bytes=raw_bytes)

    from ...utils.image import depth_to_colormap

    colormap_bytes = depth_to_colormap(result.depth_map)

    return Response(
        content=colormap_bytes,
        media_type="image/jpeg",
        headers={
            "X-Depth-Min": str(round(result.min_depth, 4)),
            "X-Depth-Max": str(round(result.max_depth, 4)),
            "X-Is-Metric": str(result.is_metric).lower(),
            "X-Processing-Ms": str(round(result.processing_time_ms, 2)),
            "X-Model": result.model_used,
            "X-License": result.model_license,
        },
    )


async def _load_and_validate(image: UploadFile):
    """Read, validate, return (raw_bytes, rgb_array)."""
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
        raise HTTPException(status_code=400, detail=str(e)) from None

    rgb = resize_if_needed(rgb, max_size=4096)
    return content, rgb
