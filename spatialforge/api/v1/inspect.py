"""/v1/inspect — Railway track inspection endpoints.

Combines depth estimation + defect detection for track inspection.
  - POST /inspect: Single image inspection (sync)
  - POST /inspect/video: Video inspection (async via Celery)
  - GET /inspect/{job_id}: Poll video inspection results
"""

from __future__ import annotations

import base64
import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.inspection import (
    InspectModel,
    InspectResponse,
    InspectVideoJobResponse,
    InspectionReportResponse,
    Severity,
)

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB

_ERROR_RESPONSES = {
    400: {"description": "Invalid input (bad image/video format)"},
    401: {"description": "Missing or invalid API key"},
    413: {"description": "File exceeds size limit"},
    429: {"description": "Monthly rate limit exceeded"},
    503: {"description": "Inspection model not available"},
    504: {"description": "Inference timed out"},
}


def _get_defect_engine(request: Request):
    """Get or create the defect detection engine."""
    if not hasattr(request.app.state, "defect_engine") or request.app.state.defect_engine is None:
        from ...inference.defect_engine import DefectEngine
        from ...config import get_settings

        settings = get_settings()
        model_path = settings.model_dir / "railscan-yolov8m.pt"

        request.app.state.defect_engine = DefectEngine(
            model_path=model_path,
            device=settings.device if settings.device != "cuda" or __import__("torch").cuda.is_available() else "cpu",
        )

    return request.app.state.defect_engine


def _get_inspection_pipeline(request: Request):
    """Get or create the inspection pipeline."""
    if not hasattr(request.app.state, "inspection_pipeline") or request.app.state.inspection_pipeline is None:
        from ...inference.depth_engine import DepthEngine
        from ...inference.inspector import InspectionPipeline, InspectionConfig

        defect_engine = _get_defect_engine(request)

        # Create depth engine from model manager
        depth_engine = None
        if hasattr(request.app.state, "model_manager") and request.app.state.model_manager is not None:
            depth_engine = DepthEngine(request.app.state.model_manager)

        config = InspectionConfig(
            confidence_threshold=0.25,
            enable_depth=depth_engine is not None,
        )

        request.app.state.inspection_pipeline = InspectionPipeline(
            defect_engine=defect_engine,
            depth_engine=depth_engine,
            config=config,
        )

    return request.app.state.inspection_pipeline


@router.post("/inspect", response_model=InspectResponse, responses=_ERROR_RESPONSES)
async def inspect_image(
    request: Request,
    image: UploadFile = File(..., description="Track image (JPEG/PNG/WebP)"),
    model: InspectModel = Form(InspectModel.STANDARD),
    km_marker: float | None = Form(None, description="Kilometer post marker"),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Inspect a single track image for defects.

    Runs defect detection + depth estimation, returns detected defects
    with bounding boxes, severity, and confidence scores.

    **Models**:
    - `standard` (default): YOLOv8m — balanced speed/accuracy
    - `fast`: YOLOv8n — real-time, lower accuracy
    - `precise`: YOLOv8x — highest accuracy, slower
    """
    # Load and validate image
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

    # Run inspection
    pipeline = _get_inspection_pipeline(request)
    result = pipeline.inspect_image(rgb, km_marker=km_marker)

    # Severity summary
    severity_summary = {s.value: 0 for s in Severity}
    for defect in result.unique_defects:
        severity_summary[defect.severity.value] += 1

    # Encode annotated image if available
    annotated_url = None
    if result.annotated_frames:
        import cv2

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(result.annotated_frames[0], cv2.COLOR_RGB2BGR))
        annotated_bytes = buf.tobytes()

        obj_store = getattr(request.app.state, "object_store", None)
        if obj_store is not None:
            key = await obj_store.async_upload_bytes(
                annotated_bytes, content_type="image/jpeg",
                prefix="inspect_annotated", extension="jpg",
            )
            annotated_url = obj_store.get_presigned_url(key)
        else:
            b64 = base64.b64encode(annotated_bytes).decode()
            annotated_url = f"data:image/jpeg;base64,{b64}"

    return InspectResponse(
        defects=result.unique_defects,
        defect_count=len(result.unique_defects),
        severity_summary=severity_summary,
        annotated_image_url=annotated_url,
        depth_map_url=None,  # TODO: include depth viz
        processing_time_ms=round(result.total_processing_time_ms, 2),
    )


@router.post("/inspect/video", response_model=InspectVideoJobResponse, responses=_ERROR_RESPONSES)
async def inspect_video(
    request: Request,
    video: UploadFile = File(..., description="Track footage (MP4/MOV/AVI)"),
    model: InspectModel = Form(InspectModel.STANDARD),
    km_start: float | None = Form(None, description="Starting kilometer marker"),
    km_end: float | None = Form(None, description="Ending kilometer marker"),
    sample_fps: float = Form(3.0, description="Analysis frame rate (FPS)"),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Submit track footage for async inspection.

    The video will be analyzed frame-by-frame at the specified sample rate.
    Poll the job status with GET /v1/inspect/{job_id}.

    **Supported formats**: MP4, MOV, AVI
    **Max file size**: 200MB
    **Max duration**: 120 seconds
    """
    content = await video.read()
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="Video exceeds 200MB limit")

    content_type = video.content_type or ""
    if not any(t in content_type for t in ("video/", "application/octet-stream")):
        raise HTTPException(status_code=400, detail="File must be a video (MP4/MOV/AVI)")

    # Generate job ID
    job_id = f"insp_{uuid.uuid4().hex[:16]}"

    # Store video in object store for worker processing
    obj_store = getattr(request.app.state, "object_store", None)
    if obj_store is not None:
        ext = video.filename.rsplit(".", 1)[-1] if video.filename and "." in video.filename else "mp4"
        video_key = await obj_store.async_upload_bytes(
            content, content_type=content_type,
            prefix="inspect_video", extension=ext,
        )

        # Store job metadata in Redis
        redis = getattr(request.app.state, "redis", None)
        if redis:
            import json

            await redis.hset(f"inspect_job:{job_id}", mapping={
                "status": "processing",
                "video_key": video_key,
                "model": model.value,
                "km_start": str(km_start or ""),
                "km_end": str(km_end or ""),
                "sample_fps": str(sample_fps),
            })
            await redis.expire(f"inspect_job:{job_id}", 86400)  # 24h TTL

        # TODO: Submit to Celery worker for async processing
        # from ...workers.tasks import inspect_video_task
        # inspect_video_task.delay(job_id, video_key, model.value, km_start, km_end, sample_fps)

    # Estimated processing time (rough: 0.5s per sampled frame)
    est_duration = len(content) / (5 * 1024 * 1024)  # Very rough: 5MB/s
    est_frames = est_duration * sample_fps
    est_time = est_frames * 0.5  # 0.5s per frame

    return InspectVideoJobResponse(
        job_id=job_id,
        status="processing",
        estimated_time_s=round(est_time, 1),
    )


@router.get("/inspect/{job_id}", response_model=InspectionReportResponse, responses=_ERROR_RESPONSES)
async def get_inspection_result(
    request: Request,
    job_id: str,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Get the result of an async video inspection job.

    Returns the full inspection report when processing is complete,
    or current status if still in progress.
    """
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise HTTPException(status_code=503, detail="Job tracking unavailable (Redis not connected)")

    job_data = await redis.hgetall(f"inspect_job:{job_id}")
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    status = job_data.get("status", "unknown")

    if status == "complete":
        import json

        summary_raw = job_data.get("summary")
        report_url = job_data.get("report_url")

        summary = None
        if summary_raw:
            from ...models.inspection import InspectionReportSummary
            summary = InspectionReportSummary.model_validate_json(summary_raw)

        return InspectionReportResponse(
            job_id=job_id,
            status="complete",
            summary=summary,
            report_url=report_url,
            processing_time_ms=float(job_data.get("processing_time_ms", 0)),
        )
    elif status == "failed":
        return InspectionReportResponse(
            job_id=job_id,
            status="failed",
            error=job_data.get("error", "Unknown error"),
        )
    else:
        return InspectionReportResponse(
            job_id=job_id,
            status=status,
        )
