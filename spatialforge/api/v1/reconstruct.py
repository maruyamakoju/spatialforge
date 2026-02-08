"""/v1/reconstruct — 3D reconstruction from video (async)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.requests import ReconstructOutput, ReconstructQuality
from ...models.responses import ReconstructJobResponse, ReconstructResultResponse, ReconstructStats

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB


@router.post("/reconstruct", response_model=ReconstructJobResponse)
async def start_reconstruction(
    request: Request,
    video: UploadFile = File(..., description="Video file (MP4, max 120s)"),
    quality: ReconstructQuality = Form(ReconstructQuality.STANDARD),
    output: ReconstructOutput = Form(ReconstructOutput.GAUSSIAN),
    webhook_url: str | None = Form(None, description="Optional webhook URL for completion notification"),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Start async 3D reconstruction from a video.

    Extracts keyframes, estimates depth and camera poses, and generates
    a 3D Gaussian Splat, point cloud, or mesh.

    Returns a job_id to poll for results via GET /v1/reconstruct/{job_id}.
    """
    content = await video.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Video exceeds 500MB limit")

    # Validate video
    from ...utils.video import save_uploaded_video, validate_video

    path = save_uploaded_video(content)
    try:
        info = validate_video(path, max_duration_s=120)

        obj_store = request.app.state.object_store
        if obj_store is None:
            raise HTTPException(status_code=503, detail="Object store not available")

        video_key = obj_store.upload_file(str(path), content_type="video/mp4", prefix="uploads")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    finally:
        path.unlink(missing_ok=True)

    # Submit async task
    from ...workers.tasks import reconstruct_task

    task = reconstruct_task.delay(
        video_object_key=video_key,
        quality=quality.value,
        output_format=output.value,
        webhook_url=webhook_url,
    )

    # Estimate processing time based on video duration and quality
    time_multiplier = {"draft": 1.0, "standard": 2.0, "high": 4.0}
    estimated = info.duration_s * time_multiplier.get(quality.value, 2.0)

    return ReconstructJobResponse(
        job_id=task.id,
        status="processing",
        estimated_time_s=round(estimated, 1),
    )


@router.get("/reconstruct/{job_id}", response_model=ReconstructResultResponse)
async def get_reconstruction_result(
    job_id: str,
    request: Request,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Poll for reconstruction job result."""
    from ...workers.celery_app import celery_app

    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return ReconstructResultResponse(job_id=job_id, status="pending")

    if result.state == "PROCESSING":
        meta = result.info or {}
        return ReconstructResultResponse(
            job_id=job_id,
            status=f"processing:{meta.get('step', 'unknown')}",
        )

    if result.state == "SUCCESS":
        data = result.result or {}
        if data.get("status") == "failed":
            return ReconstructResultResponse(
                job_id=job_id,
                status="failed",
                error=data.get("error", "Unknown error"),
            )

        stats = data.get("stats", {})
        return ReconstructResultResponse(
            job_id=job_id,
            status="complete",
            scene_url=data.get("scene_url"),
            viewer_url=None,  # TODO: 3D viewer hosting
            stats=ReconstructStats(
                num_gaussians=stats.get("num_gaussians"),
                num_points=stats.get("num_points"),
                num_vertices=stats.get("num_vertices"),
                bounding_box=stats.get("bounding_box"),
            ),
            processing_time_ms=data.get("processing_time_ms"),
        )

    if result.state == "FAILURE":
        return ReconstructResultResponse(
            job_id=job_id,
            status="failed",
            error=str(result.result),
        )

    return ReconstructResultResponse(job_id=job_id, status=result.state.lower())
