"""/v1/reconstruct — 3D reconstruction from video (async)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...config import MAX_ASYNC_VIDEO_FILE_SIZE
from ...models.requests import ReconstructOutput, ReconstructQuality
from ...models.responses import ReconstructJobResponse, ReconstructResultResponse, ReconstructStats
from ._video_job_utils import build_error_responses, validate_and_store_video

router = APIRouter()

_ERROR_RESPONSES = build_error_responses({
    400: "Invalid input (video too short, unsupported format)",
    413: "Video exceeds 500MB size limit",
})


@router.post("/reconstruct", response_model=ReconstructJobResponse, responses=_ERROR_RESPONSES)
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
    uploaded = await validate_and_store_video(
        request,
        video,
        max_file_size=MAX_ASYNC_VIDEO_FILE_SIZE,
        max_duration_s=120,
    )

    # Submit async task
    from ...workers.tasks import reconstruct_task

    task = reconstruct_task.delay(
        video_object_key=uploaded.video_key,
        quality=quality.value,
        output_format=output.value,
        webhook_url=webhook_url,
    )

    # Estimate processing time based on video duration and quality
    time_multiplier = {"draft": 1.0, "standard": 2.0, "high": 4.0}
    estimated = uploaded.info.duration_s * time_multiplier.get(quality.value, 2.0)

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
    from ._video_job_utils import poll_celery_job

    def _map_success(data: dict) -> dict:
        stats = data.get("stats", {})
        return {
            "scene_url": data.get("scene_url"),
            "stats": ReconstructStats(
                num_gaussians=stats.get("num_gaussians"),
                num_points=stats.get("num_points"),
                num_vertices=stats.get("num_vertices"),
                bounding_box=stats.get("bounding_box"),
            ),
        }

    return poll_celery_job(job_id, ReconstructResultResponse, success_mapper=_map_success)
