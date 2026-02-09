"""/v1/floorplan — Auto-generated floor plans from room walkthrough video.

STATUS: BETA — Wall detection and room segmentation are approximate.
Accuracy improves with longer, slower walkthrough videos.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.requests import FloorplanOutputFormat
from ...models.responses import FloorplanJobResponse, FloorplanResultResponse
from ._video_job_utils import validate_and_store_video

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB

_ERROR_RESPONSES = {
    400: {"description": "Invalid input (video too short, unsupported format)"},
    401: {"description": "Missing or invalid API key"},
    413: {"description": "Video exceeds 500MB size limit"},
    429: {"description": "Monthly rate limit exceeded"},
}


@router.post("/floorplan", response_model=FloorplanJobResponse, responses=_ERROR_RESPONSES)
async def start_floorplan(
    request: Request,
    video: UploadFile = File(..., description="Room walkthrough video (MP4, 30s+ recommended)"),
    output_format: FloorplanOutputFormat = Form(FloorplanOutputFormat.SVG),
    user: APIKeyRecord = Depends(get_current_user),
):
    """Generate a floor plan from a room walkthrough video.

    Records a slow walkthrough of the room (30+ seconds recommended).
    Returns a job_id — poll GET /v1/floorplan/{job_id} for results.

    Output includes auto-generated floor plan (SVG/DXF) and floor area in m².
    """
    uploaded = await validate_and_store_video(
        request,
        video,
        max_file_size=MAX_FILE_SIZE,
        max_duration_s=300,  # Allow up to 5 min for floorplan
        min_duration_s=10,
        min_duration_error="Video should be at least 10 seconds for floor plan generation",
    )

    from ...workers.tasks import floorplan_task

    task = floorplan_task.delay(
        video_object_key=uploaded.video_key,
        output_format=output_format.value,
    )

    return FloorplanJobResponse(
        job_id=task.id,
        status="processing",
        estimated_time_s=round(uploaded.info.duration_s * 3.0, 1),
    )


@router.get("/floorplan/{job_id}", response_model=FloorplanResultResponse)
async def get_floorplan_result(
    job_id: str,
    request: Request,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Poll for floor plan generation result."""
    from ...workers.celery_app import celery_app

    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return FloorplanResultResponse(job_id=job_id, status="pending")

    if result.state == "PROCESSING":
        meta = result.info or {}
        return FloorplanResultResponse(
            job_id=job_id,
            status=f"processing:{meta.get('step', 'unknown')}",
        )

    if result.state == "SUCCESS":
        data = result.result or {}
        if data.get("status") == "failed":
            return FloorplanResultResponse(
                job_id=job_id,
                status="failed",
                error=data.get("error"),
            )
        return FloorplanResultResponse(
            job_id=job_id,
            status="complete",
            floorplan_url=data.get("floorplan_url"),
            floor_area_m2=data.get("floor_area_m2"),
            room_count=data.get("room_count"),
        )

    if result.state == "FAILURE":
        return FloorplanResultResponse(
            job_id=job_id,
            status="failed",
            error=str(result.result),
        )

    return FloorplanResultResponse(job_id=job_id, status=result.state.lower())
