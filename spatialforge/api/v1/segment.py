"""/v1/segment-3d — 3D segmentation with text prompts.

STATUS: BETA — SAM3 integration pending (requires separate container + HF access approval).
This endpoint currently returns depth-based placeholder results.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.responses import (
    AsyncJobState,
    BBox3D,
    Segment3DJobResponse,
    Segment3DResultResponse,
    SegmentedObject,
)
from ._video_job_utils import poll_celery_job, validate_and_store_video

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024

_ERROR_RESPONSES = {
    400: {"description": "Invalid input (video too short, prompt empty)"},
    401: {"description": "Missing or invalid API key"},
    413: {"description": "Video exceeds 500MB size limit"},
    429: {"description": "Monthly rate limit exceeded"},
}

BETA_NOTICE = (
    "This endpoint is in BETA. SAM3 integration is pending "
    "(requires separate container and HuggingFace access approval). "
    "Results are depth-based placeholders."
)


def _map_segment_success(data: dict[str, Any]) -> dict[str, Any]:
    """Map worker payload to segment response fields."""
    objects = []
    for obj in data.get("objects", []):
        if not isinstance(obj, dict):
            continue

        bbox_3d = None
        raw_bbox = obj.get("bbox_3d")
        if isinstance(raw_bbox, dict):
            min_point = raw_bbox.get("min_point")
            max_point = raw_bbox.get("max_point")
            if isinstance(min_point, list) and isinstance(max_point, list):
                bbox_3d = BBox3D(min_point=min_point, max_point=max_point)

        objects.append(
            SegmentedObject(
                label=str(obj.get("label", "unknown")),
                confidence=float(obj.get("confidence", 0.0)),
                mask_url=obj.get("mask_url"),
                bbox_3d=bbox_3d,
            )
        )

    return {"objects": objects}


@router.post("/segment-3d", response_model=Segment3DJobResponse, responses=_ERROR_RESPONSES)
async def start_segment_3d(
    request: Request,
    video: UploadFile = File(..., description="Video file (MP4)"),
    prompt: str = Form(..., description="Text prompt describing objects to segment"),
    output_3d_mask: bool = Form(True),
    output_bbox: bool = Form(True),
    user: APIKeyRecord = Depends(get_current_user),
):
    """**[BETA]** Segment objects in 3D using natural language.

    SAM3 integration is pending. Currently returns depth-based placeholder results.

    Returns a job_id — poll GET /v1/segment-3d/{job_id} for results.
    """
    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt is required")

    uploaded = await validate_and_store_video(
        request,
        video,
        max_file_size=MAX_FILE_SIZE,
        max_duration_s=120,
    )

    from ...workers.tasks import segment_3d_task

    task = segment_3d_task.delay(
        video_object_key=uploaded.video_key,
        prompt=prompt,
        output_3d_mask=output_3d_mask,
        output_bbox=output_bbox,
    )

    return Segment3DJobResponse(
        job_id=task.id,
        status="processing",
        state=AsyncJobState.PROCESSING,
        step=None,
        estimated_time_s=30.0,
    )


@router.get("/segment-3d/{job_id}", response_model=Segment3DResultResponse)
async def get_segment_3d_result(
    job_id: str,
    request: Request,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Poll for 3D segmentation result."""
    return poll_celery_job(
        job_id=job_id,
        response_cls=Segment3DResultResponse,
        success_mapper=_map_segment_success,
    )
