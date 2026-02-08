"""/v1/segment-3d — 3D segmentation with text prompts.

STATUS: BETA — SAM3 integration pending (requires separate container + HF access approval).
This endpoint currently returns depth-based placeholder results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from ...auth.api_keys import APIKeyRecord, get_current_user
from ...models.responses import Segment3DJobResponse, Segment3DResultResponse, SegmentedObject

router = APIRouter()

MAX_FILE_SIZE = 500 * 1024 * 1024

BETA_NOTICE = (
    "This endpoint is in BETA. SAM3 integration is pending "
    "(requires separate container and HuggingFace access approval). "
    "Results are depth-based placeholders."
)


@router.post("/segment-3d", response_model=Segment3DJobResponse)
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

    content = await video.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Video exceeds 500MB limit")

    from ...utils.video import save_uploaded_video, validate_video

    path = save_uploaded_video(content)
    try:
        validate_video(path)
    except ValueError as e:
        import os

        os.unlink(path)
        raise HTTPException(status_code=400, detail=str(e))

    obj_store = request.app.state.object_store
    if obj_store is None:
        import os

        os.unlink(path)
        raise HTTPException(status_code=503, detail="Object store not available")

    video_key = obj_store.upload_file(str(path), content_type="video/mp4", prefix="uploads")
    import os

    os.unlink(path)

    from ...workers.tasks import segment_3d_task

    task = segment_3d_task.delay(
        video_object_key=video_key,
        prompt=prompt,
        output_3d_mask=output_3d_mask,
        output_bbox=output_bbox,
    )

    return Segment3DJobResponse(
        job_id=task.id,
        status="processing",
        estimated_time_s=30.0,
    )


@router.get("/segment-3d/{job_id}", response_model=Segment3DResultResponse)
async def get_segment_3d_result(
    job_id: str,
    request: Request,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Poll for 3D segmentation result."""
    from ...workers.celery_app import celery_app

    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        return Segment3DResultResponse(job_id=job_id, status="pending")

    if result.state == "PROCESSING":
        meta = result.info or {}
        return Segment3DResultResponse(
            job_id=job_id,
            status=f"processing:{meta.get('step', 'unknown')}",
        )

    if result.state == "SUCCESS":
        data = result.result or {}
        if data.get("status") == "failed":
            return Segment3DResultResponse(
                job_id=job_id,
                status="failed",
                error=data.get("error"),
            )

        objects = []
        for obj in data.get("objects", []):
            from ...models.responses import BBox3D

            bbox_3d = None
            if obj.get("bbox_3d"):
                bbox_3d = BBox3D(
                    min_point=obj["bbox_3d"]["min_point"],
                    max_point=obj["bbox_3d"]["max_point"],
                )
            objects.append(
                SegmentedObject(
                    label=obj["label"],
                    confidence=obj["confidence"],
                    mask_url=obj.get("mask_url"),
                    bbox_3d=bbox_3d,
                )
            )

        return Segment3DResultResponse(
            job_id=job_id,
            status="complete",
            objects=objects,
        )

    if result.state == "FAILURE":
        return Segment3DResultResponse(
            job_id=job_id,
            status="failed",
            error=str(result.result),
        )

    return Segment3DResultResponse(job_id=job_id, status=result.state.lower())
