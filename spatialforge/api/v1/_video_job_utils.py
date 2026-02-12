"""Shared helpers for API v1 endpoints."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request, UploadFile

from ...utils.video import VideoInfo, validate_video

# ── Common error responses ────────────────────────────────

COMMON_ERROR_RESPONSES: dict[int, dict[str, str]] = {
    401: {"description": "Missing or invalid API key"},
    429: {"description": "Monthly rate limit exceeded"},
}


def build_error_responses(extras: dict[int, str]) -> dict[int, dict[str, str]]:
    """Build an OpenAPI responses dict from common errors plus endpoint-specific ones.

    Args:
        extras: Mapping of HTTP status code → description string.
    """
    result = dict(COMMON_ERROR_RESPONSES)
    for code, desc in extras.items():
        result[code] = {"description": desc}
    return result


@dataclass(frozen=True)
class UploadedVideoJob:
    """Validated uploaded video that has been persisted to object storage."""

    video_key: str
    info: VideoInfo


async def _write_upload_to_temp(video: UploadFile, max_file_size: int, chunk_size: int = 1024 * 1024) -> Path:
    """Write upload stream to a temp file with incremental size checks."""
    total = 0
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            path = Path(tmp.name)
            while True:
                chunk = await video.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_file_size:
                    raise HTTPException(status_code=413, detail="Video exceeds 500MB limit")
                tmp.write(chunk)
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


async def validate_and_store_video(
    request: Request,
    video: UploadFile,
    *,
    max_file_size: int,
    max_duration_s: int,
    min_duration_s: float | None = None,
    min_duration_error: str | None = None,
) -> UploadedVideoJob:
    """Validate uploaded video bytes and persist to object storage.

    Raises HTTPException with normalized API error codes:
    - 413: file too large
    - 400: validation failure
    - 503: object store unavailable
    """
    path = await _write_upload_to_temp(video, max_file_size=max_file_size)
    try:
        info = validate_video(path, max_duration_s=max_duration_s)
        if min_duration_s is not None and info.duration_s < min_duration_s:
            raise HTTPException(
                status_code=400,
                detail=min_duration_error or f"Video should be at least {min_duration_s:.0f} seconds",
            )

        obj_store = request.app.state.object_store
        if obj_store is None:
            raise HTTPException(status_code=503, detail="Object store not available")

        video_key = await obj_store.async_upload_file(str(path), content_type="video/mp4", prefix="uploads")
        return UploadedVideoJob(video_key=video_key, info=info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    finally:
        path.unlink(missing_ok=True)


def poll_celery_job(
    job_id: str,
    response_cls: type,
    *,
    success_mapper: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    async_result_getter: Callable[[str], Any] | None = None,
) -> Any:
    """Generic Celery job polling — eliminates duplication across async endpoints.

    Args:
        job_id: Celery task ID to poll.
        response_cls: Pydantic response model class (must accept job_id, status, error kwargs).
        success_mapper: Optional callable to transform SUCCESS result dict into response kwargs.

    Returns:
        An instance of response_cls with the current job state.
    """
    if async_result_getter is None:
        from ...workers.celery_app import celery_app

        async_result_getter = celery_app.AsyncResult

    result = async_result_getter(job_id)

    if result.state == "PENDING":
        return response_cls(job_id=job_id, status="pending")

    if result.state == "PROCESSING":
        meta = result.info or {}
        return response_cls(
            job_id=job_id,
            status=f"processing:{meta.get('step', 'unknown')}",
        )

    if result.state == "SUCCESS":
        data = result.result or {}
        if data.get("status") == "failed":
            return response_cls(
                job_id=job_id,
                status="failed",
                error=data.get("error", "Unknown error"),
            )
        extra_kwargs: dict[str, Any] = {}
        if success_mapper is not None:
            extra_kwargs = success_mapper(data)
        return response_cls(
            job_id=job_id,
            status="complete",
            processing_time_ms=data.get("processing_time_ms"),
            **extra_kwargs,
        )

    if result.state == "FAILURE":
        return response_cls(
            job_id=job_id,
            status="failed",
            error=str(result.result),
        )

    return response_cls(job_id=job_id, status=result.state.lower())
