"""Shared helpers for async video job endpoints."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request, UploadFile

from ...models.responses import AsyncJobState
from ...utils.video import VideoInfo, validate_video


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


def _legacy_status(state: AsyncJobState, step: str | None = None) -> str:
    """Keep backward-compatible status strings while exposing stable state."""
    if state is AsyncJobState.PROCESSING:
        return f"processing:{step}" if step else "processing"
    return str(state.value)


def poll_celery_job(
    job_id: str,
    response_cls: type,
    *,
    success_mapper: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    async_result_getter: Callable[[str], Any] | None = None,
) -> Any:
    """Generic Celery job polling for async endpoints.

    The API keeps the legacy `status` string for compatibility and also emits
    stable `state` and optional `step` fields for SDK/client correctness.
    """
    if async_result_getter is None:
        from ...workers.celery_app import celery_app

        async_result_getter = celery_app.AsyncResult

    result = async_result_getter(job_id)

    if result.state == "PENDING":
        state = AsyncJobState.PENDING
        return response_cls(
            job_id=job_id,
            status=_legacy_status(state),
            state=state,
            step=None,
        )

    if result.state == "PROCESSING":
        meta = result.info or {}
        step = str(meta.get("step") or "unknown")
        state = AsyncJobState.PROCESSING
        return response_cls(
            job_id=job_id,
            status=_legacy_status(state, step),
            state=state,
            step=step,
        )

    if result.state == "SUCCESS":
        data = result.result or {}
        if data.get("status") == "failed" or data.get("state") == AsyncJobState.FAILED.value:
            state = AsyncJobState.FAILED
            return response_cls(
                job_id=job_id,
                status=_legacy_status(state),
                state=state,
                step=None,
                error=data.get("error", "Unknown error"),
            )

        extra_kwargs: dict[str, Any] = {}
        if success_mapper is not None:
            extra_kwargs = success_mapper(data)
        state = AsyncJobState.COMPLETE
        return response_cls(
            job_id=job_id,
            status=_legacy_status(state),
            state=state,
            step=None,
            processing_time_ms=data.get("processing_time_ms"),
            **extra_kwargs,
        )

    if result.state == "FAILURE":
        state = AsyncJobState.FAILED
        return response_cls(
            job_id=job_id,
            status=_legacy_status(state),
            state=state,
            step=None,
            error=str(result.result),
        )

    # Unknown Celery state (e.g. RETRY) is exposed via legacy status while
    # mapped to `processing` for stable state semantics.
    raw_state = str(result.state or "").lower()
    step = raw_state or "unknown"
    state = AsyncJobState.PROCESSING
    return response_cls(
        job_id=job_id,
        status=raw_state or _legacy_status(state, step),
        state=state,
        step=step,
    )
