"""Shared helpers for async video job endpoints."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

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
