"""Tests for generic Celery polling helper."""

from __future__ import annotations

from pydantic import BaseModel

from spatialforge.api.v1._video_job_utils import poll_celery_job


class _JobResponse(BaseModel):
    job_id: str
    status: str
    processing_time_ms: float | None = None
    error: str | None = None
    scene_url: str | None = None


class _FakeResult:
    def __init__(self, state: str, info=None, result=None):
        self.state = state
        self.info = info
        self.result = result


def test_poll_celery_job_pending():
    response = poll_celery_job(
        "job-1",
        _JobResponse,
        async_result_getter=lambda _job_id: _FakeResult("PENDING"),
    )
    assert response.status == "pending"


def test_poll_celery_job_processing_step():
    response = poll_celery_job(
        "job-2",
        _JobResponse,
        async_result_getter=lambda _job_id: _FakeResult("PROCESSING", info={"step": "uploading"}),
    )
    assert response.status == "processing:uploading"


def test_poll_celery_job_success_mapper():
    response = poll_celery_job(
        "job-3",
        _JobResponse,
        success_mapper=lambda data: {"scene_url": data.get("scene_url")},
        async_result_getter=lambda _job_id: _FakeResult(
            "SUCCESS",
            result={"status": "complete", "scene_url": "https://example.com/s.glb", "processing_time_ms": 123.4},
        ),
    )
    assert response.status == "complete"
    assert response.scene_url == "https://example.com/s.glb"
    assert response.processing_time_ms == 123.4


def test_poll_celery_job_success_failed_payload():
    response = poll_celery_job(
        "job-4",
        _JobResponse,
        async_result_getter=lambda _job_id: _FakeResult(
            "SUCCESS",
            result={"status": "failed", "error": "boom"},
        ),
    )
    assert response.status == "failed"
    assert response.error == "boom"
