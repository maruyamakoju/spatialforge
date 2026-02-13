"""Tests for SDK data models."""

from __future__ import annotations

import pytest
from spatialforge_client.models import (
    AsyncJob,
    CameraPose,
    DepthResult,
    FloorplanJob,
    MeasureResult,
    PoseResult,
    ReconstructJob,
    Segment3DJob,
)


class TestDepthResult:
    def test_fields(self):
        r = DepthResult(
            depth_map_url="https://example.com/d.png",
            width=1920,
            height=1080,
            min_depth_m=0.1,
            max_depth_m=50.0,
            focal_length_px=1000.0,
            confidence_mean=0.92,
            processing_time_ms=500.0,
        )
        assert r.width == 1920
        assert r.height == 1080
        assert r.focal_length_px == 1000.0

    def test_optional_focal_length(self):
        r = DepthResult(
            depth_map_url="url",
            width=0,
            height=0,
            min_depth_m=0,
            max_depth_m=0,
            focal_length_px=None,
            confidence_mean=0,
            processing_time_ms=0,
        )
        assert r.focal_length_px is None

    def test_raw_default(self):
        r = DepthResult(
            depth_map_url="url",
            width=0,
            height=0,
            min_depth_m=0,
            max_depth_m=0,
            focal_length_px=None,
            confidence_mean=0,
            processing_time_ms=0,
        )
        assert r._raw == {}

    def test_repr_hides_raw(self):
        r = DepthResult(
            depth_map_url="url",
            width=0,
            height=0,
            min_depth_m=0,
            max_depth_m=0,
            focal_length_px=None,
            confidence_mean=0,
            processing_time_ms=0,
            _raw={"big": "data"},
        )
        assert "_raw" not in repr(r)


class TestMeasureResult:
    def test_distance_conversions(self):
        r = MeasureResult(
            distance_m=1.5,
            confidence=0.9,
            depth_at_points=[1.0, 2.0],
            calibration_method="default",
            processing_time_ms=100.0,
        )
        assert r.distance_cm == pytest.approx(150.0)
        assert r.distance_mm == pytest.approx(1500.0)

    def test_zero_distance(self):
        r = MeasureResult(
            distance_m=0.0,
            confidence=1.0,
            depth_at_points=[0.0, 0.0],
            calibration_method="none",
            processing_time_ms=50.0,
        )
        assert r.distance_cm == 0.0
        assert r.distance_mm == 0.0


class TestCameraPose:
    def test_fields(self):
        p = CameraPose(
            frame_index=0,
            rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            translation=[0, 0, 0],
            fx=525, fy=525, cx=320, cy=240,
            width=640, height=480,
        )
        assert p.frame_index == 0
        assert len(p.rotation) == 3
        assert len(p.translation) == 3


class TestPoseResult:
    def test_fields(self):
        r = PoseResult(
            camera_poses=[],
            pointcloud_url=None,
            num_frames=0,
            processing_time_ms=0,
        )
        assert r.num_frames == 0
        assert r.pointcloud_url is None


class TestAsyncJob:
    def test_poll_without_client_raises(self):
        job = AsyncJob(job_id="j1", status="pending")
        with pytest.raises(RuntimeError, match="not associated"):
            job.poll()

    def test_subclass_types(self):
        r = ReconstructJob(job_id="j1", status="p", estimated_time_s=10.0)
        assert isinstance(r, AsyncJob)
        f = FloorplanJob(job_id="j2", status="p", estimated_time_s=5.0)
        assert isinstance(f, AsyncJob)
        s = Segment3DJob(job_id="j3", status="p", estimated_time_s=20.0)
        assert isinstance(s, AsyncJob)

    @pytest.mark.asyncio
    async def test_async_poll_without_client_raises(self):
        job = AsyncJob(job_id="j1", status="pending")
        with pytest.raises(RuntimeError, match="not associated"):
            await job.async_poll()

    def test_wait_prefers_state_field_when_present(self):
        class _FakeClient:
            def _get(self, _path: str) -> dict:
                return {"job_id": "j1", "state": "complete", "scene_url": "https://example.com/s.glb"}

        job = AsyncJob(
            job_id="j1",
            status="processing",
            state="processing",
            _client=_FakeClient(),
            _endpoint="/v1/reconstruct",
        )
        result = job.wait(poll_interval=0.0, timeout=1.0)
        assert result["scene_url"].endswith(".glb")
        assert job.state == "complete"
        assert job.status == "processing"  # preserved when payload omits legacy status

    def test_wait_parses_legacy_processing_step(self):
        class _FakeClient:
            def __init__(self):
                self._calls = 0

            def _get(self, _path: str) -> dict:
                self._calls += 1
                if self._calls == 1:
                    return {"job_id": "j1", "status": "processing:triangulation"}
                return {"job_id": "j1", "status": "complete", "scene_url": "https://example.com/s.glb"}

        job = AsyncJob(
            job_id="j1",
            status="processing",
            _client=_FakeClient(),
            _endpoint="/v1/reconstruct",
        )
        result = job.wait(poll_interval=0.0, timeout=1.0)
        assert result["status"] == "complete"
        assert job.state == "complete"

    @pytest.mark.asyncio
    async def test_async_wait_prefers_state_field_when_present(self):
        class _FakeAsyncClient:
            async def _get(self, _path: str) -> dict:
                return {"job_id": "j1", "state": "complete", "scene_url": "https://example.com/s.glb"}

        job = AsyncJob(
            job_id="j1",
            status="processing",
            state="processing",
            _client=_FakeAsyncClient(),
            _endpoint="/v1/reconstruct",
        )
        result = await job.async_wait(poll_interval=0.0, timeout=1.0)
        assert result["scene_url"].endswith(".glb")
        assert job.state == "complete"
