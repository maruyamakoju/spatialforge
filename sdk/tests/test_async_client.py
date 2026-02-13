"""Tests for the asynchronous SpatialForge client."""

from __future__ import annotations

import pytest
from spatialforge_client import (
    AsyncClient,
    DepthResult,
    FloorplanJob,
    MeasureResult,
    PoseResult,
    ReconstructJob,
    Segment3DJob,
    SpatialForgeError,
)


@pytest.mark.asyncio
class TestAsyncClientInit:
    async def test_async_context_manager(self, async_client):
        async with async_client as c:
            assert c is async_client


@pytest.mark.asyncio
class TestAsyncDepth:
    async def test_depth_from_file(self, async_client, tiny_image):
        result = await async_client.depth(tiny_image)
        assert isinstance(result, DepthResult)
        assert result.width == 640
        assert result.height == 480
        assert result.min_depth_m == 0.5
        assert result.max_depth_m == 12.3

    async def test_depth_from_bytes(self, async_client):
        result = await async_client.depth(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        assert isinstance(result, DepthResult)

    async def test_depth_from_string_path(self, async_client, tiny_image):
        result = await async_client.depth(str(tiny_image))
        assert isinstance(result, DepthResult)


@pytest.mark.asyncio
class TestAsyncMeasure:
    async def test_measure_basic(self, async_client, tiny_image):
        result = await async_client.measure(
            tiny_image, point1=(100, 200), point2=(500, 200)
        )
        assert isinstance(result, MeasureResult)
        assert result.distance_m == 2.45

    async def test_measure_with_reference(self, async_client, tiny_image):
        ref = {"type": "door", "bbox": [10, 20, 100, 400]}
        result = await async_client.measure(
            tiny_image, (0, 0), (100, 100), reference_object=ref
        )
        assert isinstance(result, MeasureResult)


@pytest.mark.asyncio
class TestAsyncPose:
    async def test_pose_from_video(self, async_client, tiny_video):
        result = await async_client.pose(video=tiny_video)
        assert isinstance(result, PoseResult)
        assert result.num_frames == 2

    async def test_pose_from_images(self, async_client, tiny_image):
        result = await async_client.pose(images=[tiny_image, tiny_image])
        assert isinstance(result, PoseResult)

    async def test_pose_no_input_raises(self, async_client):
        with pytest.raises(ValueError, match="Provide either video or images"):
            await async_client.pose()


@pytest.mark.asyncio
class TestAsyncJobs:
    async def test_reconstruct(self, async_client, tiny_video):
        job = await async_client.reconstruct(tiny_video)
        assert isinstance(job, ReconstructJob)
        assert job.job_id == "job_abc123"
        assert job.state == "processing"
        assert job.step == "queued"

    async def test_reconstruct_async_poll(self, async_client, tiny_video):
        job = await async_client.reconstruct(tiny_video)
        result = await job.async_poll()
        assert result["status"] == "complete"

    async def test_reconstruct_async_wait(self, async_client, tiny_video):
        job = await async_client.reconstruct(tiny_video)
        result = await job.async_wait()
        assert result["status"] == "complete"

    async def test_floorplan(self, async_client, tiny_video):
        job = await async_client.floorplan(tiny_video)
        assert isinstance(job, FloorplanJob)

    async def test_segment_3d(self, async_client, tiny_video):
        job = await async_client.segment_3d(tiny_video, prompt="find chairs")
        assert isinstance(job, Segment3DJob)


@pytest.mark.asyncio
class TestAsyncErrors:
    async def test_invalid_key(self, tiny_image):
        import httpx

        from .conftest import AsyncMockTransport

        client = AsyncClient(api_key="sf_invalid", base_url="https://mock.api")
        client._client = httpx.AsyncClient(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_invalid"},
            transport=AsyncMockTransport(),
        )
        with pytest.raises(SpatialForgeError) as exc_info:
            await client.depth(tiny_image)
        assert exc_info.value.status_code == 401
        await client.close()
