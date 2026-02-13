"""Tests for the synchronous SpatialForge client."""

from __future__ import annotations

import pytest
from spatialforge_client import (
    Client,
    DepthResult,
    FloorplanJob,
    MeasureResult,
    PoseResult,
    ReconstructJob,
    Segment3DJob,
    SpatialForgeError,
)


class TestClientInit:
    def test_default_base_url(self):
        c = Client(api_key="sf_test")
        assert "spatialforge" in c._base_url
        c.close()

    def test_custom_base_url(self):
        c = Client(api_key="sf_test", base_url="http://localhost:8000/")
        assert c._base_url == "http://localhost:8000"
        c.close()

    def test_context_manager(self, sync_client):
        with sync_client as c:
            assert c is sync_client


class TestDepth:
    def test_depth_from_file(self, sync_client, tiny_image):
        result = sync_client.depth(tiny_image)
        assert isinstance(result, DepthResult)
        assert result.width == 640
        assert result.height == 480
        assert result.min_depth_m == 0.5
        assert result.max_depth_m == 12.3
        assert result.focal_length_px == 525.0
        assert result.confidence_mean == 0.95
        assert result.processing_time_ms == 1234.5
        assert result.depth_map_url.startswith("https://")

    def test_depth_from_bytes(self, sync_client):
        result = sync_client.depth(b"\xff\xd8\xff\xe0" + b"\x00" * 20)
        assert isinstance(result, DepthResult)
        assert result.width == 640

    def test_depth_from_string_path(self, sync_client, tiny_image):
        result = sync_client.depth(str(tiny_image))
        assert isinstance(result, DepthResult)

    def test_depth_raw_data(self, sync_client, tiny_image):
        result = sync_client.depth(tiny_image)
        assert isinstance(result._raw, dict)
        assert "depth_map_url" in result._raw


class TestMeasure:
    def test_measure_basic(self, sync_client, tiny_image):
        result = sync_client.measure(tiny_image, point1=(100, 200), point2=(500, 200))
        assert isinstance(result, MeasureResult)
        assert result.distance_m == 2.45
        assert result.confidence == 0.87
        assert result.calibration_method == "reference_object"

    def test_measure_distance_conversions(self, sync_client, tiny_image):
        result = sync_client.measure(tiny_image, (0, 0), (100, 100))
        assert result.distance_cm == pytest.approx(245.0)
        assert result.distance_mm == pytest.approx(2450.0)

    def test_measure_with_reference(self, sync_client, tiny_image):
        ref = {"type": "door", "bbox": [10, 20, 100, 400]}
        result = sync_client.measure(
            tiny_image, (0, 0), (100, 100), reference_object=ref
        )
        assert isinstance(result, MeasureResult)


class TestPose:
    def test_pose_from_video(self, sync_client, tiny_video):
        result = sync_client.pose(video=tiny_video)
        assert isinstance(result, PoseResult)
        assert result.num_frames == 2
        assert len(result.camera_poses) == 2

    def test_pose_camera_intrinsics(self, sync_client, tiny_video):
        result = sync_client.pose(video=tiny_video)
        pose = result.camera_poses[0]
        assert pose.fx == 525.0
        assert pose.fy == 525.0
        assert pose.cx == 320.0
        assert pose.cy == 240.0
        assert pose.width == 640
        assert pose.height == 480

    def test_pose_from_images(self, sync_client, tiny_image):
        result = sync_client.pose(images=[tiny_image, tiny_image])
        assert isinstance(result, PoseResult)

    def test_pose_from_bytes(self, sync_client):
        result = sync_client.pose(video=b"\x00" * 64)
        assert isinstance(result, PoseResult)

    def test_pose_no_input_raises(self, sync_client):
        with pytest.raises(ValueError, match="Provide either video or images"):
            sync_client.pose()


class TestAsyncJobs:
    def test_reconstruct(self, sync_client, tiny_video):
        job = sync_client.reconstruct(tiny_video)
        assert isinstance(job, ReconstructJob)
        assert job.job_id == "job_abc123"
        assert job.status == "processing"
        assert job.state == "processing"
        assert job.step == "queued"

    def test_reconstruct_poll(self, sync_client, tiny_video):
        job = sync_client.reconstruct(tiny_video)
        result = job.poll()
        assert result["status"] == "complete"

    def test_reconstruct_wait(self, sync_client, tiny_video):
        job = sync_client.reconstruct(tiny_video)
        result = job.wait()
        assert result["status"] == "complete"
        assert "scene_url" in result

    def test_floorplan(self, sync_client, tiny_video):
        job = sync_client.floorplan(tiny_video)
        assert isinstance(job, FloorplanJob)
        assert job.job_id == "job_abc123"

    def test_segment_3d(self, sync_client, tiny_video):
        job = sync_client.segment_3d(tiny_video, prompt="find the chairs")
        assert isinstance(job, Segment3DJob)
        assert job.job_id == "job_abc123"


class TestErrorHandling:
    def test_invalid_key_depth(self, invalid_key_client, tiny_image):
        with pytest.raises(SpatialForgeError) as exc_info:
            invalid_key_client.depth(tiny_image)
        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail

    def test_invalid_key_measure(self, invalid_key_client, tiny_image):
        with pytest.raises(SpatialForgeError) as exc_info:
            invalid_key_client.measure(tiny_image, (0, 0), (1, 1))
        assert exc_info.value.status_code == 401

    def test_error_str(self):
        err = SpatialForgeError(429, "Rate limited")
        assert "429" in str(err)
        assert "Rate limited" in str(err)
