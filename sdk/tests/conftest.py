"""Shared fixtures for SDK tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest


# ── Mock API responses ────────────────────────────────────

DEPTH_RESPONSE = {
    "depth_map_url": "https://example.com/depth.png",
    "processing_time_ms": 1234.5,
    "metadata": {
        "width": 640,
        "height": 480,
        "min_depth_m": 0.5,
        "max_depth_m": 12.3,
        "focal_length_px": 525.0,
        "confidence_mean": 0.95,
    },
}

MEASURE_RESPONSE = {
    "distance_m": 2.45,
    "confidence": 0.87,
    "depth_at_points": [1.2, 3.7],
    "calibration_method": "reference_object",
    "processing_time_ms": 890.1,
}

POSE_RESPONSE = {
    "camera_poses": [
        {
            "frame_index": 0,
            "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0.0, 0.0, 0.0],
            "intrinsics": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480,
            },
        },
        {
            "frame_index": 1,
            "rotation": [[0.99, 0.01, 0], [-0.01, 0.99, 0], [0, 0, 1]],
            "translation": [0.1, 0.0, 0.05],
            "intrinsics": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": 640,
                "height": 480,
            },
        },
    ],
    "pointcloud_url": None,
    "num_frames": 2,
    "processing_time_ms": 3456.7,
}

ASYNC_JOB_RESPONSE = {
    "job_id": "job_abc123",
    "status": "processing",
    "estimated_time_s": 30.0,
}

ASYNC_JOB_COMPLETE = {
    "job_id": "job_abc123",
    "status": "complete",
    "scene_url": "https://example.com/scene.ply",
    "floor_area_m2": 45.2,
}

ASYNC_JOB_FAILED = {
    "job_id": "job_abc123",
    "status": "failed",
    "error": "Video too short for reconstruction",
}

ERROR_401 = {"detail": "Invalid or disabled API key."}
ERROR_429 = {"detail": "Rate limit exceeded. Try again in 60 seconds."}


def _route(request: httpx.Request) -> httpx.Response:
    """Mock HTTP handler that routes requests to appropriate responses."""
    path = request.url.path

    # Auth check
    api_key = request.headers.get("X-API-Key", "")
    if api_key == "sf_invalid":
        return httpx.Response(401, json=ERROR_401)
    if api_key == "sf_ratelimited":
        return httpx.Response(429, json=ERROR_429)

    if path == "/v1/depth" and request.method == "POST":
        return httpx.Response(200, json=DEPTH_RESPONSE)

    if path == "/v1/measure" and request.method == "POST":
        return httpx.Response(200, json=MEASURE_RESPONSE)

    if path == "/v1/pose" and request.method == "POST":
        return httpx.Response(200, json=POSE_RESPONSE)

    if path == "/v1/reconstruct" and request.method == "POST":
        return httpx.Response(200, json=ASYNC_JOB_RESPONSE)

    if path == "/v1/floorplan" and request.method == "POST":
        return httpx.Response(200, json=ASYNC_JOB_RESPONSE)

    if path == "/v1/segment-3d" and request.method == "POST":
        return httpx.Response(200, json=ASYNC_JOB_RESPONSE)

    # Job polling
    if "/job_abc123" in path and request.method == "GET":
        return httpx.Response(200, json=ASYNC_JOB_COMPLETE)

    return httpx.Response(404, json={"detail": "Not found"})


class MockTransport(httpx.BaseTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return _route(request)


class AsyncMockTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return _route(request)


@pytest.fixture
def sync_client():
    """Create a sync client with mock transport."""
    from spatialforge_client import Client

    client = Client(api_key="sf_test_key", base_url="https://mock.api")
    client._client = httpx.Client(
        base_url="https://mock.api",
        headers={"X-API-Key": "sf_test_key"},
        transport=MockTransport(),
    )
    yield client
    client.close()


@pytest.fixture
def async_client():
    """Create an async client with mock transport."""
    from spatialforge_client import AsyncClient

    client = AsyncClient(api_key="sf_test_key", base_url="https://mock.api")
    client._client = httpx.AsyncClient(
        base_url="https://mock.api",
        headers={"X-API-Key": "sf_test_key"},
        transport=AsyncMockTransport(),
    )
    return client


@pytest.fixture
def invalid_key_client():
    """Create a sync client with invalid API key."""
    from spatialforge_client import Client

    client = Client(api_key="sf_invalid", base_url="https://mock.api")
    client._client = httpx.Client(
        base_url="https://mock.api",
        headers={"X-API-Key": "sf_invalid"},
        transport=MockTransport(),
    )
    yield client
    client.close()


@pytest.fixture
def tiny_image(tmp_path):
    """Create a tiny valid JPEG for testing."""
    # Minimal JPEG: SOI + APP0 + minimal content + EOI
    jpeg_bytes = bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
        0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
        0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9,
    ])
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(jpeg_bytes)
    return img_path


@pytest.fixture
def tiny_video(tmp_path):
    """Create a tiny file for testing video uploads."""
    vid_path = tmp_path / "test.mp4"
    vid_path.write_bytes(b"\x00" * 64)
    return vid_path
