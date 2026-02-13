"""API endpoint tests for SpatialForge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_health(client):
    """Health endpoint returns OK."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


def test_root(client):
    """Root endpoint returns service info."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "SpatialForge"


def test_security_txt(client):
    """security.txt endpoint is present for vulnerability disclosure."""
    resp = client.get("/.well-known/security.txt")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "Contact: mailto:security@spatialforge.example.com" in body
    assert "Canonical: https://spatialforge-demo.fly.dev/.well-known/security.txt" in body


def test_depth_with_auth(client, api_key, sample_image_bytes):
    """Depth endpoint succeeds with valid image (auth overridden in test)."""
    resp = client.post(
        "/v1/depth",
        headers={"X-API-Key": api_key},
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"model": "large", "output_format": "png16", "metric": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "depth_map_url" in data
    assert "metadata" in data
    assert data["metadata"]["width"] == 100
    assert data["metadata"]["height"] == 100


def test_depth_invalid_file(client, api_key):
    """Depth endpoint rejects non-image files."""
    resp = client.post(
        "/v1/depth",
        headers={"X-API-Key": api_key},
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data={"model": "large", "output_format": "png16", "metric": "true"},
    )
    assert resp.status_code == 400


def test_depth_colormap_url(client, api_key, sample_image_bytes):
    """Depth response includes colormap_url."""
    resp = client.post(
        "/v1/depth",
        headers={"X-API-Key": api_key},
        files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        data={"model": "large", "output_format": "png16", "metric": "true"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("colormap_url") is not None


def test_measure_invalid_points(client, api_key):
    """Measure endpoint rejects invalid points format."""
    resp = client.post(
        "/v1/measure",
        headers={"X-API-Key": api_key},
        files={"image": ("test.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        data={"points": "invalid json"},
    )
    assert resp.status_code == 400


def test_docs_accessible(client):
    """Swagger docs are accessible."""
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_redoc_accessible(client):
    """ReDoc is accessible."""
    resp = client.get("/redoc")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    ("endpoint", "form_data"),
    [
        ("/v1/reconstruct", {"quality": "standard", "output": "gaussian"}),
        ("/v1/floorplan", {"output_format": "svg"}),
        ("/v1/segment-3d", {"prompt": "chair", "output_3d_mask": "true", "output_bbox": "true"}),
    ],
)
def test_async_job_post_includes_state_step_contract(client, api_key, monkeypatch, endpoint, form_data):
    """Async job submission endpoints expose stable state/step fields."""
    from spatialforge.api.v1 import floorplan, reconstruct, segment
    from spatialforge.api.v1._video_job_utils import UploadedVideoJob
    from spatialforge.workers import tasks

    async def fake_validate_and_store_video(*_args, **_kwargs):
        return UploadedVideoJob(
            video_key="uploads/test.mp4",
            info=SimpleNamespace(duration_s=12.0),
        )

    monkeypatch.setattr(reconstruct, "validate_and_store_video", fake_validate_and_store_video)
    monkeypatch.setattr(floorplan, "validate_and_store_video", fake_validate_and_store_video)
    monkeypatch.setattr(segment, "validate_and_store_video", fake_validate_and_store_video)

    monkeypatch.setattr(
        tasks,
        "reconstruct_task",
        SimpleNamespace(delay=lambda **_kwargs: SimpleNamespace(id="job_reconstruct")),
    )
    monkeypatch.setattr(
        tasks,
        "floorplan_task",
        SimpleNamespace(delay=lambda **_kwargs: SimpleNamespace(id="job_floorplan")),
    )
    monkeypatch.setattr(
        tasks,
        "segment_3d_task",
        SimpleNamespace(delay=lambda **_kwargs: SimpleNamespace(id="job_segment3d")),
    )

    resp = client.post(
        endpoint,
        headers={"X-API-Key": api_key},
        files={"video": ("test.mp4", b"fake video bytes", "video/mp4")},
        data=form_data,
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["job_id"].startswith("job_")
    assert body["status"] == "processing"
    assert body["state"] == "processing"
    assert "step" in body
    assert body["step"] is None
