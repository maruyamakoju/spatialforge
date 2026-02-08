"""API endpoint tests for SpatialForge."""

from __future__ import annotations


def test_health(client):
    """Health endpoint returns OK without auth."""
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


def test_depth_no_auth(client):
    """Depth endpoint requires API key."""
    resp = client.post("/v1/depth")
    assert resp.status_code in (401, 422)


def test_depth_with_auth(client, api_key, sample_image_bytes):
    """Depth endpoint succeeds with valid auth and image."""
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


def test_depth_invalid_file(client, api_key):
    """Depth endpoint rejects non-image files."""
    resp = client.post(
        "/v1/depth",
        headers={"X-API-Key": api_key},
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data={"model": "large", "output_format": "png16", "metric": "true"},
    )
    assert resp.status_code == 400


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
