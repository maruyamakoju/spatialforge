"""API endpoint tests for SpatialForge."""

from __future__ import annotations


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
