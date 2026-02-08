"""Comprehensive error handling tests for all API endpoints."""

from __future__ import annotations

import json

# ── Depth endpoint error handling ────────────────────────


class TestDepthErrors:
    def test_no_file_returns_422(self, client, api_key):
        """Missing image file returns 422 (validation error)."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            data={"model": "large"},
        )
        assert resp.status_code == 422

    def test_empty_file_returns_error(self, client, api_key):
        """Empty file body returns an error (400 or 500)."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("empty.jpg", b"", "image/jpeg")},
            data={"model": "large", "output_format": "png16", "metric": "true"},
        )
        assert resp.status_code >= 400

    def test_text_file_returns_400(self, client, api_key):
        """Non-image content returns 400."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("test.txt", b"hello world", "text/plain")},
            data={"model": "large"},
        )
        assert resp.status_code == 400

    def test_valid_image_returns_depth_metadata(self, client, api_key, sample_image_bytes):
        """Valid image returns proper depth metadata."""
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
        meta = data["metadata"]
        assert "width" in meta
        assert "height" in meta
        assert "min_depth_m" in meta
        assert "max_depth_m" in meta
        assert meta["min_depth_m"] >= 0
        assert meta["max_depth_m"] > meta["min_depth_m"]
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] > 0

    def test_depth_with_npy_output(self, client, api_key, sample_image_bytes):
        """NPY output format returns valid response."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"model": "large", "output_format": "npy", "metric": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "depth_map_url" in data

    def test_depth_small_model(self, client, api_key, sample_image_bytes):
        """Small model variant works."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"model": "small", "output_format": "png16", "metric": "false"},
        )
        assert resp.status_code == 200


# ── Measure endpoint error handling ──────────────────────


class TestMeasureErrors:
    def test_invalid_json_points(self, client, api_key, sample_image_bytes):
        """Invalid JSON in points returns 400."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"points": "not json"},
        )
        assert resp.status_code == 400

    def test_single_point_returns_400(self, client, api_key, sample_image_bytes):
        """Single point instead of two returns 400."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"points": json.dumps([{"x": 10, "y": 10}])},
        )
        assert resp.status_code == 400

    def test_missing_points_returns_422(self, client, api_key, sample_image_bytes):
        """Missing points parameter returns 422."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_no_image_returns_422(self, client, api_key):
        """Missing image returns 422."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            data={"points": json.dumps([{"x": 10, "y": 10}, {"x": 50, "y": 50}])},
        )
        assert resp.status_code == 422

    def test_valid_measure_returns_distance(self, client, api_key, sample_image_bytes):
        """Valid measure request returns distance data."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"points": json.dumps([{"x": 10, "y": 10}, {"x": 50, "y": 50}])},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "distance_m" in data
        assert "confidence" in data
        assert data["distance_m"] >= 0

    def test_points_out_of_bounds_returns_400(self, client, api_key, sample_image_bytes):
        """Points outside image bounds returns 400."""
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"points": json.dumps([{"x": 10, "y": 10}, {"x": 9999, "y": 9999}])},
        )
        assert resp.status_code == 400
        assert "outside image bounds" in resp.json()["detail"].lower() or "bounds" in resp.json()["detail"].lower()


# ── Pose endpoint error handling ─────────────────────────


class TestPoseErrors:
    def test_no_input_returns_400(self, client, api_key):
        """No video or images returns 400."""
        resp = client.post(
            "/v1/pose",
            headers={"X-API-Key": api_key},
            data={"output_pointcloud": "false"},
        )
        assert resp.status_code in (400, 422)


# ── Reconstruct endpoint error handling ──────────────────


class TestReconstructErrors:
    def test_no_video_returns_422(self, client, api_key):
        """Missing video returns 422."""
        resp = client.post(
            "/v1/reconstruct",
            headers={"X-API-Key": api_key},
            data={"quality": "standard"},
        )
        assert resp.status_code == 422


# ── Error response format ────────────────────────────────


class TestErrorResponseFormat:
    def test_400_has_detail_field(self, client, api_key):
        """400 errors include a detail field."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("test.txt", b"not an image", "text/plain")},
            data={"model": "large"},
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)
        assert len(data["detail"]) > 0

    def test_422_has_detail_field(self, client, api_key):
        """422 errors include detail with validation info."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 422
        data = resp.json()
        assert "detail" in data

    def test_error_response_is_json(self, client, api_key):
        """Error responses have JSON content type."""
        resp = client.post(
            "/v1/depth",
            headers={"X-API-Key": api_key},
            files={"image": ("bad.txt", b"not image", "text/plain")},
            data={"model": "large"},
        )
        assert "application/json" in resp.headers.get("content-type", "")
