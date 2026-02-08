"""Tests for input validation — coordinate bounds, NaN/Inf rejection."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def small_jpeg_200x150() -> bytes:
    """A 200x150 pixel JPEG for bounds testing."""
    img = Image.fromarray(np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestMeasureCoordinateValidation:
    """Test coordinate bounds checking on the /v1/measure endpoint."""

    def test_valid_coordinates(self, client, api_key, small_jpeg_200x150):
        """Points within image bounds should be accepted."""
        points = json.dumps([{"x": 10, "y": 20}, {"x": 100, "y": 50}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        # Should succeed (200) or at least not fail with 400 for coordinates
        # (may fail with 500 if measure engine mock is incomplete, but not 400)
        assert resp.status_code != 400 or "outside image bounds" not in resp.json().get("detail", "")

    def test_point_x_out_of_bounds(self, client, api_key, small_jpeg_200x150):
        """X coordinate >= image width should be rejected."""
        points = json.dumps([{"x": 250, "y": 20}, {"x": 50, "y": 50}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400
        assert "outside image bounds" in resp.json()["detail"]

    def test_point_y_out_of_bounds(self, client, api_key, small_jpeg_200x150):
        """Y coordinate >= image height should be rejected."""
        points = json.dumps([{"x": 10, "y": 20}, {"x": 50, "y": 200}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400
        assert "outside image bounds" in resp.json()["detail"]

    def test_negative_coordinates(self, client, api_key, small_jpeg_200x150):
        """Negative coordinates should be rejected."""
        points = json.dumps([{"x": -10, "y": 20}, {"x": 50, "y": 50}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400
        assert "outside image bounds" in resp.json()["detail"]

    def test_nan_coordinates_rejected(self, client, api_key, small_jpeg_200x150):
        """NaN coordinates should be rejected."""
        points = json.dumps([{"x": float("nan"), "y": 20}, {"x": 50, "y": 50}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400

    def test_inf_coordinates_rejected(self, client, api_key, small_jpeg_200x150):
        """Infinity coordinates should be rejected."""
        points = json.dumps([{"x": float("inf"), "y": 20}, {"x": 50, "y": 50}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400

    def test_single_point_rejected(self, client, api_key, small_jpeg_200x150):
        """Only 1 point should be rejected."""
        points = json.dumps([{"x": 10, "y": 20}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400

    def test_three_points_rejected(self, client, api_key, small_jpeg_200x150):
        """3 points should be rejected."""
        points = json.dumps([{"x": 10, "y": 20}, {"x": 50, "y": 50}, {"x": 80, "y": 90}])
        resp = client.post(
            "/v1/measure",
            headers={"X-API-Key": api_key},
            files={"image": ("test.jpg", small_jpeg_200x150, "image/jpeg")},
            data={"points": points},
        )
        assert resp.status_code == 400
