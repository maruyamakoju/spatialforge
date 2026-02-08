"""Unit tests for depth estimation engine."""

from __future__ import annotations

import numpy as np
import pytest


def test_depth_to_png16():
    """Test depth map to PNG16 encoding."""
    from spatialforge.utils.image import depth_to_png16

    depth = np.random.rand(100, 100).astype(np.float32) * 10.0
    png_bytes = depth_to_png16(depth, max_depth=10.0)

    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0
    # PNG signature
    assert png_bytes[:4] == b"\x89PNG"


def test_depth_to_npy():
    """Test depth map to NPY encoding."""
    from spatialforge.utils.image import depth_to_npy

    depth = np.random.rand(50, 50).astype(np.float32)
    npy_bytes = depth_to_npy(depth)

    assert isinstance(npy_bytes, bytes)
    # Verify roundtrip
    import io

    loaded = np.load(io.BytesIO(npy_bytes))
    np.testing.assert_array_equal(loaded, depth)


def test_depth_to_colormap():
    """Test depth map to colorized visualization."""
    from spatialforge.utils.image import depth_to_colormap

    depth = np.random.rand(100, 100).astype(np.float32) * 5.0
    jpg_bytes = depth_to_colormap(depth)

    assert isinstance(jpg_bytes, bytes)
    assert len(jpg_bytes) > 0
    # JPEG signature
    assert jpg_bytes[:2] == b"\xff\xd8"


def test_load_image_rgb():
    """Test image loading from bytes."""
    import io

    from PIL import Image

    from spatialforge.utils.image import load_image_rgb

    # Create a test image
    img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    rgb = load_image_rgb(buf.getvalue())
    assert rgb.shape == (50, 50, 3)
    assert rgb.dtype == np.uint8


def test_load_image_invalid():
    """Test that invalid image bytes raise ValueError."""
    from spatialforge.utils.image import load_image_rgb

    with pytest.raises(ValueError, match="Failed to decode"):
        load_image_rgb(b"not an image")


def test_resize_if_needed():
    """Test image resizing."""
    from spatialforge.utils.image import resize_if_needed

    large = np.zeros((8000, 6000, 3), dtype=np.uint8)
    resized = resize_if_needed(large, max_size=4096)
    assert max(resized.shape[:2]) <= 4096

    small = np.zeros((100, 100, 3), dtype=np.uint8)
    not_resized = resize_if_needed(small, max_size=4096)
    assert not_resized.shape == (100, 100, 3)
