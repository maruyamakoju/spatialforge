"""Image processing utilities."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_image_from_bytes(data: bytes) -> NDArray[np.uint8]:
    """Load an image from raw bytes into a BGR numpy array (OpenCV format)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image. Ensure the file is a valid image (JPEG/PNG/WebP).")
    return img


def load_image_rgb(data: bytes) -> NDArray[np.uint8]:
    """Load an image from raw bytes into an RGB numpy array."""
    bgr = load_image_from_bytes(data)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_if_needed(img: NDArray[np.uint8], max_size: int = 4096) -> NDArray[np.uint8]:
    """Resize image so the longer side does not exceed max_size."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def depth_to_png16(depth: NDArray[np.float32], max_depth: float | None = None) -> bytes:
    """Convert a float32 depth map to 16-bit PNG bytes.

    Normalizes depth values to 0-65535 range where 0 = max_depth, 65535 = 0m.
    """
    if max_depth is None:
        max_depth = float(np.max(depth))
    if max_depth <= 0:
        max_depth = 1.0

    # Invert: closer = higher value (standard depth map convention)
    normalized = np.clip(1.0 - depth / max_depth, 0, 1)
    uint16 = (normalized * 65535).astype(np.uint16)

    success, encoded = cv2.imencode(".png", uint16)
    if not success:
        raise RuntimeError("Failed to encode depth map as PNG16")
    return encoded.tobytes()


def depth_to_npy(depth: NDArray[np.float32]) -> bytes:
    """Serialize a float32 depth map to .npy bytes."""
    buf = io.BytesIO()
    np.save(buf, depth)
    return buf.getvalue()


def depth_to_colormap(depth: NDArray[np.float32]) -> bytes:
    """Convert a float32 depth map to a colorized visualization (JPEG)."""
    d_min, d_max = float(np.min(depth)), float(np.max(depth))
    if d_max - d_min < 1e-6:
        d_max = d_min + 1.0
    normalized = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    success, encoded = cv2.imencode(".jpg", colorized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise RuntimeError("Failed to encode depth colormap")
    return encoded.tobytes()


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
