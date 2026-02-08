"""Unit tests for video processing utilities."""

from __future__ import annotations

import gc
import os
import tempfile

import cv2
import numpy as np
import pytest


def _create_test_video(path: str, duration_s: float = 3.0, fps: float = 30.0) -> None:
    """Create a minimal test video file."""
    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    n_frames = int(duration_s * fps)
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _safe_unlink(path: str) -> None:
    """Delete a temp file, tolerating Windows file lock errors."""
    try:
        gc.collect()  # Release any dangling cv2 handles
        os.unlink(path)
    except PermissionError:
        pass  # Windows may keep the file locked briefly; CI cleanup will handle it


def test_video_info():
    """Test VideoInfo extraction."""
    from spatialforge.utils.video import VideoInfo

    path = tempfile.mktemp(suffix=".mp4")
    try:
        _create_test_video(path, duration_s=2.0)
        info = VideoInfo(path)

        assert info.width == 320
        assert info.height == 240
        assert info.fps > 0
        assert info.duration_s > 0
    finally:
        _safe_unlink(path)


def test_extract_keyframes():
    """Test keyframe extraction."""
    from spatialforge.utils.video import extract_keyframes

    path = tempfile.mktemp(suffix=".mp4")
    try:
        _create_test_video(path, duration_s=3.0, fps=30.0)
        frames = extract_keyframes(path, target_fps=2.0, max_frames=10)

        assert len(frames) > 0
        assert len(frames) <= 10
        assert frames[0].shape[2] == 3  # RGB
        assert frames[0].dtype == np.uint8
    finally:
        _safe_unlink(path)


def test_validate_video_too_long():
    """Test that overly long videos are rejected."""
    from spatialforge.utils.video import validate_video

    path = tempfile.mktemp(suffix=".mp4")
    try:
        _create_test_video(path, duration_s=5.0)

        # Should pass with high limit
        validate_video(path, max_duration_s=10)

        # Should fail with low limit
        with pytest.raises(ValueError, match="exceeds maximum"):
            validate_video(path, max_duration_s=1)
    finally:
        _safe_unlink(path)
