"""Video processing utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VideoInfo:
    """Metadata about a video file."""

    def __init__(self, path: str | Path) -> None:
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {path}")
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration_s = self.frame_count / self.fps if self.fps > 0 else 0
        finally:
            cap.release()


def extract_keyframes(
    video_path: str | Path,
    target_fps: float = 2.0,
    max_frames: int = 200,
) -> list[NDArray[np.uint8]]:
    """Extract keyframes from video at a target FPS.

    Returns list of RGB numpy arrays.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / target_fps))

        frames: list[NDArray[np.uint8]] = []
        frame_idx = 0

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            frame_idx += 1
    finally:
        cap.release()

    return frames


def save_uploaded_video(data: bytes, suffix: str = ".mp4") -> Path:
    """Save uploaded video bytes to a temporary file. Caller is responsible for cleanup."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def validate_video(path: str | Path, max_duration_s: int = 120) -> VideoInfo:
    """Validate video file and return its info. Raises ValueError if invalid."""
    info = VideoInfo(path)
    if info.duration_s > max_duration_s:
        raise ValueError(f"Video duration {info.duration_s:.1f}s exceeds maximum {max_duration_s}s")
    if info.width < 64 or info.height < 64:
        raise ValueError(f"Video resolution {info.width}x{info.height} is too small")
    return info
