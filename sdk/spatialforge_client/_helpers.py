"""Shared helpers for sync and async SpatialForge clients.

Extracts common logic (file loading, response parsing, error extraction)
so that client.py and async_client.py focus only on HTTP transport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from .models import (
    CameraPose,
    DepthResult,
    MeasureResult,
    PoseResult,
)

# ── File loading ─────────────────────────────────────────


def load_file_data(
    source: str | Path | bytes,
    default_name: str = "file.bin",
) -> tuple[bytes, str]:
    """Load file data from a path string, Path object, or raw bytes.

    Returns:
        (file_bytes, filename) tuple.
    """
    if isinstance(source, str | Path):
        p = Path(source)
        return p.read_bytes(), p.name
    return source, default_name


def build_pose_files(
    video: str | Path | bytes | None,
    images: list[str | Path | bytes] | None,
) -> dict[str, Any]:
    """Build the multipart files dict for the /pose endpoint."""
    files: dict[str, Any] = {}

    if video is not None:
        data, name = load_file_data(video, "video.mp4")
        files["video"] = (name, data)
    elif images is not None:
        for i, img in enumerate(images):
            data, name = load_file_data(img, f"image_{i}.jpg")
            files[f"images_{i}"] = (name, data)
    else:
        raise ValueError("Provide either video or images")

    return files


def build_measure_form_data(
    point1: tuple[float, float],
    point2: tuple[float, float],
    reference_object: dict | None,
) -> dict[str, str]:
    """Build form data dict for the /measure endpoint."""
    form_data: dict[str, str] = {
        "points": json.dumps([
            {"x": point1[0], "y": point1[1]},
            {"x": point2[0], "y": point2[1]},
        ]),
    }
    if reference_object:
        form_data["reference_object"] = json.dumps(reference_object)
    return form_data


# ── Response parsing ─────────────────────────────────────


def parse_depth_response(data: dict) -> DepthResult:
    """Parse /depth JSON response into a DepthResult."""
    meta = data.get("metadata", {})
    return DepthResult(
        depth_map_url=data["depth_map_url"],
        width=meta.get("width", 0),
        height=meta.get("height", 0),
        min_depth_m=meta.get("min_depth_m", 0),
        max_depth_m=meta.get("max_depth_m", 0),
        focal_length_px=meta.get("focal_length_px"),
        confidence_mean=meta.get("confidence_mean", 0),
        processing_time_ms=data.get("processing_time_ms", 0),
        _raw=data,
    )


def parse_measure_response(data: dict) -> MeasureResult:
    """Parse /measure JSON response into a MeasureResult."""
    return MeasureResult(
        distance_m=data["distance_m"],
        confidence=data["confidence"],
        depth_at_points=data["depth_at_points"],
        calibration_method=data["calibration_method"],
        processing_time_ms=data.get("processing_time_ms", 0),
    )


def parse_pose_response(data: dict) -> PoseResult:
    """Parse /pose JSON response into a PoseResult."""
    poses = []
    for p in data.get("camera_poses", []):
        intr = p.get("intrinsics", {})
        poses.append(
            CameraPose(
                frame_index=p["frame_index"],
                rotation=p["rotation"],
                translation=p["translation"],
                fx=intr.get("fx", 0),
                fy=intr.get("fy", 0),
                cx=intr.get("cx", 0),
                cy=intr.get("cy", 0),
                width=intr.get("width", 0),
                height=intr.get("height", 0),
            )
        )

    return PoseResult(
        camera_poses=poses,
        pointcloud_url=data.get("pointcloud_url"),
        num_frames=data.get("num_frames", len(poses)),
        processing_time_ms=data.get("processing_time_ms", 0),
    )


# ── HTTP error helpers ───────────────────────────────────


def parse_error(resp: httpx.Response) -> str:
    """Extract error detail from HTTP response."""
    try:
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json"):
            return resp.json().get("detail", resp.text)
    except Exception:
        pass
    return resp.text


def get_retry_after(resp: httpx.Response) -> float | None:
    """Extract Retry-After header value in seconds."""
    val = resp.headers.get("retry-after")
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return None


# ── Async job construction ───────────────────────────────


def make_async_job(data: dict, job_cls: type, client: Any, endpoint: str) -> Any:
    """Create an async job model from API response data."""
    return job_cls(
        job_id=data["job_id"],
        status=data["status"],
        estimated_time_s=data.get("estimated_time_s"),
        _client=client,
        _endpoint=endpoint,
    )
