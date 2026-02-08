"""Async SpatialForge API client.

Usage:
    import spatialforge_client as sf

    async with sf.AsyncClient(api_key="sf_xxx") as client:
        result = await client.depth("photo.jpg")
        print(result.min_depth_m, result.max_depth_m)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from .client import DEFAULT_BASE_URL, SpatialForgeError
from .models import (
    CameraPose,
    DepthResult,
    FloorplanJob,
    MeasureResult,
    PoseResult,
    ReconstructJob,
    Segment3DJob,
)


class AsyncClient:
    """Async SpatialForge API client.

    Args:
        api_key: Your SpatialForge API key (starts with 'sf_').
        base_url: API base URL. Defaults to production.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    def _parse_error(self, resp: httpx.Response) -> str:
        """Extract error detail from HTTP response."""
        try:
            ct = resp.headers.get("content-type", "")
            if ct.startswith("application/json"):
                return resp.json().get("detail", resp.text)
        except Exception:
            pass
        return resp.text

    async def _get(self, path: str) -> dict:
        resp = await self._client.get(path)
        if resp.status_code >= 400:
            raise SpatialForgeError(resp.status_code, self._parse_error(resp))
        return resp.json()

    async def _post_file(
        self, path: str, files: dict, data: dict | None = None
    ) -> dict:
        resp = await self._client.post(path, files=files, data=data or {})
        if resp.status_code >= 400:
            raise SpatialForgeError(resp.status_code, self._parse_error(resp))
        return resp.json()

    # ── /depth ──────────────────────────────────────────────

    async def depth(
        self,
        image: str | Path | bytes,
        model: str = "large",
        output_format: str = "png16",
        metric: bool = True,
    ) -> DepthResult:
        """Estimate depth from an image.

        Args:
            image: Path to image file, or raw bytes.
            model: Model size — 'giant', 'large', 'base', 'small'.
            output_format: Output format — 'png16', 'exr', 'npy'.
            metric: If True, output is in metric depth (meters).

        Returns:
            DepthResult with depth map URL and metadata.
        """
        if isinstance(image, str | Path):
            image_path = Path(image)
            file_data = image_path.read_bytes()
            filename = image_path.name
        else:
            file_data = image
            filename = "image.jpg"

        data = await self._post_file(
            "/v1/depth",
            files={"image": (filename, file_data)},
            data={
                "model": model,
                "output_format": output_format,
                "metric": str(metric).lower(),
            },
        )

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

    # ── /measure ────────────────────────────────────────────

    async def measure(
        self,
        image: str | Path | bytes,
        point1: tuple[float, float],
        point2: tuple[float, float],
        reference_object: dict | None = None,
    ) -> MeasureResult:
        """Measure distance between two points in an image.

        Args:
            image: Path to image file, or raw bytes.
            point1: (x, y) coordinates of first point.
            point2: (x, y) coordinates of second point.
            reference_object: Optional dict with 'type' and 'bbox' keys.

        Returns:
            MeasureResult with distance in meters.
        """
        if isinstance(image, str | Path):
            image_path = Path(image)
            file_data = image_path.read_bytes()
            filename = image_path.name
        else:
            file_data = image
            filename = "image.jpg"

        form_data: dict[str, str] = {
            "points": json.dumps(
                [
                    {"x": point1[0], "y": point1[1]},
                    {"x": point2[0], "y": point2[1]},
                ]
            ),
        }
        if reference_object:
            form_data["reference_object"] = json.dumps(reference_object)

        data = await self._post_file(
            "/v1/measure",
            files={"image": (filename, file_data)},
            data=form_data,
        )

        return MeasureResult(
            distance_m=data["distance_m"],
            confidence=data["confidence"],
            depth_at_points=data["depth_at_points"],
            calibration_method=data["calibration_method"],
            processing_time_ms=data.get("processing_time_ms", 0),
        )

    # ── /pose ───────────────────────────────────────────────

    async def pose(
        self,
        video: str | Path | bytes | None = None,
        images: list[str | Path | bytes] | None = None,
        output_pointcloud: bool = False,
    ) -> PoseResult:
        """Estimate camera poses from video or multiple images.

        Args:
            video: Path to video file, or raw bytes.
            images: List of image paths or raw bytes.
            output_pointcloud: Whether to output a sparse point cloud.

        Returns:
            PoseResult with camera poses.
        """
        files: dict[str, Any] = {}

        if video is not None:
            if isinstance(video, str | Path):
                video_path = Path(video)
                files["video"] = (video_path.name, video_path.read_bytes())
            else:
                files["video"] = ("video.mp4", video)
        elif images is not None:
            for i, img in enumerate(images):
                if isinstance(img, str | Path):
                    img_path = Path(img)
                    files[f"images_{i}"] = (img_path.name, img_path.read_bytes())
                else:
                    files[f"images_{i}"] = (f"image_{i}.jpg", img)
        else:
            raise ValueError("Provide either video or images")

        data = await self._post_file(
            "/v1/pose",
            files=files,
            data={"output_pointcloud": str(output_pointcloud).lower()},
        )

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

    # ── /reconstruct ────────────────────────────────────────

    async def reconstruct(
        self,
        video: str | Path | bytes,
        quality: str = "standard",
        output: str = "gaussian",
    ) -> ReconstructJob:
        """Start async 3D reconstruction from video.

        Args:
            video: Path to video file, or raw bytes.
            quality: 'draft', 'standard', or 'high'.
            output: 'gaussian', 'pointcloud', or 'mesh'.

        Returns:
            ReconstructJob — call await .async_wait() to block until complete.
        """
        if isinstance(video, str | Path):
            video_path = Path(video)
            file_data = video_path.read_bytes()
            filename = video_path.name
        else:
            file_data = video
            filename = "video.mp4"

        data = await self._post_file(
            "/v1/reconstruct",
            files={"video": (filename, file_data)},
            data={"quality": quality, "output": output},
        )

        return ReconstructJob(
            job_id=data["job_id"],
            status=data["status"],
            estimated_time_s=data.get("estimated_time_s"),
            _client=self,
            _endpoint="/v1/reconstruct",
        )

    # ── /floorplan ──────────────────────────────────────────

    async def floorplan(
        self,
        video: str | Path | bytes,
        output_format: str = "svg",
    ) -> FloorplanJob:
        """Start async floorplan generation from walkthrough video.

        Args:
            video: Path to video file, or raw bytes.
            output_format: 'svg', 'dxf', or 'json'.

        Returns:
            FloorplanJob — call await .async_wait() to block until complete.
        """
        if isinstance(video, str | Path):
            video_path = Path(video)
            file_data = video_path.read_bytes()
            filename = video_path.name
        else:
            file_data = video
            filename = "video.mp4"

        data = await self._post_file(
            "/v1/floorplan",
            files={"video": (filename, file_data)},
            data={"output_format": output_format},
        )

        return FloorplanJob(
            job_id=data["job_id"],
            status=data["status"],
            estimated_time_s=data.get("estimated_time_s"),
            _client=self,
            _endpoint="/v1/floorplan",
        )

    # ── /segment-3d ─────────────────────────────────────────

    async def segment_3d(
        self,
        video: str | Path | bytes,
        prompt: str,
    ) -> Segment3DJob:
        """Start async 3D segmentation with text prompt.

        Args:
            video: Path to video file, or raw bytes.
            prompt: Natural language description of objects to segment.

        Returns:
            Segment3DJob — call await .async_wait() to block until complete.
        """
        if isinstance(video, str | Path):
            video_path = Path(video)
            file_data = video_path.read_bytes()
            filename = video_path.name
        else:
            file_data = video
            filename = "video.mp4"

        data = await self._post_file(
            "/v1/segment-3d",
            files={"video": (filename, file_data)},
            data={"prompt": prompt},
        )

        return Segment3DJob(
            job_id=data["job_id"],
            status=data["status"],
            estimated_time_s=data.get("estimated_time_s"),
            _client=self,
            _endpoint="/v1/segment-3d",
        )
