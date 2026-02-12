"""SpatialForge API client.

Usage:
    import spatialforge_client as sf

    client = sf.Client(api_key="sf_xxx")
    result = client.depth("photo.jpg")
    print(result.min_depth_m, result.max_depth_m)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from ._helpers import (
    build_measure_form_data,
    build_pose_files,
    get_retry_after,
    load_file_data,
    make_async_job,
    parse_depth_response,
    parse_error,
    parse_measure_response,
    parse_pose_response,
)
from .exceptions import RETRYABLE_STATUS_CODES, SpatialForgeError, raise_for_status
from .models import (
    DepthResult,
    FloorplanJob,
    MeasureResult,
    PoseResult,
    ReconstructJob,
    Segment3DJob,
)

DEFAULT_BASE_URL = "https://spatialforge-demo.fly.dev"

logger = logging.getLogger("spatialforge_client")


class Client:
    """SpatialForge API client.

    Args:
        api_key: Your SpatialForge API key (starts with 'sf_').
        base_url: API base URL. Defaults to production.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for transient errors (429, 5xx).
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _request_with_retry(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry for transient errors."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.request(method, path, **kwargs)
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < self._max_retries:
                    delay = self._retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Request timeout (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, self._max_retries + 1, delay,
                    )
                    time.sleep(delay)
                    continue
                raise SpatialForgeError(504, f"Request timed out after {self._max_retries + 1} attempts") from e

            if resp.status_code < 400:
                return resp

            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < self._max_retries:
                retry = get_retry_after(resp)
                delay = retry if retry is not None else self._retry_base_delay * (2 ** attempt)
                logger.warning(
                    "HTTP %d (attempt %d/%d), retrying in %.1fs",
                    resp.status_code, attempt + 1, self._max_retries + 1, delay,
                )
                time.sleep(delay)
                continue

            raise_for_status(
                resp.status_code,
                parse_error(resp),
                retry_after=get_retry_after(resp),
            )

        raise SpatialForgeError(500, "Unexpected retry exhaustion") from last_exc

    def _get(self, path: str) -> dict:
        resp = self._request_with_retry("GET", path)
        return resp.json()

    def _post_file(self, path: str, files: dict, data: dict | None = None) -> dict:
        resp = self._request_with_retry("POST", path, files=files, data=data or {})
        return resp.json()

    # ── /depth ──────────────────────────────────────────────

    def depth(
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
        file_data, filename = load_file_data(image, "image.jpg")
        data = self._post_file(
            "/v1/depth",
            files={"image": (filename, file_data)},
            data={"model": model, "output_format": output_format, "metric": str(metric).lower()},
        )
        return parse_depth_response(data)

    # ── /measure ────────────────────────────────────────────

    def measure(
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
            reference_object: Optional dict with 'type' and 'bbox' keys for calibration.

        Returns:
            MeasureResult with distance in meters.
        """
        file_data, filename = load_file_data(image, "image.jpg")
        data = self._post_file(
            "/v1/measure",
            files={"image": (filename, file_data)},
            data=build_measure_form_data(point1, point2, reference_object),
        )
        return parse_measure_response(data)

    # ── /pose ───────────────────────────────────────────────

    def pose(
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
        data = self._post_file(
            "/v1/pose",
            files=build_pose_files(video, images),
            data={"output_pointcloud": str(output_pointcloud).lower()},
        )
        return parse_pose_response(data)

    # ── /reconstruct ────────────────────────────────────────

    def reconstruct(
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
            ReconstructJob — call .wait() to block until complete.
        """
        file_data, filename = load_file_data(video, "video.mp4")
        data = self._post_file(
            "/v1/reconstruct",
            files={"video": (filename, file_data)},
            data={"quality": quality, "output": output},
        )
        return make_async_job(data, ReconstructJob, self, "/v1/reconstruct")

    # ── /floorplan ──────────────────────────────────────────

    def floorplan(
        self,
        video: str | Path | bytes,
        output_format: str = "svg",
    ) -> FloorplanJob:
        """Start async floorplan generation from walkthrough video.

        Args:
            video: Path to video file, or raw bytes.
            output_format: 'svg', 'dxf', or 'json'.

        Returns:
            FloorplanJob — call .wait() to block until complete.
        """
        file_data, filename = load_file_data(video, "video.mp4")
        data = self._post_file(
            "/v1/floorplan",
            files={"video": (filename, file_data)},
            data={"output_format": output_format},
        )
        return make_async_job(data, FloorplanJob, self, "/v1/floorplan")

    # ── /segment-3d ─────────────────────────────────────────

    def segment_3d(
        self,
        video: str | Path | bytes,
        prompt: str,
    ) -> Segment3DJob:
        """Start async 3D segmentation with text prompt.

        Args:
            video: Path to video file, or raw bytes.
            prompt: Natural language description of objects to segment.

        Returns:
            Segment3DJob — call .wait() to block until complete.
        """
        file_data, filename = load_file_data(video, "video.mp4")
        data = self._post_file(
            "/v1/segment-3d",
            files={"video": (filename, file_data)},
            data={"prompt": prompt},
        )
        return make_async_job(data, Segment3DJob, self, "/v1/segment-3d")
