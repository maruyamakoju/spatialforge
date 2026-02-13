"""Data models for the SpatialForge SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DepthResult:
    """Result from /depth endpoint."""

    depth_map_url: str
    width: int
    height: int
    min_depth_m: float
    max_depth_m: float
    focal_length_px: float | None
    confidence_mean: float
    processing_time_ms: float
    _raw: dict = field(default_factory=dict, repr=False)

    def save_depth_map(self, path: str | Path) -> Path:
        """Download and save the depth map to a local file."""
        import httpx

        resp = httpx.get(self.depth_map_url)
        resp.raise_for_status()
        out = Path(path)
        out.write_bytes(resp.content)
        return out


@dataclass
class MeasureResult:
    """Result from /measure endpoint."""

    distance_m: float
    confidence: float
    depth_at_points: list[float]
    calibration_method: str
    processing_time_ms: float

    @property
    def distance_cm(self) -> float:
        return self.distance_m * 100

    @property
    def distance_mm(self) -> float:
        return self.distance_m * 1000


@dataclass
class CameraPose:
    """Single camera pose."""

    frame_index: int
    rotation: list[list[float]]
    translation: list[float]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class PoseResult:
    """Result from /pose endpoint."""

    camera_poses: list[CameraPose]
    pointcloud_url: str | None
    num_frames: int
    processing_time_ms: float


@dataclass
class AsyncJob:
    """Base class for async job results."""

    job_id: str
    status: str
    state: str | None = None
    step: str | None = None
    _client: Any = field(default=None, repr=False)
    _endpoint: str = field(default="", repr=False)

    @staticmethod
    def _extract_state_step(result: dict[str, Any]) -> tuple[str, str | None]:
        """Normalize job state from either new (`state`) or legacy (`status`) fields."""
        raw_state = result.get("state")
        if isinstance(raw_state, str) and raw_state:
            raw_step = result.get("step")
            step = raw_step if isinstance(raw_step, str) and raw_step else None
            return raw_state, step

        raw_status = result.get("status")
        status = raw_status if isinstance(raw_status, str) else ""
        if status.startswith("processing:"):
            _, _, step = status.partition(":")
            return "processing", step or None
        if status in {"pending", "processing", "complete", "failed"}:
            return status, None
        if status:
            return status, None
        return "processing", None

    def _update_job_state(self, result: dict[str, Any]) -> str:
        """Persist current status/state on the job object and return normalized state."""
        raw_status = result.get("status")
        if isinstance(raw_status, str) and raw_status:
            self.status = raw_status

        state, step = self._extract_state_step(result)
        self.state = state
        self.step = step
        return state

    def poll(self) -> dict:
        """Check current job status (sync)."""
        if self._client is None:
            raise RuntimeError("Job not associated with a client")
        result = self._client._get(f"{self._endpoint}/{self.job_id}")
        self._update_job_state(result)
        return result

    def wait(self, poll_interval: float = 3.0, timeout: float = 600.0) -> dict:
        """Block until job completes or times out (sync)."""
        import time

        start = time.time()
        while time.time() - start < timeout:
            result = self.poll()
            state = self._update_job_state(result)
            if state == "complete":
                return result
            if state == "failed":
                raise RuntimeError(f"Job failed: {result.get('error', 'Unknown error')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {self.job_id} did not complete within {timeout}s")

    async def async_poll(self) -> dict:
        """Check current job status (async)."""
        if self._client is None:
            raise RuntimeError("Job not associated with a client")
        result = await self._client._get(f"{self._endpoint}/{self.job_id}")
        self._update_job_state(result)
        return result

    async def async_wait(
        self, poll_interval: float = 3.0, timeout: float = 600.0
    ) -> dict:
        """Await until job completes or times out (async)."""
        import asyncio
        import time

        start = time.time()
        while time.time() - start < timeout:
            result = await self.async_poll()
            state = self._update_job_state(result)
            if state == "complete":
                return result
            if state == "failed":
                raise RuntimeError(f"Job failed: {result.get('error', 'Unknown error')}")
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Job {self.job_id} did not complete within {timeout}s")


@dataclass
class ReconstructJob(AsyncJob):
    """Async 3D reconstruction job."""

    estimated_time_s: float | None = None

    def wait(self, poll_interval: float = 5.0, timeout: float = 600.0) -> dict:
        return super().wait(poll_interval=poll_interval, timeout=timeout)

    def download(self, path: str | Path) -> Path:
        """Wait for completion and download the result."""
        result = self.wait()
        scene_url = result.get("scene_url")
        if not scene_url:
            raise RuntimeError("No scene URL in result")
        import httpx

        resp = httpx.get(scene_url)
        resp.raise_for_status()
        out = Path(path)
        out.write_bytes(resp.content)
        return out


@dataclass
class FloorplanJob(AsyncJob):
    """Async floorplan generation job."""

    estimated_time_s: float | None = None


@dataclass
class Segment3DJob(AsyncJob):
    """Async 3D segmentation job."""

    estimated_time_s: float | None = None
