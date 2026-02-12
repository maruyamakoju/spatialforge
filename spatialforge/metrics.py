"""Prometheus metrics for SpatialForge.

Exposes:
  - Request latency per endpoint
  - Inference duration per model
  - GPU memory usage
  - Active jobs gauge
  - Error counters
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram, Info
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

# ── Pre-compiled regexes for path normalization ──────────────

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
_HEX_RE = re.compile(r"^[0-9a-f]{16,}$", re.IGNORECASE)

# ── Metrics definitions ──────────────────────────────────────

APP_INFO = Info("spatialforge", "SpatialForge application info")
APP_INFO.info({"version": __version__})

REQUEST_LATENCY = Histogram(
    "spatialforge_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "endpoint", "status"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

REQUEST_COUNT = Counter(
    "spatialforge_requests_total",
    "Total number of HTTP requests",
    labelnames=["method", "endpoint", "status"],
)

INFERENCE_DURATION = Histogram(
    "spatialforge_inference_duration_seconds",
    "Model inference duration in seconds",
    labelnames=["model", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

INFERENCE_COUNT = Counter(
    "spatialforge_inferences_total",
    "Total number of inference calls",
    labelnames=["model", "endpoint"],
)

GPU_MEMORY_USED = Gauge(
    "spatialforge_gpu_memory_used_bytes",
    "GPU memory currently allocated",
)

GPU_MEMORY_TOTAL = Gauge(
    "spatialforge_gpu_memory_total_bytes",
    "Total GPU memory",
)

ACTIVE_JOBS = Gauge(
    "spatialforge_active_jobs",
    "Number of currently processing async jobs",
    labelnames=["job_type"],
)

ERROR_COUNT = Counter(
    "spatialforge_errors_total",
    "Total number of errors",
    labelnames=["endpoint", "error_type"],
)

API_KEY_USAGE = Counter(
    "spatialforge_api_key_calls_total",
    "Total API calls per plan",
    labelnames=["plan"],
)

FILE_UPLOAD_SIZE = Histogram(
    "spatialforge_upload_size_bytes",
    "Size of uploaded files in bytes",
    labelnames=["endpoint"],
    buckets=(1024, 10240, 102400, 524288, 1048576, 5242880, 10485760, 20971520),
)

RATE_LIMIT_HITS = Counter(
    "spatialforge_rate_limit_hits_total",
    "Total number of rate limit rejections",
    labelnames=["plan"],
)


# ── Middleware ────────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track request latency and count."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        # Skip metrics for the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Normalize endpoint path (remove UUIDs/job IDs)
        endpoint = self._normalize_path(request.url.path)
        method = request.method
        status = str(response.status_code)

        REQUEST_LATENCY.labels(method=method, endpoint=endpoint, status=status).observe(duration)
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()

        # Track errors (4xx client errors and 5xx server errors)
        status_code = response.status_code
        if status_code >= 400:
            error_type = "client" if status_code < 500 else "server"
            ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize paths to group similar endpoints.

        e.g., /v1/reconstruct/abc123-def456 -> /v1/reconstruct/{id}
        """
        parts = path.strip("/").split("/")
        normalized = []
        for part in parts:
            if _UUID_RE.match(part) or _HEX_RE.match(part):
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/" + "/".join(normalized)


# ── Helper functions ─────────────────────────────────────────

def record_inference(model: str, endpoint: str, duration_s: float) -> None:
    """Record an inference call duration."""
    INFERENCE_DURATION.labels(model=model, endpoint=endpoint).observe(duration_s)
    INFERENCE_COUNT.labels(model=model, endpoint=endpoint).inc()


def update_gpu_metrics() -> None:
    """Update GPU memory metrics (call periodically)."""
    try:
        import torch

        if torch.cuda.is_available():
            GPU_MEMORY_USED.set(torch.cuda.memory_allocated(0))
            GPU_MEMORY_TOTAL.set(torch.cuda.get_device_properties(0).total_mem)
    except Exception:
        pass
