"""Request timeout middleware for inference endpoints."""

from __future__ import annotations

import asyncio
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that run inference and may take a long time
_INFERENCE_PATHS = ("/v1/depth", "/v1/measure", "/v1/pose")

# Default timeout in seconds (generous for CPU inference)
DEFAULT_TIMEOUT_S = 120


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce a timeout on inference requests to prevent hangs.

    Non-inference endpoints (health, billing, admin) are not affected.
    Async job submission endpoints (/reconstruct, /floorplan, /segment-3d)
    return immediately so they don't need a timeout.
    """

    def __init__(self, app, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        super().__init__(app)
        self.timeout_s = timeout_s

    async def dispatch(self, request: Request, call_next) -> Response:
        # Only apply timeout to inference endpoints
        path = request.url.path
        if not any(path.startswith(p) for p in _INFERENCE_PATHS):
            return await call_next(request)

        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_s,
            )
        except TimeoutError:
            logger.warning(
                "Request timeout after %ds: %s %s",
                self.timeout_s,
                request.method,
                path,
            )
            return JSONResponse(
                status_code=504,
                content={
                    "detail": f"Request timed out after {self.timeout_s}s. "
                    "Try a smaller image or a faster model (small/base)."
                },
            )
