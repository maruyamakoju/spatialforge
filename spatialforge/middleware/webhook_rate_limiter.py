"""Simple IP-based rate limiter for webhook endpoints to prevent DoS."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class WebhookRateLimiterMiddleware(BaseHTTPMiddleware):
    """Rate limiter specifically for webhook endpoints.

    Prevents DoS attacks on unauthenticated webhook endpoints by limiting
    requests per IP address. Uses in-memory storage (not distributed).
    """

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Only apply to webhook endpoints
        if not request.url.path.endswith("/webhooks"):
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Clean old requests
        now = time.time()
        cutoff = now - self.window_seconds
        self._requests[client_ip] = [t for t in self._requests[client_ip] if t > cutoff]

        # Check rate limit
        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning(
                "Webhook rate limit exceeded for IP %s (%d requests in %ds)",
                client_ip,
                len(self._requests[client_ip]),
                self.window_seconds,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Too many webhook requests. Limit: {self.max_requests} per {self.window_seconds}s"
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Record request
        self._requests[client_ip].append(now)

        return await call_next(request)
