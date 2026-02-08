"""Token bucket rate limiter backed by Redis."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from starlette.responses import Response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Per-IP sliding window rate limiter for unauthenticated endpoints.

    Authenticated endpoints are limited via the API key monthly quota instead.
    This middleware protects public routes (health, demo, docs).
    """

    WINDOW_S = 60  # 1-minute window
    MAX_REQUESTS = 60  # 60 requests per minute for unauthenticated

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        # Skip rate limiting for authenticated endpoints (handled by API key quota)
        if request.headers.get("X-API-Key"):
            return await call_next(request)

        # Skip health check
        if request.url.path in ("/health", "/"):
            return await call_next(request)

        redis = request.app.state.redis
        if redis is None:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}"
        now = time.time()

        pipe = redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - self.WINDOW_S)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, self.WINDOW_S)
        results = await pipe.execute()

        request_count = results[2]
        if request_count > self.MAX_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please slow down.",
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(max(0, self.MAX_REQUESTS - request_count))
        return response
