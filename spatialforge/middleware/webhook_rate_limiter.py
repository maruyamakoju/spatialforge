"""Simple IP-based rate limiter for webhook endpoints to prevent DoS."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

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

    async def _is_rate_limited_redis(self, redis_client: Any, client_ip: str) -> bool | None:
        """Return Redis-backed limit result; None if Redis check failed."""
        now = time.time()
        cutoff = now - self.window_seconds
        member = f"{now:.6f}:{time.monotonic_ns()}"
        key = f"webhook_rl:{client_ip}"

        try:
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zcard(key)
            pipe.zadd(key, {member: now})
            pipe.expire(key, self.window_seconds + 5)
            _, current_count, _, _ = await pipe.execute()
            return int(current_count) >= self.max_requests
        except Exception:
            logger.warning("Redis webhook rate limit check failed; falling back to in-memory limiter", exc_info=True)
            return None

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Only apply to webhook endpoints
        if not request.url.path.endswith("/webhooks"):
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Prefer Redis-backed limiting when available (shared across replicas).
        redis_client = getattr(request.app.state, "redis", None)
        if redis_client is not None:
            limited = await self._is_rate_limited_redis(redis_client, client_ip)
            if limited is True:
                logger.warning(
                    "Webhook rate limit exceeded for IP %s (%d requests in %ds; redis)",
                    client_ip,
                    self.max_requests,
                    self.window_seconds,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Too many webhook requests. Limit: {self.max_requests} per {self.window_seconds}s"
                    },
                    headers={"Retry-After": str(self.window_seconds)},
                )
            if limited is False:
                return await call_next(request)

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
