"""Typed exceptions for the SpatialForge SDK."""

from __future__ import annotations


class SpatialForgeError(Exception):
    """Base exception for all SpatialForge API errors."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AuthenticationError(SpatialForgeError):
    """Raised when the API key is missing, invalid, or disabled (401)."""


class ForbiddenError(SpatialForgeError):
    """Raised when the plan does not allow access to this endpoint (403)."""


class ValidationError(SpatialForgeError):
    """Raised for invalid input: bad format, missing params, out-of-bounds (400)."""


class PayloadTooLargeError(SpatialForgeError):
    """Raised when the uploaded file exceeds the size limit (413)."""


class RateLimitError(SpatialForgeError):
    """Raised when the rate limit or monthly quota is exceeded (429).

    Attributes:
        retry_after: Seconds to wait before retrying (from Retry-After header).
    """

    def __init__(self, status_code: int, detail: str, retry_after: float | None = None) -> None:
        super().__init__(status_code, detail)
        self.retry_after = retry_after


class ServerError(SpatialForgeError):
    """Raised for 5xx server-side errors (500, 503)."""


class TimeoutError(SpatialForgeError):
    """Raised when the server-side inference times out (504)."""


_STATUS_MAP: dict[int, type[SpatialForgeError]] = {
    400: ValidationError,
    401: AuthenticationError,
    403: ForbiddenError,
    413: PayloadTooLargeError,
    429: RateLimitError,
    504: TimeoutError,
}

# Status codes that are safe to retry (transient errors)
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def raise_for_status(status_code: int, detail: str, retry_after: float | None = None) -> None:
    """Raise the appropriate typed exception for an HTTP error status code."""
    exc_class = _STATUS_MAP.get(status_code)

    if exc_class is RateLimitError:
        raise RateLimitError(status_code, detail, retry_after=retry_after)

    if exc_class is not None:
        raise exc_class(status_code, detail)

    if status_code >= 500:
        raise ServerError(status_code, detail)

    raise SpatialForgeError(status_code, detail)
