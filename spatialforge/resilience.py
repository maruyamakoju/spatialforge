"""Resilience patterns: circuit breaker, retry with exponential backoff."""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Retryable exceptions for external services
REDIS_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)
MINIO_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)
STRIPE_RETRYABLE_EXCEPTIONS = (ConnectionError, TimeoutError)


def _make_retry_decorator(
    retryable_exceptions: tuple[type[Exception], ...],
    service_name: str,
    max_attempts: int = 3,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Factory for service-specific retry decorators with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @retry(
            retry=retry_if_exception_type(retryable_exceptions),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning("%s operation failed: %s, error: %s", service_name, func.__name__, e)
                raise

        return wrapper

    return decorator


with_redis_retry = _make_retry_decorator(REDIS_RETRYABLE_EXCEPTIONS, "Redis")
with_minio_retry = _make_retry_decorator(MINIO_RETRYABLE_EXCEPTIONS, "MinIO")
with_stripe_retry = _make_retry_decorator(STRIPE_RETRYABLE_EXCEPTIONS, "Stripe")
