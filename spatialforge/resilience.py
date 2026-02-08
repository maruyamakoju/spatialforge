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


def with_redis_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add exponential backoff retry for Redis operations."""

    @retry(
        retry=retry_if_exception_type(REDIS_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Redis operation failed: {func.__name__}, error: {e}")
            raise

    return wrapper


def with_minio_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add exponential backoff retry for MinIO/S3 operations."""

    @retry(
        retry=retry_if_exception_type(MINIO_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"MinIO operation failed: {func.__name__}, error: {e}")
            raise

    return wrapper


def with_stripe_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to add exponential backoff retry for Stripe API calls."""

    @retry(
        retry=retry_if_exception_type(STRIPE_RETRYABLE_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Stripe operation failed: {func.__name__}, error: {e}")
            raise

    return wrapper
