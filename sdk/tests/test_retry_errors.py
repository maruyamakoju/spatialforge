"""Tests for retry logic and typed error classes."""

from __future__ import annotations

import httpx
import pytest
from spatialforge_client.exceptions import (
    AuthenticationError,
    ForbiddenError,
    PayloadTooLargeError,
    RateLimitError,
    ServerError,
    SpatialForgeError,
    TimeoutError,
    ValidationError,
    raise_for_status,
)

# ── Exception hierarchy tests ────────────────────────────


class TestExceptionHierarchy:
    def test_all_errors_inherit_from_base(self):
        """All typed errors are subclasses of SpatialForgeError."""
        for cls in [AuthenticationError, ForbiddenError, ValidationError,
                    PayloadTooLargeError, RateLimitError, ServerError, TimeoutError]:
            err = cls(400, "test")
            assert isinstance(err, SpatialForgeError)

    def test_error_has_status_and_detail(self):
        err = SpatialForgeError(401, "Invalid key")
        assert err.status_code == 401
        assert err.detail == "Invalid key"
        assert "401" in str(err)
        assert "Invalid key" in str(err)

    def test_rate_limit_error_has_retry_after(self):
        err = RateLimitError(429, "Too fast", retry_after=30.0)
        assert err.retry_after == 30.0
        assert err.status_code == 429

    def test_rate_limit_error_retry_after_none(self):
        err = RateLimitError(429, "Too fast")
        assert err.retry_after is None


# ── raise_for_status mapping tests ───────────────────────


class TestRaiseForStatus:
    def test_400_raises_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            raise_for_status(400, "Bad input")
        assert exc_info.value.status_code == 400

    def test_401_raises_authentication_error(self):
        with pytest.raises(AuthenticationError):
            raise_for_status(401, "Invalid key")

    def test_403_raises_forbidden_error(self):
        with pytest.raises(ForbiddenError):
            raise_for_status(403, "Not allowed")

    def test_413_raises_payload_too_large(self):
        with pytest.raises(PayloadTooLargeError):
            raise_for_status(413, "Too big")

    def test_429_raises_rate_limit_error(self):
        with pytest.raises(RateLimitError) as exc_info:
            raise_for_status(429, "Rate limited", retry_after=60.0)
        assert exc_info.value.retry_after == 60.0

    def test_500_raises_server_error(self):
        with pytest.raises(ServerError):
            raise_for_status(500, "Internal error")

    def test_502_raises_server_error(self):
        with pytest.raises(ServerError):
            raise_for_status(502, "Bad gateway")

    def test_503_raises_server_error(self):
        with pytest.raises(ServerError):
            raise_for_status(503, "Service unavailable")

    def test_504_raises_timeout_error(self):
        with pytest.raises(TimeoutError):
            raise_for_status(504, "Gateway timeout")

    def test_unknown_4xx_raises_base_error(self):
        with pytest.raises(SpatialForgeError) as exc_info:
            raise_for_status(418, "I'm a teapot")
        assert exc_info.value.status_code == 418


# ── Client typed error integration ───────────────────────


class TestClientTypedErrors:
    def test_401_raises_authentication_error(self, invalid_key_client, tiny_image):
        with pytest.raises(AuthenticationError) as exc_info:
            invalid_key_client.depth(tiny_image)
        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail

    def test_403_raises_forbidden_error(self, forbidden_client, tiny_image):
        with pytest.raises(ForbiddenError):
            forbidden_client.depth(tiny_image)

    def test_413_raises_payload_too_large(self, toolarge_client, tiny_image):
        with pytest.raises(PayloadTooLargeError):
            toolarge_client.depth(tiny_image)

    def test_400_raises_validation_error(self, badrequest_client, tiny_image):
        with pytest.raises(ValidationError):
            badrequest_client.depth(tiny_image)

    def test_500_raises_server_error(self, servererror_client, tiny_image):
        with pytest.raises(ServerError):
            servererror_client.depth(tiny_image)

    def test_504_raises_timeout_error(self, timeout_client, tiny_image):
        with pytest.raises(TimeoutError):
            timeout_client.depth(tiny_image)

    def test_catch_base_class(self, invalid_key_client, tiny_image):
        """Typed errors can be caught as SpatialForgeError."""
        with pytest.raises(SpatialForgeError):
            invalid_key_client.depth(tiny_image)


# ── Retry logic tests ───────────────────────────────────


class TestRetryLogic:
    def test_retry_on_429_succeeds(self, tiny_image):
        """Client retries on 429 and succeeds on second attempt."""
        from spatialforge_client import Client

        from .conftest import DEPTH_RESPONSE, ERROR_429

        call_count = 0

        class RetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return httpx.Response(429, json=ERROR_429, headers={"Retry-After": "0"})
                return httpx.Response(200, json=DEPTH_RESPONSE)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=2, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=RetryTransport(),
        )

        result = client.depth(tiny_image)
        assert result.min_depth_m == 0.5
        assert call_count == 2
        client.close()

    def test_retry_on_500_succeeds(self, tiny_image):
        """Client retries on 500 and succeeds on second attempt."""
        from spatialforge_client import Client

        from .conftest import DEPTH_RESPONSE, ERROR_500

        call_count = 0

        class RetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return httpx.Response(500, json=ERROR_500)
                return httpx.Response(200, json=DEPTH_RESPONSE)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=3, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=RetryTransport(),
        )

        client.depth(tiny_image)
        assert call_count == 3
        client.close()

    def test_no_retry_on_401(self, tiny_image):
        """Client does not retry on 401 (non-retryable)."""
        from spatialforge_client import Client

        from .conftest import ERROR_401

        call_count = 0

        class NoRetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                return httpx.Response(401, json=ERROR_401)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=3, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=NoRetryTransport(),
        )

        with pytest.raises(AuthenticationError):
            client.depth(tiny_image)
        assert call_count == 1  # No retries
        client.close()

    def test_no_retry_on_400(self, tiny_image):
        """Client does not retry on 400 (non-retryable)."""
        from spatialforge_client import Client

        from .conftest import ERROR_400

        call_count = 0

        class NoRetryTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                return httpx.Response(400, json=ERROR_400)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=3, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=NoRetryTransport(),
        )

        with pytest.raises(ValidationError):
            client.depth(tiny_image)
        assert call_count == 1
        client.close()

    def test_max_retries_exhausted(self, tiny_image):
        """Client raises after exhausting retries."""
        from spatialforge_client import Client

        from .conftest import ERROR_500

        call_count = 0

        class AlwaysFailTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                return httpx.Response(500, json=ERROR_500)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=2, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=AlwaysFailTransport(),
        )

        with pytest.raises(ServerError):
            client.depth(tiny_image)
        assert call_count == 3  # 1 initial + 2 retries
        client.close()

    def test_retry_respects_retry_after_header(self, tiny_image):
        """Client uses Retry-After header value for delay."""
        from spatialforge_client import Client

        from .conftest import DEPTH_RESPONSE, ERROR_429

        call_count = 0

        class RetryAfterTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return httpx.Response(429, json=ERROR_429, headers={"Retry-After": "0.01"})
                return httpx.Response(200, json=DEPTH_RESPONSE)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=1, retry_base_delay=100.0)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=RetryAfterTransport(),
        )

        # Should succeed quickly because Retry-After is 0.01, not 100.0
        result = client.depth(tiny_image)
        assert result.min_depth_m == 0.5
        assert call_count == 2
        client.close()

    def test_zero_retries_no_retry(self, tiny_image):
        """With max_retries=0, no retry is attempted."""
        from spatialforge_client import Client

        from .conftest import ERROR_500

        call_count = 0

        class FailTransport(httpx.BaseTransport):
            def handle_request(self, request):
                nonlocal call_count
                call_count += 1
                return httpx.Response(500, json=ERROR_500)

        client = Client(api_key="sf_test", base_url="https://mock.api", max_retries=0, retry_base_delay=0.01)
        client._client = httpx.Client(
            base_url="https://mock.api",
            headers={"X-API-Key": "sf_test"},
            transport=FailTransport(),
        )

        with pytest.raises(ServerError):
            client.depth(tiny_image)
        assert call_count == 1
        client.close()
