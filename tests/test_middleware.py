"""Tests for middleware (security headers, request timeout)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from spatialforge.middleware.security_headers import SecurityHeadersMiddleware
from spatialforge.middleware.timeout import RequestTimeoutMiddleware


def _make_app_with_middleware(*middlewares):
    """Create a minimal FastAPI app with the given middleware stack."""

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield

    app = FastAPI(lifespan=noop_lifespan)

    for mw_cls, kwargs in middlewares:
        app.add_middleware(mw_cls, **kwargs)

    return app


class TestSecurityHeaders:
    """Tests for SecurityHeadersMiddleware."""

    def test_adds_security_headers(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")

        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "1; mode=block"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "camera=()" in resp.headers["Permissions-Policy"]

    def test_no_hsts_for_localhost(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test", headers={"host": "localhost:8000"})

        assert "Strict-Transport-Security" not in resp.headers

    def test_hsts_for_production_host(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test", headers={"host": "spatialforge-demo.fly.dev"})

        assert "Strict-Transport-Security" in resp.headers
        assert "max-age=31536000" in resp.headers["Strict-Transport-Security"]

    def test_csp_restrictive_for_api(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/v1/test")
        async def test_api():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/v1/test")

        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'none'" in csp

    def test_csp_permissive_for_docs(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/docs")
        async def test_docs():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/docs")

        csp = resp.headers.get("Content-Security-Policy", "")
        assert "unsafe-inline" in csp

    def test_api_version_header(self):
        app = _make_app_with_middleware((SecurityHeadersMiddleware, {}))

        @app.get("/test")
        async def test_endpoint():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/test")

        assert "X-API-Version" in resp.headers
        assert resp.headers["X-API-Version"] == "0.1.0"


class TestRequestTimeout:
    """Tests for RequestTimeoutMiddleware."""

    def test_non_inference_not_affected(self):
        app = _make_app_with_middleware((RequestTimeoutMiddleware, {"timeout_s": 1}))

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_inference_within_timeout_succeeds(self):
        app = _make_app_with_middleware((RequestTimeoutMiddleware, {"timeout_s": 10}))

        @app.post("/v1/depth")
        async def fast_depth():
            return {"depth_map_url": "test"}

        client = TestClient(app)
        resp = client.post("/v1/depth")
        assert resp.status_code == 200

    def test_timeout_returns_504(self):
        app = _make_app_with_middleware((RequestTimeoutMiddleware, {"timeout_s": 0.1}))

        @app.post("/v1/depth")
        async def slow_depth():
            await asyncio.sleep(5)
            return {"depth_map_url": "test"}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/depth")
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]

    def test_measure_path_has_timeout(self):
        app = _make_app_with_middleware((RequestTimeoutMiddleware, {"timeout_s": 0.1}))

        @app.post("/v1/measure")
        async def slow_measure():
            await asyncio.sleep(5)
            return {"distance_m": 1.0}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/measure")
        assert resp.status_code == 504

    def test_pose_path_has_timeout(self):
        app = _make_app_with_middleware((RequestTimeoutMiddleware, {"timeout_s": 0.1}))

        @app.post("/v1/pose")
        async def slow_pose():
            await asyncio.sleep(5)
            return {"camera_poses": []}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/pose")
        assert resp.status_code == 504
