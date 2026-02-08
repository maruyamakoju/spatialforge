"""SpatialForge — Any Camera. Instant 3D. One API.

Main FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from . import __version__
from .auth.api_keys import APIKeyManager
from .auth.rate_limiter import RateLimiterMiddleware
from .config import get_settings
from .inference.model_manager import ModelManager
from .logging_config import RequestTracingMiddleware, setup_logging
from .metrics import MetricsMiddleware
from .middleware.security_headers import SecurityHeadersMiddleware
from .middleware.timeout import RequestTimeoutMiddleware
from .models.responses import HealthResponse
from .storage.object_store import ObjectStore

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: init and cleanup shared resources."""
    settings = get_settings()
    logger.info("Starting SpatialForge v%s", __version__)

    # Redis (optional — app starts without it, but auth is disabled)
    try:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        app.state.redis = redis

        # API key manager
        key_manager = APIKeyManager(redis, settings.api_key_secret)
        app.state.key_manager = key_manager

        # Ensure an admin key exists
        existing = await key_manager.validate_key(settings.admin_api_key)
        if existing is None:
            from .auth.api_keys import Plan, hash_api_key

            key_hash = hash_api_key(settings.admin_api_key, settings.api_key_secret)
            await redis.hset(
                f"apikey:{key_hash}",
                mapping={
                    "key_hash": key_hash,
                    "plan": Plan.ADMIN.value,
                    "owner": "admin",
                    "created_at": "0",
                    "monthly_calls": "0",
                    "monthly_limit": "999999999",
                    "enabled": "1",
                },
            )
            logger.info("Admin API key registered")
    except Exception:
        logger.warning("Redis not available — auth and rate limiting disabled")
        app.state.redis = None
        app.state.key_manager = None

    # Model manager
    device = settings.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available — running on CPU (slow)")

    model_manager = ModelManager(
        model_dir=settings.model_dir,
        device=device,
        dtype=settings.torch_dtype if device != "cpu" else "float32",
        research_mode=settings.research_mode,
    )
    app.state.model_manager = model_manager

    # Object store
    try:
        obj_store = ObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
        app.state.object_store = obj_store
        logger.info("Object store connected: %s/%s", settings.minio_endpoint, settings.minio_bucket)
    except Exception:
        logger.warning("Object store not available — file storage disabled")
        app.state.object_store = None

    # Stripe billing (optional — app works without it)
    if settings.stripe_secret_key:
        try:
            from .billing.stripe_billing import StripeBilling

            billing = StripeBilling(
                secret_key=settings.stripe_secret_key,
                webhook_secret=settings.stripe_webhook_secret,
                redis=getattr(app.state, "redis", None),
            )
            await billing.ensure_products()
            app.state.stripe_billing = billing
            logger.info("Stripe billing initialized")
        except Exception:
            logger.warning("Stripe billing initialization failed", exc_info=True)
            app.state.stripe_billing = None
    else:
        app.state.stripe_billing = None

    logger.info("SpatialForge ready (device=%s)", device)

    yield

    # Cleanup
    model_manager.unload_all()
    if app.state.redis is not None:
        await app.state.redis.close()
    logger.info("SpatialForge shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="SpatialForge API",
        summary="Any Camera. Instant 3D. One API.",
        description=(
            "SpatialForge is a spatial intelligence API that transforms ordinary camera input "
            "into precise 3D understanding. Powered by state-of-the-art depth estimation models, "
            "it provides monocular depth estimation, real-world measurement, camera pose recovery, "
            "3D reconstruction, floor plan generation, and 3D semantic segmentation.\n\n"
            "## Authentication\n"
            "All endpoints (except `/health` and `/v1/billing/plans`) require an API key "
            "passed via the `X-API-Key` header.\n\n"
            "## Rate Limits\n"
            "| Plan | Monthly Calls | Price |\n"
            "|------|--------------|-------|\n"
            "| Free | 100 | $0 |\n"
            "| Builder | 5,000 | $29/mo |\n"
            "| Pro | 50,000 | $99/mo |\n"
            "| Enterprise | Unlimited | $499/mo |\n"
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "SpatialForge", "url": "https://github.com/maruyamakoju/spatialforge"},
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type", "Authorization"],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining"],
    )

    # Security headers (OWASP best practices)
    app.add_middleware(SecurityHeadersMiddleware)

    # Request timeout for inference endpoints
    app.add_middleware(RequestTimeoutMiddleware, timeout_s=120)

    # Rate limiter
    app.add_middleware(RateLimiterMiddleware)

    # Prometheus metrics middleware
    app.add_middleware(MetricsMiddleware)

    # Request tracing (adds X-Request-ID to all requests)
    app.add_middleware(RequestTracingMiddleware)

    # Prometheus metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Static files (demo page)
    from pathlib import Path

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Health endpoint (no auth required)
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        gpu_available = torch.cuda.is_available()
        models = app.state.model_manager.loaded_models if hasattr(app.state, "model_manager") else []
        return HealthResponse(
            status="ok",
            version=__version__,
            gpu_available=gpu_available,
            models_loaded=models,
        )

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": "SpatialForge",
            "tagline": "Any Camera. Instant 3D. One API.",
            "version": __version__,
            "docs": "/docs",
            "demo": "/static/demo/index.html",
        }

    # Register API v1 routes
    from .api.v1 import depth, floorplan, measure, pose, reconstruct, segment

    app.include_router(depth.router, prefix="/v1", tags=["depth"])
    app.include_router(pose.router, prefix="/v1", tags=["pose"])
    app.include_router(reconstruct.router, prefix="/v1", tags=["reconstruct"])
    app.include_router(measure.router, prefix="/v1", tags=["measure"])
    app.include_router(floorplan.router, prefix="/v1", tags=["floorplan"])
    app.include_router(segment.router, prefix="/v1", tags=["segment-3d"])

    # Admin routes
    from .api.v1 import admin

    app.include_router(admin.router, prefix="/v1/admin", tags=["admin"])

    # Billing routes
    from .api.v1 import billing

    app.include_router(billing.router, prefix="/v1", tags=["billing"])

    return app


app = create_app()


def run() -> None:
    """CLI entry point."""
    settings = get_settings()
    setup_logging(debug=settings.debug)
    uvicorn.run(
        "spatialforge.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=settings.debug,
    )


if __name__ == "__main__":
    run()
