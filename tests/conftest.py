"""Shared test fixtures for SpatialForge API tests."""

from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(
        return_value={
            "key_hash": "test_hash",
            "plan": "admin",
            "owner": "test",
            "created_at": "0",
            "monthly_calls": "0",
            "monthly_limit": "999999999",
            "enabled": "1",
        }
    )
    redis.hincrby = AsyncMock(return_value=1)
    redis.hset = AsyncMock()
    redis.close = AsyncMock()

    # Pipeline mock for rate limiter
    pipe = AsyncMock()
    pipe.execute = AsyncMock(return_value=[None, None, 1, None])
    redis.pipeline = MagicMock(return_value=pipe)

    return redis


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager that returns fake depth results."""
    from spatialforge.inference.model_manager import ModelInfo

    mm = MagicMock()
    mm.loaded_models = ["depth_da3-metric-large"]
    mm.device = "cpu"
    mm.dtype = "float32"
    mm.research_mode = False

    # Mock depth pipeline — must use a real torch.Tensor so isinstance() check passes
    import torch

    fake_depth = np.random.rand(384, 384).astype(np.float32) * 10
    fake_tensor = torch.from_numpy(fake_depth).unsqueeze(0)  # 1 x 384 x 384

    pipe = MagicMock()
    pipe.return_value = {"predicted_depth": fake_tensor}

    model_info = ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Large",
        license="apache-2.0",
        task="metric_depth",
        description="test model",
    )

    mm.get_depth_model = MagicMock(return_value=(pipe, model_info))
    mm.gpu_status = MagicMock(return_value={"gpu_available": False})
    mm.unload_all = MagicMock()

    return mm


@pytest.fixture
def mock_object_store():
    """Mock ObjectStore."""
    store = MagicMock()
    store.upload_bytes = MagicMock(return_value="results/test.png")
    store.get_presigned_url = MagicMock(return_value="http://localhost:9000/spatialforge/results/test.png")
    store.upload_file = MagicMock(return_value="uploads/test.mp4")
    store.download_bytes = MagicMock(return_value=b"fake_video")
    return store


@pytest.fixture
def app(mock_redis, mock_model_manager, mock_object_store):
    """Create a test FastAPI app with mocked dependencies.

    Instead of relying on the lifespan (which needs real Redis/MinIO),
    we create a minimal app with the same routes and inject mocked state.
    """
    from spatialforge.auth.api_keys import APIKeyRecord, Plan, get_current_user

    # No-op lifespan for tests
    @asynccontextmanager
    async def test_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        yield

    test_app = FastAPI(lifespan=test_lifespan)

    # Inject mocked state
    test_app.state.redis = mock_redis
    test_app.state.model_manager = mock_model_manager
    test_app.state.object_store = mock_object_store
    test_app.state.key_manager = MagicMock()

    # Override auth dependency to return a fake admin user
    async def fake_auth():
        return APIKeyRecord(
            key_hash="test_hash",
            plan=Plan.ADMIN,
            owner="test",
            monthly_calls=0,
            monthly_limit=999999999,
        )

    test_app.dependency_overrides[get_current_user] = fake_auth

    # Register routes (same as main.py)
    from spatialforge.api.v1 import billing, depth, floorplan, measure, pose, reconstruct, segment
    from spatialforge.models.responses import HealthResponse

    test_app.include_router(depth.router, prefix="/v1", tags=["depth"])
    test_app.include_router(pose.router, prefix="/v1", tags=["pose"])
    test_app.include_router(reconstruct.router, prefix="/v1", tags=["reconstruct"])
    test_app.include_router(measure.router, prefix="/v1", tags=["measure"])
    test_app.include_router(floorplan.router, prefix="/v1", tags=["floorplan"])
    test_app.include_router(segment.router, prefix="/v1", tags=["segment-3d"])
    test_app.include_router(billing.router, prefix="/v1", tags=["billing"])

    @test_app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        return HealthResponse(
            status="ok",
            version="0.1.0",
            gpu_available=False,
            models_loaded=mock_model_manager.loaded_models,
        )

    @test_app.get("/", tags=["system"])
    async def root():
        return {
            "name": "SpatialForge",
            "tagline": "Any Camera. Instant 3D. One API.",
            "version": "0.1.0",
            "docs": "/docs",
        }

    yield test_app


@pytest.fixture
def client(app):
    """HTTP test client."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def api_key():
    """Test API key."""
    return "sf_test_key_12345"


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generate a small test image as JPEG bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
