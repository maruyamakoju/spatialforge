"""Shared test fixtures for SpatialForge API tests."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
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

    # Mock depth pipeline: returns dict with predicted_depth tensor
    fake_depth = np.random.rand(384, 384).astype(np.float32) * 10
    fake_tensor = MagicMock()
    fake_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = fake_depth

    pipe = MagicMock()
    pipe.return_value = {"predicted_depth": fake_tensor}

    model_info = ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Large",
        license="apache-2.0",
        task="metric_depth",
        description="test model",
    )

    # get_depth_model now returns (pipeline, ModelInfo) tuple
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
    """Create a test FastAPI app with mocked dependencies."""
    with (
        patch("spatialforge.main.aioredis") as mock_aioredis,
        patch("spatialforge.main.ObjectStore", return_value=mock_object_store),
        patch("spatialforge.main.ModelManager", return_value=mock_model_manager),
    ):
        mock_aioredis.from_url = MagicMock(return_value=mock_redis)

        from spatialforge.main import create_app

        test_app = create_app()
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
