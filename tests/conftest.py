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
    redis.pipeline = MagicMock()
    pipe = AsyncMock()
    pipe.execute = AsyncMock(return_value=[None, None, 1, None])
    redis.pipeline.return_value = pipe
    return redis


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager that returns fake depth results."""
    mm = MagicMock()
    mm.loaded_models = ["depth_giant"]
    mm.device = "cpu"
    mm.dtype = "float32"

    # Mock depth pipeline
    fake_depth = np.random.rand(384, 384).astype(np.float32) * 10
    fake_result = {
        "predicted_depth": MagicMock(squeeze=MagicMock(return_value=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=fake_depth)))))),
        "depth": Image.fromarray((fake_depth / 10 * 255).astype(np.uint8)),
    }

    pipe = MagicMock()
    pipe.return_value = fake_result
    mm.get_depth_model = MagicMock(return_value=pipe)
    mm.gpu_status = MagicMock(return_value={"gpu_available": False})

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
    with patch("spatialforge.main.aioredis") as mock_aioredis, \
         patch("spatialforge.main.ObjectStore", return_value=mock_object_store), \
         patch("spatialforge.main.ModelManager", return_value=mock_model_manager):
        mock_aioredis.from_url = MagicMock(return_value=mock_redis)

        from spatialforge.main import create_app
        test_app = create_app()

        # Inject mocks into app state
        test_app.state.redis = mock_redis
        test_app.state.model_manager = mock_model_manager
        test_app.state.object_store = mock_object_store
        test_app.state.key_manager = MagicMock()

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
