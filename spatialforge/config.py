"""SpatialForge configuration via environment variables."""

from __future__ import annotations

import logging
import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_DEFAULT_SECRET = "change-me-to-a-random-secret-key-at-least-32-chars"
_DEFAULT_ADMIN_KEY = "sf_admin_change_me"


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Auth
    api_key_secret: str = _DEFAULT_SECRET
    admin_api_key: str = _DEFAULT_ADMIN_KEY

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "spatialforge"
    minio_secure: bool = False

    # Models
    model_dir: Path = Path("./models")
    default_depth_model: str = "large"  # large | base | small (Apache 2.0 only)
    device: str = "cuda"
    torch_dtype: str = "float16"
    research_mode: bool = False  # DANGER: enables CC-BY-NC models. Never in production.

    # Rate limiting (calls per month)
    rate_limit_free: int = 100
    rate_limit_builder: int = 5_000
    rate_limit_pro: int = 50_000

    # Processing
    max_image_size: int = 4096
    max_video_duration_s: int = 120
    result_ttl_hours: int = 24


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    if s.api_key_secret == _DEFAULT_SECRET:
        logger.warning(
            "API_KEY_SECRET is using the default value. "
            "Set a strong random secret via API_KEY_SECRET env var."
        )
    if s.admin_api_key == _DEFAULT_ADMIN_KEY:
        logger.warning(
            "ADMIN_API_KEY is using the default value. "
            "Set a unique admin key via ADMIN_API_KEY env var."
        )
    if len(s.api_key_secret) < 32:
        # Generate a secure secret at runtime if too short
        s.api_key_secret = secrets.token_urlsafe(48)
        logger.warning("API_KEY_SECRET too short — generated a random runtime secret.")
    return s
