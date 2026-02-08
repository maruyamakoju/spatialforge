"""SpatialForge configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


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
    api_key_secret: str = "change-me-to-a-random-secret-key-at-least-32-chars"
    admin_api_key: str = "sf_admin_change_me"

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
    default_depth_model: str = "giant"  # giant | large | base | small
    device: str = "cuda"
    torch_dtype: str = "float16"

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
    return Settings()
