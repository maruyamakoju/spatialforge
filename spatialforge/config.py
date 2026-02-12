"""SpatialForge configuration via environment variables."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# ── Upload size limits ───────────────────────────────────
MAX_IMAGE_FILE_SIZE = 20 * 1024 * 1024  # 20 MB — depth, measure
MAX_VIDEO_FILE_SIZE = 100 * 1024 * 1024  # 100 MB — pose
MAX_ASYNC_VIDEO_FILE_SIZE = 500 * 1024 * 1024  # 500 MB — reconstruct, floorplan, segment

_DEFAULT_SECRET = "change-me-to-a-random-secret-key-at-least-32-chars"
_DEFAULT_ADMIN_KEY = "sf_admin_change_me"
_DEFAULT_SECURITY_CONTACT = "mailto:security@spatialforge.example.com"
_DEFAULT_SECURITY_CANONICAL = "https://spatialforge-demo.fly.dev/.well-known/security.txt"


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    debug: bool = False
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://maruyamakoju.github.io",
        "https://spatialforge-demo.fly.dev",
    ]
    security_contact: str = _DEFAULT_SECURITY_CONTACT
    security_expires: str = "2027-02-09T00:00:00Z"
    security_preferred_languages: str = "en,ja"
    security_canonical_url: str = _DEFAULT_SECURITY_CANONICAL
    security_encryption_url: str = ""

    # Auth
    api_key_secret: str = _DEFAULT_SECRET
    admin_api_key: str = _DEFAULT_ADMIN_KEY
    demo_mode: bool = False  # Allow unauthenticated access when Redis is down

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
    depth_backend: str = "hf"  # hf | da3
    device: str = "cuda"
    torch_dtype: str = "float16"
    research_mode: bool = False  # DANGER: enables CC-BY-NC models. Never in production.

    # Stripe billing (optional — app works without it)
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Sentry error tracking (optional)
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    sentry_traces_sample_rate: float = 0.1  # 10% of transactions

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

    # SECURITY: Enforce non-default secrets in production
    if not s.demo_mode:
        if s.api_key_secret == _DEFAULT_SECRET:
            raise RuntimeError(
                "CRITICAL SECURITY ERROR: API_KEY_SECRET is using the default value. "
                "This allows attackers to forge valid API keys. "
                "Set a strong random secret via API_KEY_SECRET environment variable. "
                "Example: export API_KEY_SECRET=$(openssl rand -base64 48)"
            )
        if s.admin_api_key == _DEFAULT_ADMIN_KEY:
            raise RuntimeError(
                "CRITICAL SECURITY ERROR: ADMIN_API_KEY is using the default value. "
                "Set a unique admin key via ADMIN_API_KEY environment variable. "
                "Example: export ADMIN_API_KEY=sf_$(openssl rand -hex 32)"
            )
        if s.minio_access_key == "minioadmin" and s.minio_secret_key == "minioadmin":
            logger.warning(
                "SECURITY WARNING: MinIO using default credentials (minioadmin/minioadmin). "
                "This is insecure for production. Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY."
            )
    else:
        # Demo mode warnings (not errors)
        if s.api_key_secret == _DEFAULT_SECRET:
            logger.warning("Demo mode: API_KEY_SECRET using default value")
        if s.admin_api_key == _DEFAULT_ADMIN_KEY:
            logger.warning("Demo mode: ADMIN_API_KEY using default value")

    if len(s.api_key_secret) < 32:
        raise ValueError(
            "API_KEY_SECRET must be at least 32 characters. "
            f"Current length: {len(s.api_key_secret)}"
        )

    s.depth_backend = s.depth_backend.lower()
    if s.depth_backend not in {"hf", "da3"}:
        raise ValueError(
            "DEPTH_BACKEND must be one of ['hf', 'da3']. "
            f"Current value: {s.depth_backend}"
        )

    if "*" in s.allowed_origins:
        if s.demo_mode:
            logger.warning("Demo mode: ALLOWED_ORIGINS contains '*'")
        else:
            raise RuntimeError(
                "CORS SECURITY ERROR: ALLOWED_ORIGINS contains wildcard '*'. "
                "Use explicit origins in production or set DEMO_MODE=true only for demos."
            )

    if not s.demo_mode:
        if s.security_contact == _DEFAULT_SECURITY_CONTACT:
            logger.warning(
                "SECURITY WARNING: SECURITY_CONTACT is using a placeholder address. "
                "Set SECURITY_CONTACT to a monitored mailbox."
            )
        if s.security_canonical_url == _DEFAULT_SECURITY_CANONICAL:
            logger.warning(
                "SECURITY WARNING: SECURITY_CANONICAL_URL points to demo host. "
                "Set SECURITY_CANONICAL_URL to your production API domain."
            )

    return s
