"""API key authentication and management."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

if TYPE_CHECKING:
    from redis.asyncio import Redis

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class Plan(str, Enum):
    FREE = "free"
    BUILDER = "builder"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"


@dataclass
class APIKeyRecord:
    key_hash: str
    plan: Plan
    owner: str
    created_at: float = field(default_factory=time.time)
    monthly_calls: int = 0
    monthly_limit: int = 100
    enabled: bool = True


def generate_api_key(prefix: str = "sf") -> str:
    """Generate a new API key with prefix."""
    random_part = secrets.token_urlsafe(32)
    return f"{prefix}_{random_part}"


def hash_api_key(key: str, secret: str) -> str:
    """Hash an API key using HMAC-SHA256."""
    return hmac.new(secret.encode(), key.encode(), hashlib.sha256).hexdigest()


class APIKeyManager:
    """Manages API key storage and validation via Redis."""

    KEY_PREFIX = "apikey:"

    def __init__(self, redis: Redis, secret: str) -> None:
        self._redis = redis
        self._secret = secret

    async def create_key(self, owner: str, plan: Plan) -> str:
        """Create a new API key and store its hash in Redis."""
        from ..config import get_settings

        settings = get_settings()
        limits = {
            Plan.FREE: settings.rate_limit_free,
            Plan.BUILDER: settings.rate_limit_builder,
            Plan.PRO: settings.rate_limit_pro,
            Plan.ENTERPRISE: 999_999_999,
            Plan.ADMIN: 999_999_999,
        }

        raw_key = generate_api_key()
        key_hash = hash_api_key(raw_key, self._secret)
        record = {
            "key_hash": key_hash,
            "plan": plan.value,
            "owner": owner,
            "created_at": str(time.time()),
            "monthly_calls": "0",
            "monthly_limit": str(limits.get(plan, 100)),
            "enabled": "1",
        }
        await self._redis.hset(f"{self.KEY_PREFIX}{key_hash}", mapping=record)
        return raw_key

    async def validate_key(self, raw_key: str) -> APIKeyRecord | None:
        """Validate an API key and return its record."""
        key_hash = hash_api_key(raw_key, self._secret)
        data = await self._redis.hgetall(f"{self.KEY_PREFIX}{key_hash}")
        if not data:
            return None

        record = APIKeyRecord(
            key_hash=data.get("key_hash", ""),
            plan=Plan(data.get("plan", "free")),
            owner=data.get("owner", ""),
            created_at=float(data.get("created_at", 0)),
            monthly_calls=int(data.get("monthly_calls", 0)),
            monthly_limit=int(data.get("monthly_limit", 100)),
            enabled=data.get("enabled", "1") == "1",
        )
        return record if record.enabled else None

    async def increment_usage(self, raw_key: str) -> int:
        """Increment monthly usage counter. Returns new count."""
        key_hash = hash_api_key(raw_key, self._secret)
        return await self._redis.hincrby(f"{self.KEY_PREFIX}{key_hash}", "monthly_calls", 1)


async def get_current_user(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> APIKeyRecord:
    """FastAPI dependency to validate API key from request header.

    Uses request.app.state to access the key manager — no circular imports.
    If Redis is unavailable (key_manager is None), auth is unavailable and
    requests are rejected with 503 unless DEMO_MODE is explicitly enabled.
    """
    import logging

    manager = getattr(request.app.state, "key_manager", None)

    if manager is None:
        # Check if the app is intentionally running in demo mode
        # (e.g., for the interactive demo page or local development)
        from ..config import get_settings

        settings = get_settings()
        if getattr(settings, "demo_mode", False):
            return APIKeyRecord(
                key_hash="demo",
                plan=Plan.FREE,
                owner="demo",
                monthly_calls=0,
                monthly_limit=999_999_999,
            )

        logging.getLogger(__name__).warning(
            "Auth request rejected: Redis unavailable and DEMO_MODE not enabled"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable. Try again later.",
        )

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include X-API-Key header.",
        )

    record = await manager.validate_key(api_key)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or disabled API key.",
        )

    if record.monthly_calls >= record.monthly_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Monthly limit reached ({record.monthly_limit} calls). Upgrade your plan.",
        )

    await manager.increment_usage(api_key)
    return record
