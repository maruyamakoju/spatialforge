"""MinIO / S3-compatible object storage layer with Redis TTL tracking.

Every uploaded object gets a Redis key: object_ttl:{key} = created_timestamp.
Cleanup reads these keys and deletes objects past their TTL.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from datetime import timedelta
from typing import Protocol

from minio import Minio
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

TTL_PREFIX = "object_ttl:"


class SyncRedis(Protocol):
    """Subset of sync Redis API required by ObjectStore."""

    def set(self, key: str, value: str) -> object: ...

    def get(self, key: str | bytes) -> str | bytes | None: ...

    def delete(self, key: str) -> object: ...

    def scan(
        self,
        cursor: int = 0,
        match: str | None = None,
        count: int | None = None,
    ) -> tuple[int, list[str | bytes]]: ...


class ObjectStore:
    """MinIO wrapper with TTL tracking via Redis."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        redis: SyncRedis | None = None,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._redis = redis
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def set_redis(self, redis: SyncRedis) -> None:
        """Set Redis client for TTL tracking (can be set after init)."""
        self._redis = redis

    def upload_bytes(
        self,
        data: bytes,
        content_type: str,
        prefix: str = "results",
        extension: str = "bin",
    ) -> str:
        """Upload raw bytes and return the object key."""
        object_key = f"{prefix}/{uuid.uuid4().hex}.{extension}"
        self._client.put_object(
            self._bucket,
            object_key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        self._track_ttl(object_key)
        return object_key

    def upload_file(self, file_path: str, content_type: str, prefix: str = "uploads") -> str:
        """Upload a local file and return the object key."""
        filename = file_path.replace("\\", "/").split("/")[-1]
        object_key = f"{prefix}/{uuid.uuid4().hex}_{filename}"
        self._client.fput_object(self._bucket, object_key, file_path, content_type=content_type)
        self._track_ttl(object_key)
        return object_key

    async def async_upload_bytes(
        self,
        data: bytes,
        content_type: str,
        prefix: str = "results",
        extension: str = "bin",
    ) -> str:
        """Async wrapper for upload_bytes to avoid blocking the event loop."""
        return await run_in_threadpool(
            self.upload_bytes,
            data,
            content_type,
            prefix,
            extension,
        )

    async def async_upload_file(self, file_path: str, content_type: str, prefix: str = "uploads") -> str:
        """Async wrapper for upload_file to avoid blocking the event loop."""
        return await run_in_threadpool(self.upload_file, file_path, content_type, prefix)

    def get_presigned_url(self, object_key: str, expires: timedelta | None = None) -> str:
        """Generate a presigned download URL."""
        if expires is None:
            expires = timedelta(hours=24)
        return self._client.presigned_get_object(self._bucket, object_key, expires=expires)

    def download_bytes(self, object_key: str) -> bytes:
        """Download an object as bytes."""
        response = self._client.get_object(self._bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete(self, object_key: str) -> None:
        """Delete an object and its TTL tracking."""
        self._client.remove_object(self._bucket, object_key)
        if self._redis:
            try:
                self._redis.delete(f"{TTL_PREFIX}{object_key}")
            except Exception:
                logger.debug("Failed to delete TTL key for %s", object_key)

    def list_objects(self, prefix: str = "") -> list[str]:
        """List object keys under a prefix."""
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects if obj.object_name]

    # ── TTL tracking ─────────────────────────────────────────

    def _track_ttl(self, object_key: str) -> None:
        """Record upload timestamp in Redis for TTL cleanup."""
        if self._redis is None:
            return
        try:
            self._redis.set(f"{TTL_PREFIX}{object_key}", str(time.time()))
        except Exception:
            logger.debug("Failed to track TTL for %s", object_key)

    def get_expired_keys(self, ttl_hours: int = 24) -> list[str]:
        """Return object keys that have exceeded their TTL.

        Scans Redis for object_ttl:* keys and checks timestamps.
        """
        if self._redis is None:
            return []

        expired = []
        cutoff = time.time() - (ttl_hours * 3600)

        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match=f"{TTL_PREFIX}*", count=100)
                for key in keys:
                    ts_str = self._redis.get(key)
                    if ts_str:
                        try:
                            ts = float(ts_str)
                        except (ValueError, TypeError):
                            continue
                        if ts < cutoff:
                            # Extract object_key from Redis key
                            key_str = key if isinstance(key, str) else key.decode()
                            object_key = key_str.removeprefix(TTL_PREFIX)
                            expired.append(object_key)
                if cursor == 0:
                    break
        except Exception:
            logger.warning("Error scanning TTL keys", exc_info=True)

        return expired

    def cleanup_expired(self, ttl_hours: int = 24) -> int:
        """Delete objects that have exceeded their TTL. Returns count deleted."""
        expired = self.get_expired_keys(ttl_hours)
        deleted = 0
        for key in expired:
            try:
                self.delete(key)
                deleted += 1
            except Exception:
                logger.debug("Failed to delete expired object: %s", key)
        return deleted
