"""MinIO / S3-compatible object storage layer."""

from __future__ import annotations

import io
import uuid
from datetime import timedelta
from typing import TYPE_CHECKING

from minio import Minio

if TYPE_CHECKING:
    pass


class ObjectStore:
    """Wrapper around MinIO for storing depth maps, 3D assets, and results."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ) -> None:
        self._client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self._bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

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
        return object_key

    def upload_file(self, file_path: str, content_type: str, prefix: str = "uploads") -> str:
        """Upload a local file and return the object key."""
        object_key = f"{prefix}/{uuid.uuid4().hex}_{file_path.split('/')[-1]}"
        self._client.fput_object(self._bucket, object_key, file_path, content_type=content_type)
        return object_key

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
        """Delete an object."""
        self._client.remove_object(self._bucket, object_key)

    def list_objects(self, prefix: str = "") -> list[str]:
        """List object keys under a prefix."""
        objects = self._client.list_objects(self._bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects if obj.object_name]
