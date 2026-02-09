"""Tests for async wrappers in ObjectStore."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spatialforge.storage.object_store import ObjectStore


@pytest.mark.asyncio
async def test_async_upload_file_wraps_sync_method():
    store = ObjectStore.__new__(ObjectStore)
    store.upload_file = MagicMock(return_value="uploads/key.mp4")

    key = await store.async_upload_file("video.mp4", content_type="video/mp4", prefix="uploads")

    assert key == "uploads/key.mp4"
    store.upload_file.assert_called_once_with("video.mp4", "video/mp4", "uploads")


@pytest.mark.asyncio
async def test_async_upload_bytes_wraps_sync_method():
    store = ObjectStore.__new__(ObjectStore)
    store.upload_bytes = MagicMock(return_value="depth/key.png")

    key = await store.async_upload_bytes(b"abc", content_type="image/png", prefix="depth", extension="png")

    assert key == "depth/key.png"
    store.upload_bytes.assert_called_once_with(b"abc", "image/png", "depth", "png")
