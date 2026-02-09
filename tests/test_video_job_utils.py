"""Unit tests for shared async video job helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from spatialforge.api.v1 import _video_job_utils as vju


class _FakeUploadFile:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._data):
            return b""
        if size < 0:
            chunk = self._data[self._offset :]
            self._offset = len(self._data)
            return chunk
        end = min(len(self._data), self._offset + size)
        chunk = self._data[self._offset : end]
        self._offset = end
        return chunk


def _fake_request_with_store(store: object | None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(object_store=store)))


def _temp_video_path(_: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"fake-video-bytes")
        return Path(tmp.name)


async def _temp_video_path_from_stream(_video, max_file_size: int, chunk_size: int = 1024 * 1024) -> Path:
    _ = max_file_size
    _ = chunk_size
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(b"fake-video-bytes")
        return Path(tmp.name)


@pytest.mark.asyncio
async def test_validate_and_store_video_size_limit(monkeypatch):
    request = _fake_request_with_store(MagicMock())
    video = _FakeUploadFile(b"x" * 6)

    with pytest.raises(HTTPException) as exc:
        await vju.validate_and_store_video(
            request,
            video,  # type: ignore[arg-type]
            max_file_size=5,
            max_duration_s=120,
        )

    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_validate_and_store_video_min_duration(monkeypatch):
    monkeypatch.setattr(vju, "_write_upload_to_temp", _temp_video_path_from_stream)
    monkeypatch.setattr(vju, "validate_video", lambda _path, max_duration_s: SimpleNamespace(duration_s=8.0))

    request = _fake_request_with_store(MagicMock())
    video = _FakeUploadFile(b"123")

    with pytest.raises(HTTPException) as exc:
        await vju.validate_and_store_video(
            request,
            video,  # type: ignore[arg-type]
            max_file_size=500,
            max_duration_s=300,
            min_duration_s=10.0,
            min_duration_error="too short",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "too short"


@pytest.mark.asyncio
async def test_validate_and_store_video_store_required(monkeypatch):
    monkeypatch.setattr(vju, "_write_upload_to_temp", _temp_video_path_from_stream)
    monkeypatch.setattr(vju, "validate_video", lambda _path, max_duration_s: SimpleNamespace(duration_s=15.0))

    request = _fake_request_with_store(None)
    video = _FakeUploadFile(b"123")

    with pytest.raises(HTTPException) as exc:
        await vju.validate_and_store_video(
            request,
            video,  # type: ignore[arg-type]
            max_file_size=500,
            max_duration_s=120,
        )

    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_validate_and_store_video_success(monkeypatch):
    monkeypatch.setattr(vju, "_write_upload_to_temp", _temp_video_path_from_stream)
    monkeypatch.setattr(vju, "validate_video", lambda _path, max_duration_s: SimpleNamespace(duration_s=42.5))

    store = MagicMock()
    store.async_upload_file = AsyncMock(return_value="uploads/test.mp4")

    request = _fake_request_with_store(store)
    video = _FakeUploadFile(b"123")

    uploaded = await vju.validate_and_store_video(
        request,
        video,  # type: ignore[arg-type]
        max_file_size=500,
        max_duration_s=120,
    )

    assert uploaded.video_key == "uploads/test.mp4"
    assert uploaded.info.duration_s == 42.5
    store.async_upload_file.assert_awaited_once()
