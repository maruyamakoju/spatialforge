"""Tests for reconstruct task error message normalization."""

from __future__ import annotations

from spatialforge.workers.tasks import _normalize_reconstruct_error


def test_normalize_reconstruct_error_open3d_missing():
    msg = _normalize_reconstruct_error(RuntimeError("TSDF backend requires optional dependency 'open3d'."))
    assert "TSDF backend unavailable" in msg
    assert ".[tsdf]" in msg


def test_normalize_reconstruct_error_metric_required():
    msg = _normalize_reconstruct_error(RuntimeError("TSDF backend requires metric depth maps, but frame 0 failed"))
    assert "TSDF requires metric depth" in msg


def test_normalize_reconstruct_error_passthrough():
    msg = _normalize_reconstruct_error(RuntimeError("arbitrary failure"))
    assert msg == "arbitrary failure"
