"""ModelManager registry and backend selection tests."""

from __future__ import annotations

import pytest

from spatialforge.inference.model_manager import ModelManager


def test_model_manager_default_backend_is_hf(tmp_path):
    mm = ModelManager(model_dir=tmp_path, device="cpu", dtype="float32")

    assert mm.depth_backend == "hf"

    model_key = mm.resolve_model_name("small")
    info = mm.get_model_info(model_key)
    assert info.repo.endswith("-hf")


def test_model_manager_da3_backend_uses_da3_registry(tmp_path):
    mm = ModelManager(model_dir=tmp_path, device="cpu", dtype="float32", depth_backend="da3")

    model_key = mm.resolve_model_name("small")
    info = mm.get_model_info(model_key)

    assert info.repo == "depth-anything/DA3-SMALL"


def test_model_manager_invalid_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported depth backend"):
        ModelManager(model_dir=tmp_path, device="cpu", dtype="float32", depth_backend="invalid")


def test_model_manager_list_models_includes_backend(tmp_path):
    mm = ModelManager(model_dir=tmp_path, device="cpu", dtype="float32", depth_backend="da3")

    available = mm.list_available_models()

    assert available["backend"] == "da3"
    assert "commercial_apache2" in available
