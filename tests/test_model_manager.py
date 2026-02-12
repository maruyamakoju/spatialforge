"""ModelManager registry and backend selection tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spatialforge.inference import model_manager as model_manager_module
from spatialforge.inference.model_manager import ModelManager, create_model_manager_from_settings


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


def test_create_model_manager_from_settings_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager_module.torch.cuda, "is_available", lambda: False)
    settings = SimpleNamespace(
        model_dir=tmp_path,
        device="cuda",
        torch_dtype="float16",
        research_mode=True,
        depth_backend="da3",
    )

    mm = create_model_manager_from_settings(settings)

    assert mm.device == "cpu"
    assert mm.research_mode is True
    assert mm.depth_backend == "da3"
    assert mm.dtype == model_manager_module.torch.float32


def test_create_model_manager_from_settings_gpu_uses_configured_dtype(tmp_path, monkeypatch):
    monkeypatch.setattr(model_manager_module.torch.cuda, "is_available", lambda: True)
    settings = SimpleNamespace(
        model_dir=tmp_path,
        device="cuda",
        torch_dtype="float16",
        research_mode=False,
        depth_backend="hf",
    )

    mm = create_model_manager_from_settings(settings)

    assert mm.device == "cuda"
    assert mm.research_mode is False
    assert mm.depth_backend == "hf"
    assert mm.dtype == model_manager_module.torch.float16
