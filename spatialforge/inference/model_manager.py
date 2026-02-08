"""Model manager: loads, caches, and serves DA3 + auxiliary models."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Mapping from model size to HuggingFace model IDs (Depth Anything V2 as fallback, DA3 when available)
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "small": {
        "repo": "depth-anything/Depth-Anything-V2-Small",
        "backbone": "vits",
    },
    "base": {
        "repo": "depth-anything/Depth-Anything-V2-Base",
        "backbone": "vitb",
    },
    "large": {
        "repo": "depth-anything/Depth-Anything-V2-Large",
        "backbone": "vitl",
    },
    "giant": {
        "repo": "depth-anything/Depth-Anything-V2-Giant",
        "backbone": "vitg",
    },
}

# DA3 models (when available, these take priority)
DA3_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "small": {"repo": "DepthAnything/Depth-Anything-3-Small", "backbone": "vits"},
    "base": {"repo": "DepthAnything/Depth-Anything-3-Base", "backbone": "vitb"},
    "large": {"repo": "DepthAnything/Depth-Anything-3-Large", "backbone": "vitl"},
    "giant": {"repo": "DepthAnything/Depth-Anything-3-Giant", "backbone": "vitg"},
}


class ModelManager:
    """Thread-safe model loader with LRU-style caching.

    Ensures only one copy of each model variant lives in GPU memory.
    """

    def __init__(self, model_dir: Path, device: str = "cuda", dtype: str = "float16") -> None:
        self._model_dir = model_dir
        self._device = device
        self._dtype = getattr(torch, dtype, torch.float16)
        self._models: dict[str, Any] = {}
        self._lock = Lock()
        model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def get_depth_model(self, size: str = "giant") -> Any:
        """Load and return a depth estimation model (DA3 preferred, DAv2 fallback)."""
        cache_key = f"depth_{size}"
        if cache_key in self._models:
            return self._models[cache_key]

        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._models:
                return self._models[cache_key]

            model = self._load_depth_model(size)
            self._models[cache_key] = model
            return model

    def _load_depth_model(self, size: str) -> Any:
        """Attempt to load DA3 model, fall back to Depth Anything V2."""
        # Try DA3 first
        da3_info = DA3_MODEL_REGISTRY.get(size)
        if da3_info:
            try:
                model = self._load_da3(da3_info)
                logger.info("Loaded DA3-%s model", size)
                return model
            except Exception:
                logger.info("DA3-%s not available, falling back to DAv2", size)

        # Fallback to Depth Anything V2 via transformers pipeline
        dav2_info = MODEL_REGISTRY.get(size)
        if not dav2_info:
            raise ValueError(f"Unknown model size: {size}. Choose from: {list(MODEL_REGISTRY.keys())}")

        model = self._load_dav2(dav2_info)
        logger.info("Loaded DAv2-%s model", size)
        return model

    def _load_da3(self, info: dict[str, str]) -> Any:
        """Load a Depth Anything 3 model."""
        from transformers import pipeline

        pipe = pipeline(
            "depth-estimation",
            model=info["repo"],
            device=self._device,
            torch_dtype=self._dtype,
        )
        return pipe

    def _load_dav2(self, info: dict[str, str]) -> Any:
        """Load a Depth Anything V2 model via transformers pipeline."""
        from transformers import pipeline

        pipe = pipeline(
            "depth-estimation",
            model=info["repo"],
            device=self._device,
            torch_dtype=self._dtype,
        )
        return pipe

    def unload(self, cache_key: str) -> None:
        """Unload a model from memory."""
        with self._lock:
            model = self._models.pop(cache_key, None)
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Unloaded model: %s", cache_key)

    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            self._models.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded all models")

    def gpu_status(self) -> dict[str, Any]:
        """Return GPU memory usage stats."""
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
            "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        }
