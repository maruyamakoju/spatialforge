"""Model manager: loads, caches, and serves depth estimation models.

LICENSE SAFETY:
  - Production mode (default): Apache 2.0 models ONLY.
  - Research mode (opt-in):    CC-BY-NC 4.0 models available (non-commercial).
  - The `research_mode` flag MUST be explicitly enabled in config.
  - Shipping CC-BY-NC models to paying customers = lawsuit. Don't do it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
_SUPPORTED_BACKENDS = {"hf", "da3"}


@dataclass(frozen=True)
class ModelInfo:
    repo: str
    license: str  # "apache-2.0" | "cc-by-nc-4.0"
    task: str  # "metric_depth" | "relative_depth" | "multi_view"
    description: str


class _DA3DepthAdapter:
    """Adapter to make DepthAnything3 callable like HF depth-estimation pipeline."""

    def __init__(self, model: Any, process_res: int = 504) -> None:
        self._model = model
        self._process_res = process_res

    def __call__(self, image: Any) -> dict[str, np.ndarray]:
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        image_np = np.asarray(image)

        prediction = self._model.inference([image_np], process_res=self._process_res)
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        return {"depth": depth}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MODEL REGISTRY — source of truth for what we can ship
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Apache 2.0 (SAFE for commercial use) ─────────────────────
# Backend: HuggingFace transformers pipeline (Depth Anything V2 models).
COMMERCIAL_MODELS_HF: dict[str, ModelInfo] = {
    "da3-metric-large": ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
        license="apache-2.0",
        task="metric_depth",
        description="HF backend: DA2 Metric Large (indoor) — stable production fallback",
    ),
    "da3-metric-large-outdoor": ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
        license="apache-2.0",
        task="metric_depth",
        description="HF backend: DA2 Metric Large (outdoor)",
    ),
    "da3-base": ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Base-hf",
        license="apache-2.0",
        task="relative_depth",
        description="HF backend: DA2 Base relative depth",
    ),
    "da3-small": ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Small-hf",
        license="apache-2.0",
        task="relative_depth",
        description="HF backend: DA2 Small relative depth",
    ),
}

# Backend: depth_anything_3 native API (real DA3 model family).
COMMERCIAL_MODELS_DA3: dict[str, ModelInfo] = {
    "da3-metric-large": ModelInfo(
        repo="depth-anything/DA3METRIC-LARGE",
        license="apache-2.0",
        task="metric_depth",
        description="DA3 backend: DA3METRIC-LARGE metric depth",
    ),
    "da3-metric-large-outdoor": ModelInfo(
        repo="depth-anything/DA3METRIC-LARGE",
        license="apache-2.0",
        task="metric_depth",
        description="DA3 backend: DA3METRIC-LARGE (outdoor fallback until outdoor-specific checkpoint is added)",
    ),
    "da3-base": ModelInfo(
        repo="depth-anything/DA3-BASE",
        license="apache-2.0",
        task="relative_depth",
        description="DA3 backend: DA3-BASE relative depth",
    ),
    "da3-small": ModelInfo(
        repo="depth-anything/DA3-SMALL",
        license="apache-2.0",
        task="relative_depth",
        description="DA3 backend: DA3-SMALL relative depth",
    ),
}

# ── CC-BY-NC 4.0 (RESEARCH ONLY — never ship to paying customers) ──
RESEARCH_MODELS_HF: dict[str, ModelInfo] = {
    "da3-large": ModelInfo(
        repo="depth-anything/Depth-Anything-V2-Large-hf",
        license="cc-by-nc-4.0",
        task="relative_depth",
        description="HF backend: DA2 Large — research only",
    ),
}

RESEARCH_MODELS_DA3: dict[str, ModelInfo] = {
    "da3-large": ModelInfo(
        repo="depth-anything/DA3-LARGE",
        license="cc-by-nc-4.0",
        task="relative_depth",
        description="DA3 backend: DA3-LARGE — research only",
    ),
}

# Aliases for API compatibility (user-facing names → registry keys)
MODEL_ALIASES: dict[str, str] = {
    # These are the names users pass to model= parameter
    "small": "da3-small",
    "base": "da3-base",
    "large": "da3-metric-large",       # "large" in production → metric-large (Apache 2.0)
    "giant": "da3-metric-large",       # "giant" in production → metric-large (safest default)
    "metric-indoor": "da3-metric-large",
    "metric-outdoor": "da3-metric-large-outdoor",
    # Research aliases (only work with research_mode=true)
    "research-large": "da3-large",
}


class ModelManager:
    """Thread-safe model loader with license enforcement.

    Production mode: only Apache 2.0 models are loadable.
    Research mode:   all models available (opt-in via config).
    """

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda",
        dtype: str = "float16",
        research_mode: bool = False,
        depth_backend: str = "hf",
    ) -> None:
        backend = depth_backend.lower()
        if backend not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported depth backend: '{depth_backend}'. "
                f"Supported values: {sorted(_SUPPORTED_BACKENDS)}"
            )

        self._model_dir = model_dir
        self._device = device
        self._dtype = getattr(torch, dtype, torch.float16)
        self._research_mode = research_mode
        self._depth_backend = backend
        self._models: dict[str, Any] = {}
        self._model_info: dict[str, ModelInfo] = {}  # tracks what's loaded
        self._lock = Lock()
        model_dir.mkdir(parents=True, exist_ok=True)

        if research_mode:
            logger.warning(
                "RESEARCH MODE enabled — CC-BY-NC models available. "
                "DO NOT use in production or with paying customers."
            )

    @property
    def device(self) -> str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def research_mode(self) -> bool:
        return self._research_mode

    @property
    def depth_backend(self) -> str:
        return self._depth_backend

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def _commercial_models(self) -> dict[str, ModelInfo]:
        if self._depth_backend == "da3":
            return COMMERCIAL_MODELS_DA3
        return COMMERCIAL_MODELS_HF

    def _research_models(self) -> dict[str, ModelInfo]:
        if self._depth_backend == "da3":
            return RESEARCH_MODELS_DA3
        return RESEARCH_MODELS_HF

    def resolve_model_name(self, user_input: str) -> str:
        """Resolve user-facing model name to registry key.

        Raises ValueError if the model requires research_mode but it's disabled.
        """
        commercial_models = self._commercial_models()
        research_models = self._research_models()

        # Direct registry key
        if user_input in commercial_models:
            return user_input
        if user_input in research_models:
            if not self._research_mode:
                raise ValueError(
                    f"Model '{user_input}' is CC-BY-NC (research only). "
                    f"Enable research_mode=true in config to use it. "
                    f"DO NOT use CC-BY-NC models with paying customers."
                )
            return user_input

        # Alias lookup
        registry_key = MODEL_ALIASES.get(user_input)
        if registry_key is None:
            available = list(commercial_models.keys())
            if self._research_mode:
                available += list(research_models.keys())
            raise ValueError(
                f"Unknown model: '{user_input}'. Available: {available}"
            )

        # Check license
        if registry_key in research_models and not self._research_mode:
            # Silently remap to the best commercial alternative
            logger.info(
                "Model '%s' maps to research-only '%s'. Remapping to 'da3-metric-large' (Apache 2.0).",
                user_input, registry_key,
            )
            return "da3-metric-large"

        return registry_key

    def get_model_info(self, model_key: str) -> ModelInfo:
        """Get model metadata by registry key."""
        commercial_models = self._commercial_models()
        research_models = self._research_models()

        if model_key in commercial_models:
            return commercial_models[model_key]
        if model_key in research_models:
            return research_models[model_key]
        raise ValueError(f"Unknown model key: {model_key}")

    def get_depth_model(self, size: str = "large") -> tuple[Any, ModelInfo]:
        """Load and return a depth estimation pipeline + its metadata.

        Args:
            size: User-facing model name (e.g., "small", "base", "large", "giant").

        Returns:
            Tuple of (pipeline, ModelInfo).
        """
        registry_key = self.resolve_model_name(size)
        cache_key = f"depth_{self._depth_backend}_{registry_key}"

        if cache_key in self._models:
            return self._models[cache_key], self._model_info[cache_key]

        with self._lock:
            if cache_key in self._models:
                return self._models[cache_key], self._model_info[cache_key]

            info = self.get_model_info(registry_key)
            pipe = self._load_pipeline(info)
            self._models[cache_key] = pipe
            self._model_info[cache_key] = info
            logger.info(
                "Loaded model: %s (%s, %s)",
                registry_key, info.license, info.task,
            )
            return pipe, info

    def _load_pipeline(self, info: ModelInfo) -> Any:
        """Load a depth model using the selected backend."""
        if self._depth_backend == "da3":
            return self._load_da3_pipeline(info)
        return self._load_hf_pipeline(info)

    def _load_hf_pipeline(self, info: ModelInfo) -> Any:
        """Load a model via HuggingFace transformers pipeline."""
        from transformers import pipeline

        device_arg = 0 if self._device == "cuda" and torch.cuda.is_available() else -1

        pipe = pipeline(
            "depth-estimation",
            model=info.repo,
            device=device_arg,
            torch_dtype=self._dtype if device_arg >= 0 else torch.float32,
        )
        return pipe

    def _load_da3_pipeline(self, info: ModelInfo) -> Any:
        """Load a DA3 model via depth_anything_3 and adapt it to pipeline-like call."""
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError(
                "DA3 backend selected but depth_anything_3 is not installed. "
                "Install Depth-Anything-3 (or set DEPTH_BACKEND=hf)."
            ) from exc

        device = torch.device("cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu")

        model = DepthAnything3.from_pretrained(info.repo)
        model = model.to(device=device)
        if device.type == "cuda":
            model = model.to(dtype=self._dtype)
        model.eval()

        return _DA3DepthAdapter(model)

    def unload(self, cache_key: str) -> None:
        """Unload a model from memory."""
        with self._lock:
            model = self._models.pop(cache_key, None)
            self._model_info.pop(cache_key, None)
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("Unloaded model: %s", cache_key)

    def unload_all(self) -> None:
        """Unload all models."""
        with self._lock:
            self._models.clear()
            self._model_info.clear()
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

    def list_available_models(self) -> dict[str, Any]:
        """List all available models grouped by license."""
        commercial_models = self._commercial_models()
        research_models = self._research_models()

        commercial = [
            {"key": k, "repo": v.repo, "task": v.task, "description": v.description}
            for k, v in commercial_models.items()
        ]
        result: dict[str, list[dict] | str] = {
            "backend": self._depth_backend,
            "commercial_apache2": commercial,
        }

        if self._research_mode:
            research = [
                {"key": k, "repo": v.repo, "task": v.task, "description": v.description}
                for k, v in research_models.items()
            ]
            result["research_cc_by_nc"] = research

        return result


def create_model_manager_from_settings(settings: Any) -> ModelManager:
    """Create ModelManager from app settings with consistent runtime defaults.

    This helper keeps API server and worker initialization paths in sync.
    """
    runtime_device = settings.device if torch.cuda.is_available() else "cpu"
    if runtime_device == "cpu":
        logger.warning("CUDA not available — running on CPU (slow)")

    runtime_dtype = settings.torch_dtype if runtime_device != "cpu" else "float32"

    return ModelManager(
        model_dir=settings.model_dir,
        device=runtime_device,
        dtype=runtime_dtype,
        research_mode=settings.research_mode,
        depth_backend=settings.depth_backend,
    )
