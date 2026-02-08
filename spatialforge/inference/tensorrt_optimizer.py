"""TensorRT optimization for depth estimation models.

Converts PyTorch models to TensorRT FP16 engines for 2-5x inference speedup.
Requires: pip install tensorrt (and NVIDIA TensorRT system libraries).

Usage:
    from spatialforge.inference.tensorrt_optimizer import TensorRTOptimizer

    optimizer = TensorRTOptimizer(cache_dir=Path("./models/trt_cache"))
    engine = optimizer.optimize(torch_model, input_shape=(1, 3, 384, 384))
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TensorRTOptimizer:
    """Manages TensorRT engine creation and caching for depth models."""

    def __init__(self, cache_dir: Path, dtype: str = "float16") -> None:
        self._cache_dir = cache_dir
        self._dtype = dtype
        self._engines: dict[str, Any] = {}
        cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_available() -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt  # noqa: F401
            return True
        except ImportError:
            return False

    def get_engine(self, model_name: str) -> Any | None:
        """Get a cached TensorRT engine if available."""
        return self._engines.get(model_name)

    def optimize_onnx(
        self,
        onnx_path: Path,
        model_name: str,
        input_shapes: dict[str, tuple[int, ...]],
        force_rebuild: bool = False,
    ) -> Any:
        """Build a TensorRT engine from an ONNX model.

        Args:
            onnx_path: Path to the ONNX model file.
            model_name: Unique name for caching.
            input_shapes: Dict of input_name -> shape for the model.
            force_rebuild: Force rebuild even if cache exists.

        Returns:
            TensorRT ICudaEngine ready for inference.
        """
        if not self.is_available():
            raise RuntimeError("TensorRT is not installed. Run: pip install tensorrt")

        import tensorrt as trt

        # Check cache
        cache_path = self._cache_dir / f"{model_name}_{self._dtype}.engine"
        if cache_path.exists() and not force_rebuild:
            logger.info("Loading cached TensorRT engine: %s", cache_path)
            engine = self._load_engine(cache_path)
            self._engines[model_name] = engine
            return engine

        logger.info("Building TensorRT engine from ONNX: %s", onnx_path)
        t0 = time.perf_counter()

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt_logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        # Build config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4 GB

        if self._dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self._dtype == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

        # Set optimization profiles for dynamic batch sizes
        profile = builder.create_optimization_profile()
        for name, shape in input_shapes.items():
            min_shape = (1, *shape[1:])
            opt_shape = shape
            max_shape = (max(8, shape[0] * 2), *shape[1:])
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        # Save to cache
        cache_path.write_bytes(serialized)
        elapsed = time.perf_counter() - t0
        logger.info("TensorRT engine built in %.1fs → %s", elapsed, cache_path)

        # Deserialize
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        self._engines[model_name] = engine
        return engine

    def _load_engine(self, path: Path) -> Any:
        """Load a serialized TensorRT engine from disk."""
        import tensorrt as trt

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: Path,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict | None = None,
    ) -> Path:
        """Export a PyTorch model to ONNX format.

        Args:
            model: PyTorch model in eval mode.
            dummy_input: Example input tensor.
            output_path: Where to save the .onnx file.
            input_names: Names for input tensors.
            output_names: Names for output tensors.
            dynamic_axes: Dynamic axes specification.

        Returns:
            Path to the exported ONNX file.
        """
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]
        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        logger.info("Exported ONNX model: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
        return output_path


class TensorRTInferenceSession:
    """Run inference using a TensorRT engine.

    Manages CUDA memory and context for executing TRT engines.
    """

    def __init__(self, engine: Any) -> None:
        if not TensorRTOptimizer.is_available():
            raise RuntimeError("TensorRT not available")

        import tensorrt as trt

        self._engine = engine
        self._context = engine.create_execution_context()

        # Allocate buffers
        self._inputs: list[dict] = []
        self._outputs: list[dict] = []
        self._allocations: list[Any] = []

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            shape = engine.get_tensor_shape(name)
            size = np.prod(shape)
            allocation = torch.empty(int(size), dtype=torch.uint8, device="cuda")

            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
            }

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(binding)
            else:
                self._outputs.append(binding)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference on input data.

        Args:
            input_data: Numpy array matching the engine's input shape and dtype.

        Returns:
            Output numpy array.
        """
        # Copy input to GPU
        input_tensor = torch.from_numpy(input_data).cuda()

        # Set tensor addresses
        for inp in self._inputs:
            self._context.set_tensor_address(inp["name"], input_tensor.data_ptr())
        for out in self._outputs:
            self._context.set_tensor_address(out["name"], out["allocation"].data_ptr())

        # Execute
        self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Copy output
        out_binding = self._outputs[0]
        output = out_binding["allocation"][: np.prod(out_binding["shape"])].cpu().numpy()
        return output.reshape(out_binding["shape"]).astype(out_binding["dtype"])
