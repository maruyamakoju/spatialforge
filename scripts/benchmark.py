#!/usr/bin/env python3
"""Benchmark SpatialForge inference performance.

Usage:
    python scripts/benchmark.py                     # Benchmark default (giant)
    python scripts/benchmark.py --model small       # Benchmark specific model
    python scripts/benchmark.py --model all         # Benchmark all models
    python scripts/benchmark.py --resolution 1920   # Custom resolution
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch


def benchmark_depth(
    model_size: str,
    resolution: int = 1080,
    num_warmup: int = 3,
    num_runs: int = 10,
) -> dict:
    """Benchmark depth estimation for a given model size."""
    from spatialforge.inference.depth_engine import DepthEngine
    from spatialforge.inference.model_manager import ModelManager

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "float16" if device == "cuda" else "float32"

    print(f"\n{'='*60}")
    print(f"Benchmarking: DA-{model_size} @ {resolution}p on {device}")
    print(f"{'='*60}")

    mm = ModelManager(model_dir=Path("./models"), device=device, dtype=dtype)
    engine = DepthEngine(mm)

    # Generate test image
    h = resolution
    w = int(resolution * 16 / 9)
    test_image = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for _ in range(num_warmup):
        engine.estimate(test_image, model_size=model_size)

    # Synchronize GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    latencies = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        result = engine.estimate(test_image, model_size=model_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.1f}ms")

    stats = {
        "model": model_size,
        "resolution": f"{w}x{h}",
        "device": device,
        "mean_ms": round(statistics.mean(latencies), 1),
        "median_ms": round(statistics.median(latencies), 1),
        "min_ms": round(min(latencies), 1),
        "max_ms": round(max(latencies), 1),
        "std_ms": round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 1),
        "fps": round(1000 / statistics.mean(latencies), 1),
    }

    print(f"\nResults:")
    print(f"  Mean:   {stats['mean_ms']}ms")
    print(f"  Median: {stats['median_ms']}ms")
    print(f"  Min:    {stats['min_ms']}ms")
    print(f"  Max:    {stats['max_ms']}ms")
    print(f"  Std:    {stats['std_ms']}ms")
    print(f"  FPS:    {stats['fps']}")

    if torch.cuda.is_available():
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"  VRAM:   {vram:.2f} GB")
        stats["vram_gb"] = round(vram, 2)

    mm.unload_all()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark depth estimation")
    parser.add_argument("--model", choices=["small", "base", "large", "giant", "all"], default="giant")
    parser.add_argument("--resolution", type=int, default=1080)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    models = ["small", "base", "large", "giant"] if args.model == "all" else [args.model]

    all_results = []
    for m in models:
        result = benchmark_depth(m, resolution=args.resolution, num_runs=args.runs, num_warmup=args.warmup)
        all_results.append(result)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Model':<10} {'Resolution':<12} {'Mean(ms)':<10} {'FPS':<8} {'VRAM(GB)':<10}")
        print("-" * 50)
        for r in all_results:
            print(f"{r['model']:<10} {r['resolution']:<12} {r['mean_ms']:<10} {r['fps']:<8} {r.get('vram_gb', 'N/A')}")


if __name__ == "__main__":
    main()
