# SpatialForge Benchmarks

Reproducible local benchmark harnesses for research and production tuning.

These scripts emit JSON with:
- git commit
- runtime environment (Python, torch, CUDA, GPU)
- benchmark config
- per-case latency stats (`mean`, `p50`, `p95`, `std`) and peak VRAM

Results are written under `benchmarks/results/` by default.

## Depth Benchmark

```bash
python benchmarks/bench_depth.py \
  --backend da3 \
  --models small,base,large \
  --resolutions 512,768,1024 \
  --runs 20 --warmup 5
```

## Reconstruct Benchmark

Real pipeline (for RTX4090 reporting):

```bash
python benchmarks/bench_reconstruct.py \
  --backend da3 \
  --frame-counts 30,60,120 \
  --width 960 --height 540 \
  --quality standard --runs 5 --warmup 1
```

Fast pipeline-overhead baseline with mocked stages:

```bash
python benchmarks/bench_reconstruct.py \
  --frame-counts 30,60,120 \
  --mock-depth --mock-pose \
  --runs 5 --warmup 1
```

## Notes

- These scripts are for local/experimental use and are not part of CI.
- For consistent GPU comparisons, close other CUDA workloads and keep fixed seeds.
