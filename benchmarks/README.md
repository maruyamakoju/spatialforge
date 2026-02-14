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
  --reconstruct-backend tsdf \
  --frame-counts 30,60,120 \
  --width 960 --height 540 \
  --quality standard --runs 5 --warmup 1
```

Requires Open3D extra:

```bash
pip install -e ".[tsdf]"
```

Fast pipeline-overhead baseline with mocked stages:

```bash
python benchmarks/bench_reconstruct.py \
  --frame-counts 30,60,120 \
  --reconstruct-backend legacy \
  --mock-depth --mock-pose \
  --runs 5 --warmup 1
```

## Reconstruct Quality-Eval Skeleton

Default run uses synthetic fixtures with mocked depth/pose to produce reproducible JSON:

```bash
python benchmarks/quality_reconstruct.py \
  --reconstruct-backend legacy \
  --depth-backend da3
```

Run on real frame folders (`<dataset_dir>/<sequence_name>/*.jpg|png`) with real depth/pose:

```bash
python benchmarks/quality_reconstruct.py \
  --dataset-dir ./datasets/reconstruct_eval \
  --reconstruct-backend tsdf \
  --depth-backend da3 \
  --real-depth --real-pose
```

The quality JSON now includes `rendered_depth_fit` (GT-free):
- `coverage` (rendered depth overlap ratio)
- `abs_depth_error_m`
- `rel_depth_error`
- `inlier_ratio` (`|depth_error| < threshold`)

## Notes

- These scripts are for local/experimental use and are not part of CI.
- For consistent GPU comparisons, close other CUDA workloads and keep fixed seeds.
