<div align="center">

# SpatialForge

**Any Camera. Instant 3D. One API.**

Turn any photo or video into depth maps, 3D measurements, and floor plans with a single API call.

[![CI](https://github.com/maruyamakoju/spatialforge/actions/workflows/ci.yml/badge.svg)](https://github.com/maruyamakoju/spatialforge/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)

[Live Demo](https://maruyamakoju.github.io/spatialforge/) | [API Docs](https://maruyamakoju.github.io/spatialforge/) | [Python SDK](#python-sdk)

</div>

---

## Architecture

```
Client Request
      |
      v
+------------------+     +------------------+     +-------------------+
|   FastAPI API     |     |   Celery Workers |     |   GPU Cluster     |
|   Gateway         |---->|   (async jobs)   |---->|   (inference)     |
|                   |     |                  |     |                   |
|  - Auth (API key) |     |  - reconstruct   |     |  - Depth Anything |
|  - Rate limiting  |     |  - floorplan     |     |  - TensorRT FP16  |
|  - Metrics        |     |  - segment-3d    |     |  - Multi-model    |
+--------+---------+     +--------+---------+     +-------------------+
         |                         |
         v                         v
+------------------+     +------------------+
|   MinIO / S3     |     |   Redis          |
|   Object Store   |     |   Queue + Cache  |
+------------------+     +------------------+
```

**Sync endpoints** (`/depth`, `/pose`, `/measure`) return results directly.
**Async endpoints** (`/reconstruct`, `/floorplan`, `/segment-3d`) enqueue jobs via Celery and return a `job_id` for polling.

## API Endpoints

| Endpoint | Method | Type | Description |
|----------|--------|------|-------------|
| `POST /v1/depth` | Sync | Monocular depth estimation (metric or relative) |
| `POST /v1/depth/visualize` | Sync | Colorized depth map as JPEG |
| `POST /v1/measure` | Sync | Real-world distance between two pixels |
| `POST /v1/pose` | Sync | Camera pose estimation from video/images |
| `POST /v1/reconstruct` | Async | 3D reconstruction (point cloud, mesh, gaussian) |
| `POST /v1/floorplan` | Async | Floor plan generation from walkthrough video |
| `POST /v1/segment-3d` | Async | 3D segmentation with text prompts |

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for rate limiting and job queue)
- MinIO or S3-compatible storage (for result files)

### Install & Run

```bash
git clone https://github.com/maruyamakoju/spatialforge.git
cd spatialforge
pip install -e ".[dev]"

# Configure (copy and edit .env)
cp .env.example .env

# Start the server
spatialforge
# -> http://localhost:8000
# -> Swagger docs at http://localhost:8000/docs
```

### Try It

```bash
# Get a depth map from any image
curl -X POST http://localhost:8000/v1/depth \
  -H "X-API-Key: $API_KEY" \
  -F "image=@photo.jpg" \
  -F "model=large" \
  -F "metric=true"
```

```json
{
  "depth_map_url": "http://localhost:9000/spatialforge/depth/abc123.png",
  "colormap_url": "http://localhost:9000/spatialforge/depth_vis/abc123.jpg",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "min_depth_m": 0.42,
    "max_depth_m": 8.73,
    "focal_length_px": 1440.0,
    "confidence_mean": 0.82
  },
  "processing_time_ms": 47.3
}
```

## Python SDK

```bash
pip install spatialforge-client
```

```python
from spatialforge_client import Client

sf = Client(api_key="sf_your_key", base_url="http://localhost:8000")

# Depth estimation
result = sf.depth("room.jpg", metric=True)
print(f"Range: {result.min_depth_m}m - {result.max_depth_m}m")

# Measure distance between two points
dist = sf.measure("room.jpg", point1=(100, 200), point2=(400, 200))
print(f"Distance: {dist.distance_m:.2f}m")

# Async 3D reconstruction
job = sf.reconstruct("walkthrough.mp4", quality="standard")
scene = job.wait()  # blocks until complete
scene.download("output/")
```

## Docker

### CPU (development / CI)

```bash
docker build -f Dockerfile.cpu -t spatialforge:cpu .
docker run -p 8000:8000 --env-file .env spatialforge:cpu
```

### GPU (production)

```bash
docker build -t spatialforge:gpu .
docker run --gpus all -p 8000:8000 --env-file .env spatialforge:gpu
```

### Full Stack (API + Workers + Redis + MinIO + Prometheus + Grafana)

```bash
cp .env.example .env   # edit secrets!
docker compose up -d

# CPU override (no GPU required)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

## One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/maruyamakoju/spatialforge)

Deploys the CPU demo (API + managed Redis) with auto-generated secrets. No GPU required.

## Models

All production models are **Apache 2.0** licensed (safe for commercial use).

| Model | Key | Task | Speed | License |
|-------|-----|------|-------|---------|
| DA2 Metric Large (Indoor) | `large` | Metric depth (meters) | Default | Apache 2.0 |
| DA2 Metric Large (Outdoor) | `metric-outdoor` | Metric depth (outdoor) | Default | Apache 2.0 |
| DA2 Base | `base` | Relative depth | Fast | Apache 2.0 |
| DA2 Small | `small` | Relative depth | Fastest | Apache 2.0 |

Research-only models (CC-BY-NC) are gated behind `RESEARCH_MODE=true`.

## Configuration

Key environment variables (see `.env.example` for full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `TORCH_DTYPE` | `float16` | `float16` or `float32` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `API_KEY_SECRET` | (generate one!) | HMAC secret for API keys |
| `ADMIN_API_KEY` | (generate one!) | Admin API key |
| `RESEARCH_MODE` | `false` | Enable CC-BY-NC models |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check spatialforge/ tests/

# Type check
mypy spatialforge/
```

## Project Structure

```
spatialforge/
  api/v1/          # FastAPI route handlers
  auth/            # API key auth + rate limiting
  inference/       # Depth, pose, measure, reconstruct engines
  models/          # Pydantic request/response models
  storage/         # MinIO object store with TTL tracking
  utils/           # Image and video processing utilities
  workers/         # Celery async tasks + webhooks
  config.py        # Environment-based configuration
  main.py          # FastAPI app entry point
  metrics.py       # Prometheus metrics
sdk/               # Python SDK (spatialforge-client)
tests/             # pytest test suite
site/              # Landing page
monitoring/        # Prometheus + Grafana configs
```

## License

MIT
