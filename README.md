<div align="center">

# SpatialForge

**Any Camera. Instant 3D. One API.**

Turn any photo or video into depth maps, 3D measurements, camera poses, and floor plans with a single API call. Production-ready spatial intelligence powered by state-of-the-art depth estimation.

[![CI](https://github.com/maruyamakoju/spatialforge/actions/workflows/ci.yml/badge.svg)](https://github.com/maruyamakoju/spatialforge/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-94%20passed-brightgreen.svg)](#testing)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![PyPI](https://img.shields.io/pypi/v/spatialforge-client?label=SDK&color=3775A9)](https://pypi.org/project/spatialforge-client/)

[Live API](https://spatialforge-demo.fly.dev/docs) &middot;
[Interactive Demo](https://maruyamakoju.github.io/spatialforge/demo.html) &middot;
[Documentation](https://maruyamakoju.github.io/spatialforge/docs.html) &middot;
[Pricing](https://maruyamakoju.github.io/spatialforge/pricing.html)

</div>

---

## What It Does

<table>
<tr>
<td width="50%">

### Input: Any Photo
```
photo.jpg (1920x1080)
```

</td>
<td width="50%">

### Output: Metric Depth Map
```json
{
  "min_depth_m": 0.42,
  "max_depth_m": 8.73,
  "confidence_mean": 0.94,
  "processing_time_ms": 47.3
}
```

</td>
</tr>
</table>

**One API call. Three lines of code.**

```python
import spatialforge_client as sf

client = sf.Client(api_key="sf_your_key")
result = client.depth("photo.jpg")
print(f"Distance range: {result.min_depth_m:.1f}m – {result.max_depth_m:.1f}m")
```

---

## Highlights

| Feature | Description |
|---------|-------------|
| **Depth Estimation** | Metric depth (meters) from a single image. Models: large, base, small |
| **Distance Measurement** | Measure real-world distance between any two pixels. ±2% with reference objects |
| **Camera Pose Recovery** | 6-DoF camera poses + intrinsics from video or image sequences |
| **3D Reconstruction** | Gaussian splatting, point cloud, or mesh from video (async) |
| **Floor Plans** | Walk through a room → get SVG/DXF floor plan (async) |
| **3D Segmentation** | Open-vocabulary object detection in 3D via text prompts (async) |
| **Billing** | Stripe-powered self-service: Free → Builder → Pro → Enterprise |
| **Production-Ready** | Auth, rate limiting, security headers, timeouts, metrics, CI/CD |

## Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │           SpatialForge Platform             │
                          │                                             │
 Client ──── HTTPS ──────►│  ┌────────────┐    ┌──────────────────┐    │
 (SDK/CLI/cURL)           │  │ FastAPI API │    │  Celery Workers  │    │
                          │  │  Gateway    │───►│  (async jobs)    │    │
                          │  │            │    │                  │    │
                          │  │ • Auth      │    │ • reconstruct    │    │
                          │  │ • Rate Limit│    │ • floorplan      │    │
                          │  │ • Security  │    │ • segment-3d     │    │
                          │  │ • Metrics   │    │                  │    │
                          │  │ • Timeout   │    │                  │    │
                          │  └─────┬──────┘    └────────┬─────────┘    │
                          │        │                     │              │
                          │        ▼                     ▼              │
                          │  ┌────────────┐    ┌──────────────────┐    │
                          │  │ Redis      │    │  Inference Engine │    │
                          │  │ • API Keys │    │  • Depth Anything │    │
                          │  │ • Billing  │    │  • TensorRT FP16  │    │
                          │  │ • Quotas   │    │  • Multi-model    │    │
                          │  └────────────┘    └──────────────────┘    │
                          │                                             │
                          │  ┌────────────┐    ┌──────────────────┐    │
                          │  │ MinIO / S3 │    │  Stripe Billing  │    │
                          │  │ File Store │    │  • Subscriptions  │    │
                          │  └────────────┘    │  • Webhooks       │    │
                          │                    └──────────────────┘    │
                          └─────────────────────────────────────────────┘
```

**Sync endpoints** (`/depth`, `/pose`, `/measure`) return results directly.
**Async endpoints** (`/reconstruct`, `/floorplan`, `/segment-3d`) enqueue jobs via Celery and return a `job_id` for polling.

## API Endpoints

| Endpoint | Type | Description |
|----------|------|-------------|
| `POST /v1/depth` | Sync | Monocular depth estimation (metric or relative) |
| `POST /v1/depth/visualize` | Sync | Colorized depth visualization as JPEG |
| `POST /v1/measure` | Sync | Real-world distance between two image points |
| `POST /v1/pose` | Sync | Camera pose estimation from video or images |
| `POST /v1/reconstruct` | Async | 3D reconstruction (Gaussian, point cloud, mesh) |
| `POST /v1/floorplan` | Async | Floor plan generation from walkthrough video |
| `POST /v1/segment-3d` | Async | Open-vocabulary 3D segmentation |
| `GET /v1/billing/plans` | Public | List plans and pricing |
| `GET /v1/billing/usage` | Auth | Current usage statistics |
| `POST /v1/billing/checkout` | Auth | Create Stripe Checkout session |
| `POST /v1/billing/portal` | Auth | Manage subscription via Stripe Portal |

## Quick Start

### Python SDK

```bash
pip install spatialforge-client
```

```python
import spatialforge_client as sf

client = sf.Client(api_key="sf_your_key")

# Depth estimation
result = client.depth("photo.jpg")
print(f"Depth: {result.min_depth_m:.2f}m - {result.max_depth_m:.2f}m")
result.save_depth_map("depth.png")

# Measure distance
m = client.measure("room.jpg", point1=(100, 200), point2=(500, 200))
print(f"Distance: {m.distance_cm:.1f} cm (confidence: {m.confidence:.0%})")

# Camera poses
poses = client.pose(video="walkthrough.mp4")
for p in poses.camera_poses:
    print(f"Frame {p.frame_index}: T={p.translation}")

# 3D reconstruction (async)
job = client.reconstruct("video.mp4", quality="high")
scene = job.wait()
print(f"Scene: {scene['scene_url']}")
```

### Async Python

```python
import spatialforge_client as sf

async with sf.AsyncClient(api_key="sf_your_key") as client:
    result = await client.depth("photo.jpg")
    print(result.min_depth_m)

    # Async job polling
    job = await client.reconstruct("video.mp4")
    scene = await job.async_wait()
```

### CLI

```bash
pip install spatialforge-client[cli]

export SPATIALFORGE_API_KEY=sf_your_key

spatialforge depth photo.jpg --model large --output depth.png
spatialforge measure room.jpg --p1 100,200 --p2 500,200
spatialforge reconstruct walkthrough.mp4 --quality high
```

### cURL

```bash
curl -X POST https://spatialforge-demo.fly.dev/v1/depth \
  -H "X-API-Key: sf_your_key" \
  -F "image=@photo.jpg" \
  -F "model=large" \
  -F "metric=true"
```

<details>
<summary><strong>Example Response</strong></summary>

```json
{
  "depth_map_url": "https://storage.example.com/depth/abc123.png",
  "colormap_url": "https://storage.example.com/depth_vis/abc123.jpg",
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

</details>

## Self-Hosted Setup

### Prerequisites

- Python 3.11+
- Redis (auth, rate limiting, job queue)
- MinIO or S3-compatible storage (optional &mdash; base64 fallback available)

### Install & Run

```bash
git clone https://github.com/maruyamakoju/spatialforge.git
cd spatialforge
pip install -e ".[dev]"

cp .env.example .env  # configure secrets

spatialforge  # starts on http://localhost:8000
```

### Docker

```bash
# CPU (development / CI)
docker build -f Dockerfile.cpu -t spatialforge:cpu .
docker run -p 8000:8000 --env-file .env spatialforge:cpu

# GPU (production)
docker build -t spatialforge:gpu .
docker run --gpus all -p 8000:8000 --env-file .env spatialforge:gpu

# Full stack (API + Redis + MinIO + Prometheus + Grafana)
docker compose up -d
```

### Cloud Deploy (Fly.io)

```bash
flyctl auth login
flyctl launch --copy-config --yes
flyctl secrets set API_KEY_SECRET=$(openssl rand -base64 36) \
                  ADMIN_API_KEY=sf_$(openssl rand -hex 16)
flyctl deploy
```

CI/CD auto-deploys to Fly.io on every push to `master` (after tests pass).

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

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `TORCH_DTYPE` | `float16` | `float16` or `float32` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO/S3 endpoint |
| `API_KEY_SECRET` | (generate!) | HMAC secret for API key hashing |
| `ADMIN_API_KEY` | (generate!) | Admin API key |
| `DEMO_MODE` | `false` | Allow unauthenticated access (demos only) |
| `RESEARCH_MODE` | `false` | Enable CC-BY-NC models |
| `STRIPE_SECRET_KEY` | (optional) | Stripe secret for billing |
| `STRIPE_WEBHOOK_SECRET` | (optional) | Stripe webhook signing |
| `ALLOWED_ORIGINS` | `localhost` | CORS allowed origins (JSON list) |

## Security

- **Authentication**: HMAC-based API key validation via Redis
- **Rate Limiting**: Sliding window per-IP (unauthenticated) + monthly quota per-key (authenticated)
- **Security Headers**: HSTS, X-Content-Type-Options, X-Frame-Options, CSP, Referrer-Policy, Permissions-Policy, X-API-Version
- **Request Timeouts**: 120s timeout on inference endpoints (returns 504)
- **Input Validation**: File size limits (20MB images, 120s video), content-type checks, image dimension caps (4096px), coordinate bounds checking, NaN/Inf rejection
- **CORS**: Configurable allowed origins (restrictive by default)
- **Billing**: Stripe webhook signature verification, plan-based access control
- **Docker**: Runs as non-root user (`spatialforge`)

## Testing

```bash
# Server tests (45 tests)
pytest tests/ -v

# SDK tests (49 tests)
cd sdk && pytest tests/ -v

# Lint
ruff check --config pyproject.toml spatialforge/ tests/
```

**94 total tests** covering API endpoints, billing, depth processing, video handling, middleware (security headers, request timeouts), input validation (coordinate bounds, NaN/Inf), SDK sync/async clients, and data models.

## Project Structure

```
spatialforge/
  api/v1/          # FastAPI route handlers (depth, pose, measure, billing, admin)
  auth/            # API key management + sliding window rate limiter
  billing/         # Stripe integration (checkout, webhooks, subscription sync)
  inference/       # Depth, pose, measure, reconstruct engines
  middleware/      # Security headers + request timeout middleware
  models/          # Pydantic request/response models with OpenAPI annotations
  storage/         # MinIO object store with TTL tracking
  utils/           # Image and video processing utilities
  workers/         # Celery async tasks + webhooks
  config.py        # Environment-based configuration (pydantic-settings)
  main.py          # FastAPI app entry point
  metrics.py       # Prometheus metrics + middleware
sdk/               # Python SDK (spatialforge-client) — sync, async, CLI
site/              # Landing page, docs, demo, pricing
infra/             # Redis infrastructure (Fly.io deployment)
tests/             # pytest test suite (45 server + 49 SDK)
monitoring/        # Prometheus + Grafana configs
.github/workflows/ # CI/CD (test, lint, docker, deploy, SDK publish)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI + Uvicorn |
| Inference | PyTorch + Hugging Face Transformers |
| Task Queue | Celery + Redis |
| Auth & Cache | Redis (API keys, rate limiting, billing state) |
| Object Storage | MinIO / S3 |
| Billing | Stripe (Checkout, Webhooks, Customer Portal) |
| Monitoring | Prometheus + Grafana (12-panel dashboard) |
| CI/CD | GitHub Actions (test, lint, deploy to Fly.io) |
| SDK | httpx + pydantic + click + rich |
| Deployment | Docker + Fly.io |

## License

MIT
