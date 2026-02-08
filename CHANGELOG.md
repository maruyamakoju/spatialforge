# Changelog

All notable changes to SpatialForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-08

### Added

**API Endpoints**
- `POST /v1/depth` — Monocular metric depth estimation from a single image
- `POST /v1/depth/visualize` — Colorized depth map visualization as JPEG
- `POST /v1/measure` — Real-world distance measurement between two image points
- `POST /v1/pose` — Camera pose estimation from video or image sequences
- `POST /v1/reconstruct` — Async 3D reconstruction (Gaussian splat, point cloud, mesh)
- `POST /v1/floorplan` — Async floor plan generation from walkthrough video
- `POST /v1/segment-3d` — Async open-vocabulary 3D segmentation (beta)

**Billing**
- `GET /v1/billing/plans` — List available plans and pricing
- `GET /v1/billing/usage` — Current usage statistics
- `POST /v1/billing/checkout` — Stripe Checkout session creation
- `POST /v1/billing/portal` — Stripe Customer Portal session

**Security & Middleware**
- HMAC-based API key authentication via Redis
- Sliding window rate limiting (per-IP + monthly quota per-key)
- Security headers: HSTS, CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy
- Request timeout middleware (120s on inference endpoints, returns 504)
- Input validation: file size limits, content-type checks, coordinate bounds, NaN/Inf rejection
- CORS with configurable allowed origins
- X-API-Version response header

**Infrastructure**
- FastAPI + Uvicorn application server
- Celery + Redis async task queue for long-running jobs
- MinIO/S3 object storage with TTL tracking
- Docker images (CPU and GPU) with non-root user
- Fly.io deployment with CI/CD auto-deploy on push to master
- Prometheus metrics: request latency, inference duration, API key usage, rate limit hits, error counts, GPU memory, file upload sizes
- Grafana dashboard with 12 panels

**SDK (spatialforge-client v0.1.0)**
- Synchronous `Client` with all endpoint methods
- Asynchronous `AsyncClient` with identical API
- Typed data models: `DepthResult`, `MeasureResult`, `PoseResult`, `AsyncJob`
- Async job polling with `.wait()` and `.async_wait()`
- CLI tool: `spatialforge depth`, `spatialforge measure`, `spatialforge reconstruct`
- Retry with exponential backoff and typed error classes

**Testing**
- 139 tests: 62 server + 77 SDK
- CI pipeline: test (Python 3.11 + 3.12), SDK test, lint, Docker build, auto-deploy

**Documentation**
- Landing page with feature showcase
- Interactive demo page (depth estimation + distance measurement)
- API documentation with Python, JavaScript, Go, and cURL examples
- Pricing page with plan comparison
- OpenAPI/Swagger interactive reference

[0.1.0]: https://github.com/maruyamakoju/spatialforge/releases/tag/v0.1.0
