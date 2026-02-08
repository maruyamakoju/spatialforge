# SpatialForge Architecture

**Version**: 0.1.0
**Date**: 2026-02-08
**Status**: Production-ready

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Technology Stack](#technology-stack)
6. [Deployment Architecture](#deployment-architecture)
7. [Security Architecture](#security-architecture)
8. [Scalability & Performance](#scalability--performance)
9. [Monitoring & Observability](#monitoring--observability)
10. [Future Roadmap](#future-roadmap)

---

## System Overview

SpatialForge is a **spatial intelligence API platform** that transforms ordinary camera input into precise 3D understanding using state-of-the-art depth estimation models.

### Core Capabilities

| Feature | Description | Latency | Accuracy |
|---------|-------------|---------|----------|
| **Depth Estimation** | Monocular metric depth (meters) | ~50ms | 10cm @ 5m |
| **Distance Measurement** | Real-world distance between pixels | ~50ms | ±2% with reference |
| **Camera Pose Recovery** | 6-DoF poses + intrinsics | ~200ms | Competitive with COLMAP |
| **3D Reconstruction** | Point cloud from video | ~30s (async) | Dense, colored |
| **Floor Plan Generation** | SVG/DXF from walkthrough | ~45s (async) | Architectural scale |
| **3D Segmentation** | Open-vocabulary object detection | ~60s (async) | Depth-aware masks |

### Design Principles

1. **API-First**: Everything accessible via RESTful HTTP API
2. **Production-Ready**: Auth, rate limiting, observability, security headers
3. **License-Safe**: Apache 2.0 models only (commercial-safe)
4. **Developer-Friendly**: Comprehensive docs, Python SDK, CLI tools
5. **Scalable**: Async job processing, GPU pooling, horizontal scaling

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
│  Python SDK   │   JavaScript SDK   │   CLI   │   cURL/Direct   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                        ┌────────▼────────┐
                        │   HTTPS (TLS)   │
                        │  Load Balancer  │
                        └────────┬────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                          API GATEWAY                             │
│  FastAPI + Uvicorn (async ASGI)                                │
│                                                                  │
│  Middleware Stack (executed in order):                          │
│  1. CORS             ← Cross-origin policy                     │
│  2. SecurityHeaders  ← OWASP headers (HSTS, CSP, etc.)        │
│  3. Timeout          ← 120s timeout on /depth, /measure, /pose │
│  4. WebhookRateLimit ← DoS protection for /webhooks           │
│  5. RateLimit        ← Sliding window + monthly quota         │
│  6. Metrics          ← Prometheus instrumentation             │
│  7. RequestTracing   ← X-Request-ID propagation               │
│                                                                  │
│  Endpoints:                                                      │
│  • Sync:  /v1/depth, /v1/measure, /v1/pose                   │
│  • Async: /v1/reconstruct, /v1/floorplan, /v1/segment-3d     │
│  • Billing: /v1/billing/*                                      │
│  • Admin: /v1/admin/*                                          │
└────────┬───────────────┬───────────────┬──────────────┬─────────┘
         │               │               │              │
         ▼               ▼               ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────┐
│   Redis     │ │   MinIO/S3  │ │    Stripe    │ │  Celery  │
│             │ │             │ │              │ │  Workers │
│ • API keys  │ │ • Depth maps│ │ • Checkout   │ │          │
│ • Rate lim. │ │ • Videos    │ │ • Webhooks   │ │ GPU-heavy│
│ • Billing   │ │ • Results   │ │ • Subscript. │ │ async    │
│ • Quotas    │ │ • TTL track │ │              │ │ jobs     │
└─────────────┘ └─────────────┘ └──────────────┘ └────┬─────┘
                                                        │
                                               ┌────────▼────────┐
                                               │  Inference Eng. │
                                               │  PyTorch + GPU  │
                                               │  • DepthEngine  │
                                               │  • PoseEngine   │
                                               │  • MeasureEngine│
                                               │  • ReconEngine  │
                                               └─────────────────┘
```

---

## Component Design

### 1. API Gateway (FastAPI)

**Responsibilities**:
- HTTP request routing
- Input validation (Pydantic models)
- Authentication & authorization
- Rate limiting enforcement
- Metrics collection
- Error handling

**Key Files**:
- `main.py` - Application factory, middleware stack
- `api/v1/*.py` - Endpoint handlers
- `models/requests.py`, `models/responses.py` - Pydantic schemas

**Design Pattern**: Dependency injection for shared resources (Redis, ModelManager, ObjectStore)

**Scalability**: Stateless, horizontally scalable behind load balancer

---

### 2. Authentication & Authorization (`auth/`)

**System**: HMAC-based API key validation

**Flow**:
```
1. Client sends X-API-Key header
2. API gateway hashes key with API_KEY_SECRET
3. Lookup in Redis: apikey:{hash} → {plan, owner, quota, ...}
4. Validate quota not exceeded
5. Inject KeyInfo into request state
6. Endpoint checks plan tier for authorization
```

**Key Files**:
- `auth/api_keys.py` - APIKeyManager, hash_api_key(), Plan enum
- `auth/rate_limiter.py` - RateLimiterMiddleware (sliding window)

**Security**:
- Secrets never stored in plaintext (HMAC hash only)
- Constant-time comparison prevents timing attacks
- Monthly quotas per API key (free=100, builder=5k, pro=50k)

---

### 3. Inference Engines (`inference/`)

**Architecture**: Shared model cache per worker process

**Engines**:

1. **DepthEngine** (`depth_engine.py`)
   - Input: RGB image (numpy array)
   - Output: Depth map (float32, H×W), metadata (min/max depth, confidence)
   - Models: DA2-Small/Base/Large (Apache 2.0)
   - Focal length: EXIF → explicit → heuristic (0.9 * max(w, h))
   - GPU memory: Clears cache after each inference

2. **MeasureEngine** (`measure_engine.py`)
   - Input: Image + two pixel coordinates
   - Output: Real-world distance in meters
   - Method: Back-project pixels → 3D points → Euclidean distance
   - Calibration: Optional reference object (A4 paper, credit card)

3. **PoseEngine** (`pose_engine.py`)
   - Input: List of images (video frames)
   - Output: Camera poses (rotation, translation, intrinsics) + sparse point cloud
   - Method: ORB features → Essential matrix → pose decomposition
   - Use case: Multi-view reconstruction, SLAM

4. **ReconstructEngine** (`reconstruct_engine.py`)
   - Input: Video frames
   - Output: Dense point cloud or Gaussian splat PLY
   - Pipeline: Keyframes → Depth + Pose → Back-projection → 3D fusion
   - Quality modes: draft (small model, 30 frames) / standard (large, 100) / high (giant, 200)

**Model Management** (`model_manager.py`):
- Lazy loading: Models loaded on first use, cached per worker
- License enforcement: `research_mode=False` prevents CC-BY-NC models
- Thread-safe: Lock guards around HuggingFace pipeline() calls
- GPU monitoring: VRAM usage tracking

---

### 4. Async Job Processing (Celery)

**Why Celery?**
- GPU-intensive tasks block API workers
- Need progress updates (state="PROCESSING", meta={step: "..."})
- Retry logic with exponential backoff
- Distributed task queue (scale workers independently)

**Task Types**:

| Task | Queue | Avg Duration | Retry Strategy |
|------|-------|--------------|----------------|
| `reconstruct_task` | gpu_heavy | ~30s | 3 attempts, exp backoff |
| `floorplan_task` | gpu_heavy | ~45s | 3 attempts, exp backoff |
| `segment_3d_task` | gpu_heavy | ~60s | 3 attempts, exp backoff |
| `cleanup_expired_results` | default | ~5s | 2 attempts |

**Job Lifecycle**:
```
1. Client POSTs to /v1/reconstruct → Returns job_id immediately (202 Accepted)
2. Celery worker picks up task from Redis queue
3. Worker updates state: PENDING → PROCESSING → SUCCESS/FAILURE
4. Client polls GET /v1/jobs/{job_id} for status
5. On completion, webhook fired (if provided)
6. Result uploaded to MinIO, presigned URL returned
```

**Configuration** (`workers/celery_app.py`):
- Broker & backend: Redis
- Serialization: JSON (not pickle, for security)
- ACKs: Late (task_acks_late=True) for reliability
- Prefetch: 1 (one GPU-heavy task per worker)
- Result expiry: 24 hours

---

### 5. Object Storage (MinIO / S3)

**Purpose**: Store large binary artifacts (depth maps, videos, 3D models)

**Architecture**:
- Presigned URLs (24-hour TTL) for secure client download
- TTL tracking: Redis set tracks object keys, cleanup job deletes expired
- Bucket structure: Flat namespace with prefixes (depth/, videos/, reconstructions/)

**Files Stored**:
- Depth maps: PNG16 (uint16, meters scaled × 1000), NPY (float32)
- Videos: MP4 uploads (temporary, deleted after processing)
- 3D results: PLY (point clouds, Gaussian splats), OBJ (meshes)
- Floor plans: SVG, DXF, JSON

**Key Methods** (`storage/object_store.py`):
- `upload_file()` / `upload_bytes()` → Returns object key
- `get_presigned_url(key, ttl=86400)` → Signed URL for download
- `cleanup_expired()` → Periodic deletion of TTL-expired objects

---

### 6. Billing Integration (Stripe)

**Architecture**: Event-driven via webhooks

**Flow**:
```
1. User requests /v1/billing/checkout → Stripe Checkout session URL
2. User completes payment on Stripe-hosted page
3. Stripe fires webhook: checkout.session.completed
4. API validates signature, updates Redis: apikey:{hash}.plan = "pro"
5. User can now make 50k calls/month instead of 100
```

**Webhook Events Handled**:
- `checkout.session.completed` - New subscription created
- `customer.subscription.updated` - Plan changed (upgrade/downgrade)
- `customer.subscription.deleted` - Cancellation
- `invoice.paid` - Successful recurring payment
- `invoice.payment_failed` - Declined card

**Plan Enforcement**:
- Redis field: `apikey:{hash}.plan` = "free" | "builder" | "pro" | "enterprise" | "admin"
- Rate limiter checks monthly_calls < monthly_limit
- Function-level auth checks: e.g., `/v1/admin/*` requires plan=admin

**Key Files**:
- `billing/stripe_billing.py` - StripeBilling class
- `api/v1/billing.py` - Endpoints (/checkout, /portal, /webhooks)

---

### 7. Middleware Stack

**Execution Order** (important for correctness):

```
Request →
  1. CORSMiddleware (validate origin)
  2. SecurityHeadersMiddleware (add OWASP headers)
  3. RequestTimeoutMiddleware (120s timeout on /depth, /measure, /pose)
  4. WebhookRateLimiterMiddleware (100 req/min per IP on /webhooks)
  5. RateLimiterMiddleware (monthly quota enforcement)
  6. MetricsMiddleware (Prometheus instrumentation)
  7. RequestTracingMiddleware (X-Request-ID injection)
  → Endpoint Handler
Response ←
```

**Why This Order?**
- CORS first: Reject invalid origins before any processing
- Security headers: Add headers to all responses (even errors)
- Timeout: Prevent long-running requests from blocking workers
- Rate limiting: Check quotas before expensive operations
- Metrics: Track all requests (even rate-limited ones)
- Tracing: Unique ID for request correlation across services

**Key Files**:
- `middleware/security_headers.py` - HSTS, CSP, X-Frame-Options, etc.
- `middleware/timeout.py` - asyncio.wait_for() wrapper
- `middleware/webhook_rate_limiter.py` - IP-based DoS protection
- `auth/rate_limiter.py` - Redis sliding window + monthly quota

---

## Data Flow

### Synchronous Request (Depth Estimation)

```
┌─────────┐    POST /v1/depth      ┌─────────────┐
│ Client  │ ────────────────────────→│ API Gateway │
└─────────┘    X-API-Key: sf_xxx    │ (FastAPI)   │
                                     └──────┬──────┘
                                            │
                                     1. Auth ✓
                                     2. Rate limit ✓
                                     3. Validate input
                                            │
                                     ┌──────▼────────┐
                                     │ DepthEngine   │
                                     │ (GPU inference│
                                     │  ~50ms)       │
                                     └──────┬────────┘
                                            │
                                     ┌──────▼────────┐
                                     │ MinIO Upload  │
                                     │ depth.png     │
                                     └──────┬────────┘
                                            │
                                     ┌──────▼────────┐
                                     │ JSON Response │
┌─────────┐                          │ - depth_url   │
│ Client  │ ←────────────────────────│ - metadata    │
└─────────┘    200 OK                │ - stats       │
                                     └───────────────┘
```

### Asynchronous Request (3D Reconstruction)

```
┌─────────┐    POST /v1/reconstruct  ┌─────────────┐
│ Client  │ ──────────────────────────→│ API Gateway │
└─────────┘    video=@file.mp4        └──────┬──────┘
                                              │
                                       1. Auth ✓
                                       2. Upload to MinIO
                                       3. Enqueue Celery task
                                              │
                                       ┌──────▼────────┐
┌─────────┐    202 Accepted            │ Return job_id │
│ Client  │ ←──────────────────────────│ immediately   │
└─────────┘    {job_id: "abc123"}     └───────────────┘
     │
     │ Poll GET /v1/jobs/abc123 every 5s
     │
     │                                 ┌───────────────┐
     └────────────────────────────────→│ Job Status    │
                                       │ PROCESSING... │
                                       └───────┬───────┘
                                               │
                                        ┌──────▼──────────┐
                                        │ Celery Worker   │
                                        │ 1. Download vid │
                                        │ 2. Extract keyframes
                                        │ 3. Depth est.   │
                                        │ 4. Pose est.    │
                                        │ 5. Reconstruct  │
                                        │ 6. Upload PLY   │
                                        └──────┬──────────┘
                                               │
┌─────────┐    GET /v1/jobs/abc123    ┌───────▼──────────┐
│ Client  │ ──────────────────────────→│ Job Complete!   │
└─────────┘                            │ {scene_url: ... }│
                                       └──────────────────┘
```

---

## Technology Stack

### Core Framework
- **FastAPI** 0.115+ - Modern async web framework
- **Uvicorn** 0.32+ - ASGI server with HTTP/1.1 and HTTP/2
- **Python** 3.11+ - Type hints, asyncio, performance improvements

### ML / Inference
- **PyTorch** 2.5+ - Deep learning framework
- **Transformers** 4.47+ - HuggingFace model loading
- **OpenCV** 4.10+ - Image processing (resize, color conversion)
- **NumPy** 1.26+ - Numerical operations

### Data Stores
- **Redis** 5.2+ - API keys, rate limiting, job queue, billing cache
- **MinIO** 7.2+ - S3-compatible object storage

### Async Processing
- **Celery** 5.4+ - Distributed task queue
- **Celery Beat** - Periodic task scheduler (cleanup)

### Observability
- **Prometheus** (via prometheus-client 0.21+) - Metrics
- **Structlog** 24.4+ - Structured logging
- **Sentry** 2.19+ - Error tracking (optional)

### Billing
- **Stripe** 11.0+ - Payments, subscriptions, webhooks

### Security
- **HMAC** (stdlib) - API key hashing
- **Pydantic** 2.10+ - Input validation
- **CORS** (FastAPI built-in) - Cross-origin policy
- **Tenacity** 9.0+ - Retry with exponential backoff

### Development
- **Pytest** 8.3+ - Testing (139 tests: 62 server + 77 SDK)
- **Ruff** 0.8+ - Linter & formatter
- **MyPy** 1.13+ - Static type checking

---

## Deployment Architecture

### Development (Local)

```
┌──────────────────────────────────────┐
│         Developer Machine            │
│                                      │
│  ┌────────────┐     ┌────────────┐  │
│  │ API Server │────→│   Redis    │  │
│  │ (Uvicorn)  │     │ (Docker)   │  │
│  └─────┬──────┘     └────────────┘  │
│        │                             │
│        ▼                             │
│  ┌────────────┐     ┌────────────┐  │
│  │ Celery Wrk │────→│   MinIO    │  │
│  │ (Python)   │     │ (Docker)   │  │
│  └────────────┘     └────────────┘  │
└──────────────────────────────────────┘
```

**Commands**:
```bash
# Start dependencies
docker-compose up -d redis minio

# Run API server
uvicorn spatialforge.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker
celery -A spatialforge.workers.celery_app worker --loglevel=info -Q gpu_heavy

# Run tests
pytest tests/ -v
cd sdk && pytest tests/ -v
```

---

### Production (Fly.io)

**Current Deployment** (as of 2026-02-08):
- App: `spatialforge-demo` (region: nrt - Tokyo)
- Mode: `DEMO_MODE=true` (Redis auto-stops on trial org)
- Redis: Standalone app `spatialforge-redis` (Redis 7 Alpine)
- MinIO: Not deployed (using base64 fallback)
- CI/CD: GitHub Actions auto-deploy on master push

**Architecture**:
```
                      Fly.io Platform
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌───────────────┐          ┌───────────────┐      │
│  │ API Instance  │          │ Redis (App)   │      │
│  │ (CPU-only)    │─────────→│ Standalone    │      │
│  │ - Uvicorn     │          │ Auto-stops    │      │
│  │ - Workers     │          │ after 5min    │      │
│  └───────┬───────┘          └───────────────┘      │
│          │                                          │
│          │ Proxied by Fly.io Edge                  │
│          ▼                                          │
│  ┌─────────────────────────────────┐               │
│  │   https://spatialforge-demo     │               │
│  │        .fly.dev                 │               │
│  └─────────────────────────────────┘               │
└──────────────────────────────────────────────────────┘
```

**Limitations** (Trial Org):
- Machines stop after 5 minutes idle
- No persistent volumes
- No GPU instances
- Need credit card to remove auto-stop

**Future Production Setup** (with credit card):
```
                      Fly.io Platform
┌──────────────────────────────────────────────────────┐
│                                                      │
│  ┌───────────────┐  ┌───────────────┐              │
│  │ API Instance  │  │ API Instance  │  (Autoscale)│
│  │ (2 CPUs)      │  │ (2 CPUs)      │              │
│  └───────┬───────┘  └───────┬───────┘              │
│          │                   │                      │
│          └──────────┬────────┘                      │
│                     │                               │
│          ┌──────────▼──────────┐                    │
│          │ Redis (Managed)     │                    │
│          │ Persistent, HA      │                    │
│          └─────────────────────┘                    │
│                                                      │
│  ┌───────────────┐                                  │
│  │ Celery Worker │                                  │
│  │ (4 CPUs, GPU) │                                  │
│  │ Queue: gpu_heavy                                │
│  └───────┬───────┘                                  │
│          │                                           │
│          ▼                                           │
│  ┌─────────────────────────────────┐                │
│  │   S3-compatible Storage         │                │
│  │   (Tigris or external)          │                │
│  └─────────────────────────────────┘                │
└──────────────────────────────────────────────────────┘
```

---

### Kubernetes (Optional)

**Manifests** (see `docs/kubernetes/`):
- `deployment.yaml` - API server (3 replicas, CPU-optimized)
- `deployment-worker.yaml` - Celery workers (2 replicas, GPU)
- `service.yaml` - LoadBalancer for API
- `ingress.yaml` - HTTPS termination, routing
- `configmap.yaml` - Non-secret config
- `secret.yaml` - Secrets (API_KEY_SECRET, etc.)
- `redis-statefulset.yaml` - Redis with persistent volume
- `minio-statefulset.yaml` - MinIO with persistent volume
- `hpa.yaml` - Horizontal Pod Autoscaler (target: 70% CPU)

**Deployment**:
```bash
kubectl apply -f docs/kubernetes/
kubectl get pods -n spatialforge
kubectl logs -f deployment/spatialforge-api
```

---

## Security Architecture

**See SECURITY_AUDIT.md for comprehensive analysis.**

### Authentication Flow

```
Client Request
    │
    ├─ X-API-Key: sf_abc123xyz...
    │
    ▼
┌──────────────────────────────────┐
│ RateLimiterMiddleware (auth.py) │
│                                  │
│ 1. Extract X-API-Key header      │
│ 2. Hash key: HMAC-SHA256(key)    │
│ 3. Redis GET apikey:{hash}       │
│ 4. Validate:                     │
│    - Key exists?                 │
│    - Enabled == 1?               │
│    - monthly_calls < limit?      │
│ 5. Increment monthly_calls       │
│ 6. Inject KeyInfo into state     │
└────────────┬─────────────────────┘
             │
             ▼
       ┌─────────────┐
       │  Endpoint   │
       │  Handler    │
       └─────────────┘
```

### Security Headers

**Added by SecurityHeadersMiddleware**:

```http
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
Content-Security-Policy: default-src 'self'; ...
X-API-Version: 0.1.0
```

**HSTS**: Only added for production domains (not localhost)
**CSP**: Permissive for `/docs` (needs inline styles), restrictive for `/v1/*`

### Secrets Management

**Never Committed**:
- `API_KEY_SECRET` - HMAC key for API key hashing
- `ADMIN_API_KEY` - Admin access key
- `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY` - Object storage credentials
- `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET` - Billing secrets
- `SENTRY_DSN` - Error tracking endpoint

**How Secrets are Loaded**:
1. Environment variables (12-factor app pattern)
2. `.env` file in development (Git-ignored)
3. Kubernetes Secrets in production
4. Fly.io Secrets (`flyctl secrets set KEY=value`)

**Validation**:
- Startup check: Refuses to start if secrets are default values (in production mode)
- Minimum length: API_KEY_SECRET must be ≥32 characters
- Warnings: Logs warnings for weak MinIO credentials

---

## Scalability & Performance

### Performance Targets

| Operation | P50 Latency | P95 Latency | P99 Latency |
|-----------|-------------|-------------|-------------|
| Depth estimation (small) | 30ms | 50ms | 80ms |
| Depth estimation (large) | 50ms | 100ms | 150ms |
| Distance measurement | 60ms | 120ms | 180ms |
| Camera pose (10 frames) | 200ms | 400ms | 600ms |
| 3D reconstruction (60s video) | 25s | 45s | 60s |

### Scalability Strategy

**Horizontal Scaling**:
- API servers: Stateless, can add unlimited replicas behind load balancer
- Celery workers: Independent GPU workers, scale based on queue depth
- Redis: Single-leader replication (read replicas for caching)
- MinIO: Distributed mode with erasure coding (4+ nodes)

**Vertical Scaling**:
- GPU memory: Larger models require more VRAM (small=2GB, large=8GB, giant=16GB)
- CPU cores: More workers per machine (uvicorn --workers N)
- Disk: Fast NVMe for model cache, video uploads

**Bottlenecks**:
1. **GPU memory**: Primary constraint for concurrent inference
   - Mitigation: Queue management, small model for low-latency tier
2. **Network bandwidth**: Large video uploads
   - Mitigation: Streaming uploads, compression
3. **Redis throughput**: 100k+ ops/sec possible with pipelining
   - Mitigation: Use Redis pipeline for multi-key operations

### Caching Strategy

**Model Cache** (inference/model_manager.py):
- Models loaded once per worker process, kept in memory
- Lazy loading: Only load on first use
- Unload on shutdown (free VRAM)

**Result Cache** (Future):
- Redis cache for duplicate depth estimation requests
- Key: SHA256(image bytes) → depth_map_url
- TTL: 1 hour
- Cache hit saves ~50ms GPU inference

**CDN** (Future):
- CloudFlare in front of MinIO presigned URLs
- Edge caching for frequently accessed depth maps

---

## Monitoring & Observability

### Prometheus Metrics

**Endpoint**: `GET /metrics`

**Metrics Collected**:

1. **Request Metrics**
   - `request_count{method, endpoint, status}` - Total requests
   - `request_latency{method, endpoint}` - Histogram (p50, p95, p99)

2. **Inference Metrics**
   - `inference_duration{model, endpoint}` - Histogram
   - `inference_count{model, endpoint}` - Counter

3. **Business Metrics**
   - `api_key_usage{plan}` - Calls per plan tier
   - `rate_limit_hits{plan}` - Quota exhaustion events
   - `error_count{endpoint, status}` - 4xx/5xx errors

4. **System Metrics**
   - `gpu_memory_used_bytes` - Current VRAM usage
   - `gpu_memory_total_bytes` - Total VRAM capacity
   - `active_jobs_count` - Celery queue depth
   - `file_upload_size_bytes` - Upload size distribution

**Grafana Dashboard**: `monitoring/grafana-dashboard.json` (12 panels)

### Structured Logging

**Format**: JSON (structlog)

**Fields**:
- `timestamp` - ISO 8601 with ms precision
- `level` - DEBUG, INFO, WARNING, ERROR, CRITICAL
- `logger` - Module name (e.g., spatialforge.api.v1.depth)
- `message` - Human-readable log message
- `request_id` - Unique ID for request correlation
- `duration_ms` - Request duration (on response log)
- `status_code` - HTTP status (on response log)
- `endpoint` - Route path
- `method` - HTTP method

**Example**:
```json
{
  "timestamp": "2026-02-08T12:34:56.789Z",
  "level": "INFO",
  "logger": "spatialforge.api.v1.depth",
  "message": "Depth estimation request",
  "request_id": "abc123",
  "endpoint": "/v1/depth",
  "method": "POST",
  "model": "large",
  "image_size": "1920x1080"
}
```

### Error Tracking (Sentry)

**When Enabled** (SENTRY_DSN configured):
- All unhandled exceptions captured
- Request breadcrumbs (API calls, Redis ops, MinIO uploads)
- Release tracking (spatialforge@0.1.0)
- Environment tagging (production, staging, development)
- Performance monitoring (10% of transactions sampled)

**Privacy**:
- `send_default_pii=False` - No user IDs or emails sent
- API keys redacted from error context
- Only stack traces and request metadata captured

---

## Future Roadmap

### Planned Enhancements

**Q1 2026**:
- [ ] Depth Anything V3 integration (10%+ accuracy improvement)
- [ ] Video temporal consistency (overlapping frames)
- [ ] TensorRT FP16 optimization (2-3× speedup)
- [ ] Batch processing API (multiple images in one request)

**Q2 2026**:
- [ ] Result caching (Redis + content-addressable storage)
- [ ] WebSocket API for real-time progress updates
- [ ] GraphQL API for flexible queries
- [ ] Multi-region deployment (US, EU, Asia)

**Q3 2026**:
- [ ] Alternative models (Marigold, UniDepth V2, Metric3D v2)
- [ ] Model A/B testing framework
- [ ] Custom model fine-tuning service
- [ ] Kubernetes Helm charts

**Q4 2026**:
- [ ] Real Gaussian Splatting training (gsplat library)
- [ ] Mesh generation (Poisson reconstruction)
- [ ] 3D viewer hosting (Three.js/PlayCanvas)
- [ ] SAM3 integration for 3D segmentation

### Research Directions

- **On-device inference**: Export models to ONNX/CoreML for mobile
- **Temporal models**: Video-native depth estimation (FlashDepth, oVDA)
- **Multimodal**: Depth + surface normals + semantic segmentation
- **Zero-shot**: Few-shot adaptation to new camera types

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [OWASP API Security Top 10](https://owasp.org/API-Security/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-08
**Maintainer**: SpatialForge Team
