# SpatialForge Deployment Guide

**Version**: 0.1.0
**Date**: 2026-02-08

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Fly.io Deployment](#flyio-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Production Checklist](#production-checklist)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum** (Development):
- Python 3.11+
- 4GB RAM
- 10GB disk space
- CPU-only mode supported

**Recommended** (Production):
- Python 3.11+
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (for depth estimation)
- 50GB disk space (for models + results cache)
- Ubuntu 22.04 LTS or Docker

### Software Dependencies

```bash
# Python 3.11+
python --version  # Should be ≥3.11

# Redis 5.2+
redis-server --version

# (Optional) Docker + Docker Compose
docker --version
docker-compose --version

# (Optional) NVIDIA GPU drivers + CUDA 12.6+
nvidia-smi
```

---

## Environment Configuration

### 1. Clone Repository

```bash
git clone https://github.com/maruyamakoju/spatialforge.git
cd spatialforge
```

### 2. Create `.env` File

Copy the example and customize:

```bash
cp .env.example .env
```

**`.env` Template**:

```bash
# ========================================
# SpatialForge Configuration
# ========================================

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1
DEBUG=false
ALLOWED_ORIGINS=["http://localhost:3000","https://your-domain.com"]

# Auth (CRITICAL: Change these in production!)
API_KEY_SECRET=CHANGE_ME_TO_A_RANDOM_SECRET_AT_LEAST_32_CHARS
ADMIN_API_KEY=sf_admin_CHANGE_ME
DEMO_MODE=false

# Redis
REDIS_URL=redis://localhost:6379/0

# MinIO / S3
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=spatialforge
MINIO_SECURE=false

# Models
MODEL_DIR=./models
DEFAULT_DEPTH_MODEL=large
DEPTH_BACKEND=hf  # "hf" or "da3" (requires optional DA3 dependency)
RECONSTRUCT_BACKEND=legacy  # "legacy" (default), "tsdf" (requires open3d), or "da3" (placeholder)
DEVICE=cuda  # or "cpu"
TORCH_DTYPE=float16  # or "float32"
RESEARCH_MODE=false  # DANGER: enables CC-BY-NC models

# Stripe (optional)
STRIPE_SECRET_KEY=
STRIPE_WEBHOOK_SECRET=

# Sentry (optional)
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1

# Rate Limiting
RATE_LIMIT_FREE=100
RATE_LIMIT_BUILDER=5000
RATE_LIMIT_PRO=50000

# Processing
MAX_IMAGE_SIZE=4096
MAX_VIDEO_DURATION_S=120
RESULT_TTL_HOURS=24
```

### 3. Generate Secure Secrets

**⚠️ CRITICAL FOR PRODUCTION**:

```bash
# Generate API_KEY_SECRET (48-byte base64)
openssl rand -base64 48

# Generate ADMIN_API_KEY
echo "sf_$(openssl rand -hex 32)"

# Generate MinIO credentials
openssl rand -hex 20  # access_key
openssl rand -hex 40  # secret_key
```

Update `.env` with these values.

---

## Local Development

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install spatialforge + dev dependencies
pip install -e ".[dev]"

# Optional: TSDF backend dependencies for reconstruct
pip install -e ".[tsdf]"
```

### 2. Start Redis

**Option A: Docker**
```bash
docker run -d --name redis -p 6379:6379 redis:7-alpine
```

**Option B: Local Install**
```bash
sudo systemctl start redis
```

### 3. Start MinIO (Optional)

```bash
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create bucket
docker exec minio mc mb /data/spatialforge
```

### 4. Start API Server

```bash
# Development mode (auto-reload)
uvicorn spatialforge.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn spatialforge.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Start Celery Worker

```bash
# In a separate terminal
celery -A spatialforge.workers.celery_app worker \
  --loglevel=info \
  --queue=gpu_heavy,default \
  --concurrency=1  # Limit to 1 for GPU tasks
```

### 6. Start Celery Beat (Periodic Tasks)

```bash
# In another terminal
celery -A spatialforge.workers.celery_app beat --loglevel=info
```

### 7. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Response should be:
# {"status":"ok","version":"0.1.0","gpu_available":true,"models_loaded":[]}
```

---

## Docker Deployment

### 1. CPU-Only (Development/CI)

```bash
# Build
docker build -f Dockerfile.cpu -t spatialforge:cpu .

# Run
docker run -p 8000:8000 \
  --env-file .env \
  spatialforge:cpu
```

### 2. GPU (Production)

```bash
# Build
docker build -t spatialforge:gpu .

# Run with GPU
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  spatialforge:gpu
```

### 3. Full Stack (Docker Compose)

**`docker-compose.yml`**:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - redis
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  worker:
    build: .
    command: celery -A spatialforge.workers.celery_app worker --loglevel=info -Q gpu_heavy
    env_file: .env
    depends_on:
      - redis
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  beat:
    build: .
    command: celery -A spatialforge.workers.celery_app beat --loglevel=info
    env_file: .env
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/spatialforge.json

volumes:
  redis_data:
  minio_data:
  prometheus_data:
  grafana_data:
```

**Start All Services**:

```bash
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop all
docker-compose down
```

---

## Fly.io Deployment

### 1. Install Fly CLI

```bash
# macOS/Linux
curl -L https://fly.io/install.sh | sh

# Windows (PowerShell)
pwsh -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Verify
flyctl version
```

### 2. Login & Create App

```bash
# Login
flyctl auth login

# Create app (interactive)
flyctl launch --copy-config --yes

# This creates fly.toml from existing config
```

### 3. Deploy Redis

```bash
# Create Redis app
flyctl redis create spatialforge-redis \
  --region nrt \
  --plan free

# Note the connection string, update REDIS_URL in secrets
```

### 4. Set Secrets

```bash
# Critical secrets
flyctl secrets set \
  API_KEY_SECRET=$(openssl rand -base64 48) \
  ADMIN_API_KEY=sf_$(openssl rand -hex 32) \
  MINIO_ACCESS_KEY=$(openssl rand -hex 20) \
  MINIO_SECRET_KEY=$(openssl rand -hex 40)

# Optional: Stripe
flyctl secrets set \
  STRIPE_SECRET_KEY=sk_live_... \
  STRIPE_WEBHOOK_SECRET=whsec_...

# Optional: Sentry
flyctl secrets set SENTRY_DSN=https://...@sentry.io/...
```

### 5. Deploy

```bash
# Deploy to Fly.io
flyctl deploy

# Watch logs
flyctl logs

# Check status
flyctl status

# Open in browser
flyctl open
```

### 6. Scale (Requires Credit Card)

```bash
# Add more API instances
flyctl scale count 2

# Increase VM resources
flyctl scale vm shared-cpu-2x --memory 2048

# Stop auto-stop (requires credit card)
flyctl scale count 1 --region nrt
```

### 7. Custom Domain

```bash
# Add custom domain
flyctl certs add your-domain.com

# Configure DNS (A/AAAA records)
flyctl ips list
```

---

## Kubernetes Deployment

### 1. Prerequisites

- Kubernetes cluster (GKE, EKS, AKS, or local with minikube)
- kubectl configured
- Helm 3+ (optional)

### 2. Create Namespace

```bash
kubectl create namespace spatialforge
kubectl config set-context --current --namespace=spatialforge
```

### 3. Create Secrets

```bash
# Create secret from .env file
kubectl create secret generic spatialforge-secrets \
  --from-literal=API_KEY_SECRET=$(openssl rand -base64 48) \
  --from-literal=ADMIN_API_KEY=sf_$(openssl rand -hex 32) \
  --from-literal=MINIO_ACCESS_KEY=$(openssl rand -hex 20) \
  --from-literal=MINIO_SECRET_KEY=$(openssl rand -hex 40) \
  --namespace=spatialforge
```

### 4. Deploy Redis

```bash
# StatefulSet with persistent volume
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis
spec:
  serviceName: redis
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  ports:
  - port: 6379
  selector:
    app: redis
EOF
```

### 5. Deploy API

```bash
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spatialforge-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spatialforge-api
  template:
    metadata:
      labels:
        app: spatialforge-api
    spec:
      containers:
      - name: api
        image: your-registry/spatialforge:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: redis://redis:6379/0
        - name: API_KEY_SECRET
          valueFrom:
            secretKeyRef:
              name: spatialforge-secrets
              key: API_KEY_SECRET
        - name: ADMIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: spatialforge-secrets
              key: ADMIN_API_KEY
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: spatialforge-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: spatialforge-api
EOF
```

### 6. Deploy Celery Workers

```bash
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spatialforge-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: spatialforge-worker
  template:
    metadata:
      labels:
        app: spatialforge-worker
    spec:
      containers:
      - name: worker
        image: your-registry/spatialforge:latest
        command: ["celery"]
        args: ["-A", "spatialforge.workers.celery_app", "worker", "--loglevel=info", "-Q", "gpu_heavy"]
        env:
        - name: REDIS_URL
          value: redis://redis:6379/0
        - name: API_KEY_SECRET
          valueFrom:
            secretKeyRef:
              name: spatialforge-secrets
              key: API_KEY_SECRET
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
EOF
```

### 7. Horizontal Pod Autoscaling

```bash
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: spatialforge-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: spatialforge-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

### 8. Verify Deployment

```bash
# Check pods
kubectl get pods

# Check services
kubectl get svc

# View logs
kubectl logs -f deployment/spatialforge-api

# Test API
kubectl port-forward svc/spatialforge-api 8000:80
curl http://localhost:8000/health
```

---

## Production Checklist

### Pre-Deployment

- [ ] Generate and set all secrets (API_KEY_SECRET, ADMIN_API_KEY, MinIO credentials)
- [ ] Configure production-grade Redis (persistent, AOF enabled)
- [ ] Set up S3-compatible storage with backup policy
- [ ] Configure Stripe (if using billing)
- [ ] Set up Sentry error tracking
- [ ] Review ALLOWED_ORIGINS (only production domains)
- [ ] Set DEMO_MODE=false
- [ ] Set DEBUG=false
- [ ] Configure HTTPS/TLS certificates
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Configure log aggregation (ELK, Datadog, etc.)

### Security

- [ ] API_KEY_SECRET is NOT the default value
- [ ] ADMIN_API_KEY is NOT the default value
- [ ] MinIO credentials are NOT minioadmin/minioadmin
- [ ] Firewall rules restrict Redis/MinIO access
- [ ] HTTPS enforced (no HTTP)
- [ ] HSTS enabled
- [ ] Rate limiting configured
- [ ] Dependency vulnerability scan (`pip-audit`)
- [ ] security.txt file accessible

### Performance

- [ ] GPU available and detected (`nvidia-smi`)
- [ ] Model cache directory exists and writable
- [ ] Redis max memory policy configured (allkeys-lru)
- [ ] Celery workers = number of GPUs
- [ ] Prometheus metrics endpoint accessible
- [ ] Load testing completed (target: 100 req/s)

### Observability

- [ ] Prometheus scraping `/metrics`
- [ ] Grafana dashboard imported
- [ ] Alerts configured (high error rate, GPU memory, disk space)
- [ ] Log aggregation working
- [ ] Sentry capturing errors
- [ ] Health check endpoint monitored

### Backup & Recovery

- [ ] Redis AOF or RDB snapshots enabled
- [ ] MinIO data backed up (daily)
- [ ] Configuration backed up (.env, secrets)
- [ ] Disaster recovery plan documented
- [ ] Restore procedure tested

---

## Troubleshooting

### Issue: "API_KEY_SECRET is using default value"

**Error**:
```
RuntimeError: CRITICAL SECURITY ERROR: API_KEY_SECRET is using the default value.
```

**Solution**:
```bash
# Generate a new secret
export API_KEY_SECRET=$(openssl rand -base64 48)

# Or in .env file:
API_KEY_SECRET=YOUR_GENERATED_SECRET_HERE
```

---

### Issue: "Redis not available — auth disabled"

**Error**: Logs show "Redis not available"

**Solutions**:

1. Check Redis is running:
```bash
redis-cli ping  # Should return PONG
```

2. Check `REDIS_URL` in `.env`:
```bash
# Correct format:
REDIS_URL=redis://localhost:6379/0
```

3. Test connection:
```bash
python -c "import redis; r = redis.from_url('redis://localhost:6379/0'); print(r.ping())"
```

---

### Issue: "CUDA out of memory"

**Error**: GPU OOM during inference

**Solutions**:

1. Use smaller model:
```bash
DEFAULT_DEPTH_MODEL=small  # Instead of "large"
```

2. Reduce Celery concurrency:
```bash
celery -A spatialforge.workers.celery_app worker --concurrency=1
```

3. Clear GPU cache:
```python
import torch
torch.cuda.empty_cache()
```

---

### Issue: "Models not loading"

**Error**: "Failed to load model"

**Solutions**:

1. Check model directory exists:
```bash
mkdir -p ./models
```

2. Check HuggingFace access:
```bash
python -c "from transformers import pipeline; pipe = pipeline('depth-estimation', model='depth-anything/DA2-SMALL')"
```

3. Set HuggingFace cache:
```bash
export HF_HOME=./models
```

---

### Issue: Slow Inference (<10 FPS)

**Possible Causes**:
- Running on CPU instead of GPU
- Large model (giant) with insufficient VRAM
- Blocking I/O (synchronous operations)

**Solutions**:

1. Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

2. Check model size:
```bash
# In .env:
DEFAULT_DEPTH_MODEL=base  # Balanced speed/quality
```

3. Use TensorRT (future):
```bash
pip install tensorrt
# Enable TensorRT optimization
```

---

### Issue: High Memory Usage

**Symptoms**: API server OOM, slow response

**Solutions**:

1. Limit Uvicorn workers:
```bash
uvicorn spatialforge.main:app --workers 2  # Not 8
```

2. Enable Redis maxmemory:
```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

3. Clear model cache:
```python
# In code:
model_manager.unload_all()
```

---

### Issue: Webhook Signature Invalid

**Error**: "Invalid signature" on Stripe webhook

**Solutions**:

1. Get webhook secret from Stripe Dashboard:
```
Developers → Webhooks → [Your endpoint] → Signing secret
```

2. Update `.env`:
```bash
STRIPE_WEBHOOK_SECRET=whsec_...
```

3. Test locally with Stripe CLI:
```bash
stripe listen --forward-to localhost:8000/v1/billing/webhooks
stripe trigger checkout.session.completed
```

---

### Issue: 429 Rate Limit Exceeded

**Error**: "Rate limit exceeded"

**Solutions**:

1. Check current usage:
```bash
curl -H "X-API-Key: sf_your_key" \
  http://localhost:8000/v1/billing/usage
```

2. Upgrade plan (if on free tier):
```bash
curl -X POST -H "X-API-Key: sf_your_key" \
  http://localhost:8000/v1/billing/checkout \
  -d '{"plan":"builder","success_url":"...","cancel_url":"..."}'
```

3. Wait for monthly reset (1st of month)

---

## Support

**Documentation**: https://github.com/maruyamakoju/spatialforge
**Issues**: https://github.com/maruyamakoju/spatialforge/issues
**Security**: security@spatialforge.example.com

---

**Last Updated**: 2026-02-08
**Document Version**: 1.0
