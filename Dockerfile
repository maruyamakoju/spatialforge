# ============================================================
# SpatialForge API — GPU-enabled Docker image (Multi-stage)
# ============================================================
# Build:  docker build -t spatialforge:latest .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env spatialforge:latest
# ============================================================

# ============================================================
# Stage 1: Builder — Compile dependencies and build wheels
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gcc g++ make \
    git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Create venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml README.md ./

# Install dependencies (this layer is cached unless pyproject.toml changes)
# Build all wheels first to cache compilation artifacts
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "python-multipart>=0.0.18" \
    "pydantic>=2.10.0" \
    "pydantic-settings>=2.7.0" \
    "redis>=5.2.0" \
    "celery[redis]>=5.4.0" \
    "minio>=7.2.0" \
    "torch>=2.5.0" \
    "torchvision>=0.20.0" \
    "numpy>=1.26.0" \
    "Pillow>=11.0.0" \
    "opencv-python-headless>=4.10.0" \
    "transformers>=4.47.0" \
    "safetensors>=0.4.0" \
    "httpx>=0.28.0" \
    "prometheus-client>=0.21.0" \
    "structlog>=24.4.0" \
    "tenacity>=9.0.0" \
    "sentry-sdk[fastapi]>=2.19.0" \
    "stripe>=11.0.0"

# Copy application code and install package
COPY spatialforge ./spatialforge
COPY static ./static
RUN pip install --no-cache-dir -e .

# ============================================================
# Stage 2: Runtime — Minimal production image
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Copy venv from builder (includes all dependencies)
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --from=builder /build/spatialforge ./spatialforge
COPY --from=builder /build/static ./static
COPY pyproject.toml README.md ./

# Create non-root user and directories
RUN groupadd -r spatialforge && \
    useradd -r -g spatialforge spatialforge && \
    mkdir -p /app/models /app/uploads /app/results && \
    chown -R spatialforge:spatialforge /app

EXPOSE 8000

USER spatialforge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
CMD ["uvicorn", "spatialforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
