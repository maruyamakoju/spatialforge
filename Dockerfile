# ============================================================
# SpatialForge API — GPU-enabled Docker image
# ============================================================
# Build:  docker build -t spatialforge:latest .
# Run:    docker run --gpus all -p 8000:8000 --env-file .env spatialforge:latest
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies + Python 3.12 via deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# Install dependencies (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml ./
RUN pip install --no-cache-dir ".[dev]" 2>/dev/null || \
    pip install --no-cache-dir \
    "fastapi>=0.115.0" "uvicorn[standard]>=0.32.0" "python-multipart>=0.0.18" \
    "pydantic>=2.10.0" "pydantic-settings>=2.7.0" \
    "redis>=5.2.0" "celery[redis]>=5.4.0" "minio>=7.2.0" \
    "torch>=2.5.0" "torchvision>=0.20.0" "numpy>=1.26.0" \
    "Pillow>=11.0.0" "opencv-python-headless>=4.10.0" \
    "transformers>=4.47.0" "safetensors>=0.4.0" \
    "httpx>=0.28.0" "prometheus-client>=0.21.0" "structlog>=24.4.0" \
    "pytest>=8.3.0" "pytest-asyncio>=0.24.0" "httpx>=0.28.0"

# Copy application code and install package
COPY . .
RUN pip install --no-cache-dir -e .

# Create non-root user and directories
RUN groupadd -r spatialforge && useradd -r -g spatialforge spatialforge && \
    mkdir -p /app/models /app/uploads /app/results && \
    chown -R spatialforge:spatialforge /app

EXPOSE 8000

USER spatialforge

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
CMD ["uvicorn", "spatialforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
