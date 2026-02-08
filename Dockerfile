# ============================================================
# SpatialForge API — GPU-enabled Docker image
# ============================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /app

# Python dependencies (cached layer)
COPY pyproject.toml ./
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -e ".[dev]"

ENV PATH="/opt/venv/bin:$PATH"

# Application code
COPY . .
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /app/models /app/uploads /app/results

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
CMD ["uvicorn", "spatialforge.main:app", "--host", "0.0.0.0", "--port", "8000"]
