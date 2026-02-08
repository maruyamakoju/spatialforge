# Docker Multi-Stage Build Optimization

## Overview

All SpatialForge Docker images have been optimized using multi-stage builds to reduce final image size, improve build caching, and separate build-time from runtime dependencies.

## Optimization Strategy

### 1. **Two-Stage Architecture**

Each Dockerfile now uses a two-stage build:

- **Stage 1 (Builder)**: Compiles dependencies, builds wheels, installs build tools
- **Stage 2 (Runtime)**: Copies only necessary artifacts, minimal runtime dependencies

### 2. **Separation of Concerns**

**Builder Stage Includes:**
- Compilers (gcc, g++, make)
- Python development headers
- Build tools (pip, setuptools, wheel)
- Git (for pip installing from repos)

**Runtime Stage Includes:**
- Python interpreter only (no -dev packages)
- Shared libraries needed at runtime
- Compiled Python packages (wheels) from builder
- Application code

### 3. **Layer Caching Optimization**

Dependencies are installed before copying application code:
```dockerfile
# Dependencies (cached layer — only re-runs when pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir <dependencies>

# Application code (changes frequently, doesn't invalidate dependency cache)
COPY spatialforge ./spatialforge
```

## Optimized Dockerfiles

### 1. **Dockerfile** (Main API Server)

**Base Image:** `nvidia/cuda:12.4.1-runtime-ubuntu22.04`

**Builder Stage:**
- Installs gcc, g++, make, git
- Compiles PyTorch, transformers, and other ML packages
- ~4-5 GB intermediate layer

**Runtime Stage:**
- Only runtime libraries (libgl1, libglib2.0-0, etc.)
- No build tools
- **Expected Size:** 2.5-3 GB (vs 3.5-4 GB before)

### 2. **Dockerfile.worker** (Celery GPU Worker)

**Base Image:** `nvidia/cuda:12.4.1-runtime-ubuntu22.04`

**Optimizations:**
- Shared base image with main Dockerfile (Docker reuses layers)
- Identical builder stage for dependency compilation
- Runtime stage optimized for worker-only operation (no uvicorn)

**Expected Size:** 2.5-3 GB (shared layers with main image)

### 3. **Dockerfile.cpu** (CPU-Only Variant)

**Base Image:** `python:3.12-slim`

**Optimizations:**
- Uses CPU-only PyTorch wheels (`--index-url https://download.pytorch.org/whl/cpu`)
- **PyTorch CPU wheels are ~1.5 GB smaller than CUDA wheels**
- No CUDA runtime dependencies
- Minimal base image (Debian slim)

**Expected Size:** 1.2-1.5 GB (vs 2-2.5 GB before)

## Build Performance Improvements

### 1. **Faster Rebuilds**

When only application code changes (most common scenario):
- **Before:** Full dependency reinstall (~5-10 minutes)
- **After:** Only app code copy (~10-30 seconds)

### 2. **Better CI/CD**

GitHub Actions and other CI systems benefit from:
- Fewer layers to upload/download
- Better layer caching (dependencies cached separately)
- Parallel builds possible (builder stage can be cached)

### 3. **Reduced Network Transfer**

Pushing/pulling images from registries:
- **Before:** 3.5-4 GB per image
- **After:** 2.5-3 GB per image (~30% reduction)

## Security Improvements

### 1. **Smaller Attack Surface**

Runtime images no longer include:
- Compilers (gcc, g++, make)
- Build tools (git, wget, curl during build)
- Development headers
- Package manager (apt-get removed after final install)

### 2. **Non-Root User**

All images run as `spatialforge` user (not root):
```dockerfile
USER spatialforge
```

### 3. **Minimal Base Images**

Only essential runtime dependencies installed:
- No text editors
- No shells beyond /bin/sh
- No debugging tools in production

## Build Commands

### Standard Build (GPU)
```bash
docker build -t spatialforge:latest .
```

### CPU-Only Build
```bash
docker build -f Dockerfile.cpu -t spatialforge:cpu .
```

### Worker Build
```bash
docker build -f Dockerfile.worker -t spatialforge:worker .
```

### Build with Cache from Registry
```bash
docker build --cache-from spatialforge:latest -t spatialforge:latest .
```

## Expected Size Comparison

| Image | Before | After | Reduction |
|-------|--------|-------|-----------|
| spatialforge:latest (GPU) | 3.5-4 GB | 2.5-3 GB | ~30% |
| spatialforge:worker (GPU) | 3.5-4 GB | 2.5-3 GB | ~30% |
| spatialforge:cpu | 2.5-3 GB | 1.2-1.5 GB | ~50% |

## Verification

### Check Image Size
```bash
docker images spatialforge
```

### Check Image Layers
```bash
docker history spatialforge:latest
```

### Verify Build Cache
```bash
# First build (no cache)
docker build --no-cache -t spatialforge:test .

# Change app code only
echo "# comment" >> spatialforge/main.py

# Rebuild (should use cached dependency layers)
docker build -t spatialforge:test .
```

Expected: Dependency layers show "Using cache", only app layers rebuild.

## Best Practices Applied

1. **Multi-stage builds** separate build and runtime environments
2. **Explicit COPY** of only necessary files (no `COPY . .` in runtime)
3. **Layer ordering** optimized for caching (least changing → most changing)
4. **No-cache-dir** pip flag to avoid storing pip cache in layers
5. **Apt cleanup** (`rm -rf /var/lib/apt/lists/*`) to reduce layer size
6. **Combining RUN commands** to minimize layer count
7. **Non-root user** for security
8. **Healthchecks** for container orchestration

## Trade-offs

### Pros
- ✅ Smaller final images (~30-50% reduction)
- ✅ Faster rebuilds when only app code changes
- ✅ Better security (no build tools in production)
- ✅ Improved caching in CI/CD

### Cons
- ❌ Slightly longer initial build time (two stages)
- ❌ More complex Dockerfile (harder to debug build issues)
- ❌ Requires understanding of multi-stage builds for maintenance

**Verdict:** The benefits far outweigh the costs for production deployments.

## Further Optimization Opportunities

### 1. **BuildKit** (Experimental)
```bash
DOCKER_BUILDKIT=1 docker build -t spatialforge:latest .
```
- Parallel stage execution
- Advanced caching
- Buildx for multi-platform builds

### 2. **Distroless Base Images** (Future)
Replace `ubuntu22.04` / `python:3.12-slim` with Google's distroless:
```dockerfile
FROM gcr.io/distroless/python3-debian12
```
- Even smaller (~100 MB base)
- No shell (maximum security)
- Harder to debug

### 3. **Pre-built Wheels** (Advanced)
Host compiled wheels on private PyPI:
```dockerfile
RUN pip install --index-url https://pypi.internal/simple spatialforge
```
- Near-instant dependency install
- Consistent builds across environments
- Requires infrastructure

## References

- [Docker Multi-stage Builds Documentation](https://docs.docker.com/build/building/multi-stage/)
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/dev-best-practices/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Python Docker Best Practices](https://docs.python.org/3/using/docker.html)

## Changelog

- **2026-02-08**: Initial multi-stage optimization (Task #96)
  - Converted Dockerfile, Dockerfile.worker, Dockerfile.cpu to multi-stage
  - Expected 30-50% size reduction
  - Improved build caching and security
