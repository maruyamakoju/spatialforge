# SpatialForge Python SDK

**Any Camera. Instant 3D. One API.**

[![PyPI](https://img.shields.io/pypi/v/spatialforge-client)](https://pypi.org/project/spatialforge-client/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Python SDK for [SpatialForge](https://github.com/maruyamakoju/spatialforge) — spatial intelligence API for depth estimation, measurement, 3D reconstruction, and more.

## Installation

```bash
pip install spatialforge-client

# With CLI support
pip install spatialforge-client[cli]
```

## Quick Start

```python
import spatialforge_client as sf

client = sf.Client(api_key="sf_your_key")

# Depth estimation
result = client.depth("photo.jpg")
print(f"Depth: {result.min_depth_m:.2f}m - {result.max_depth_m:.2f}m")
result.save_depth_map("depth.png")

# Distance measurement
measure = client.measure("room.jpg", point1=(100, 200), point2=(500, 200))
print(f"Distance: {measure.distance_cm:.1f} cm")

# Camera poses from video
poses = client.pose(video="walkthrough.mp4")
for p in poses.camera_poses:
    print(f"Frame {p.frame_index}: T={p.translation}")

# 3D reconstruction (async)
job = client.reconstruct("walkthrough.mp4", quality="high")
scene = job.wait()  # Blocks until complete
print(f"Scene URL: {scene['scene_url']}")
```

## Async Client

```python
import spatialforge_client as sf

async with sf.AsyncClient(api_key="sf_your_key") as client:
    # All methods are async
    result = await client.depth("photo.jpg")
    print(result.min_depth_m)

    # Async job polling
    job = await client.reconstruct("video.mp4")
    scene = await job.async_wait()
```

## Async Job Status Contract

For async endpoints (`reconstruct`, `floorplan`, `segment_3d`), responses include:

- `state` (recommended): stable lifecycle enum (`pending`, `processing`, `complete`, `failed`)
- `step`: optional processing phase when `state == "processing"`
- `status` (legacy): backward-compatible string, sometimes `processing:<step>`

The SDK's `wait()` / `async_wait()` methods prefer `state` when present and fall back to legacy `status`.

## CLI

```bash
export SPATIALFORGE_API_KEY=sf_your_key

# Depth estimation
spatialforge depth photo.jpg --model large --output depth.png

# Measure distance
spatialforge measure room.jpg --p1 100,200 --p2 500,200

# 3D reconstruction
spatialforge reconstruct walkthrough.mp4 --quality high

# All commands support --json for machine-readable output
spatialforge depth photo.jpg --json
```

## Custom API Endpoint

```python
# Point to a local or self-hosted instance
client = sf.Client(
    api_key="sf_your_key",
    base_url="http://localhost:8000",
)
```

## Endpoints

| Method | Description |
|--------|-------------|
| `client.depth(image)` | Monocular depth estimation |
| `client.measure(image, p1, p2)` | Real-world distance measurement |
| `client.pose(video=..., images=...)` | Camera pose estimation |
| `client.reconstruct(video)` | 3D reconstruction (async) |
| `client.floorplan(video)` | Floor plan generation (async) |
| `client.segment_3d(video, prompt)` | 3D segmentation (async) |

## Error Handling

```python
from spatialforge_client import SpatialForgeError

try:
    result = client.depth("photo.jpg")
except SpatialForgeError as e:
    print(f"HTTP {e.status_code}: {e.detail}")
```

## Links

- [Live API Docs](https://spatialforge-demo.fly.dev/docs)
- [Interactive Demo](https://maruyamakoju.github.io/spatialforge/demo.html)
- [Documentation](https://maruyamakoju.github.io/spatialforge/docs.html)
- [GitHub](https://github.com/maruyamakoju/spatialforge)
