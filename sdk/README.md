# SpatialForge Python SDK

**Any Camera. Instant 3D. One API.**

[![PyPI](https://img.shields.io/pypi/v/spatialforge-client)](https://pypi.org/project/spatialforge-client/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Installation

```bash
pip install spatialforge-client
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

# 3D reconstruction (async)
job = client.reconstruct("walkthrough.mp4", quality="high")
scene = job.wait()  # Blocks until complete
print(f"Scene URL: {scene['scene_url']}")

# Floor plan generation (async)
plan = client.floorplan("room_tour.mp4")
result = plan.wait()
print(f"Floor area: {result['floor_area_m2']} m2")
```

## Custom API endpoint

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

## Links

- [Live API Docs](https://spatialforge-demo.fly.dev/docs)
- [Interactive Demo](https://maruyamakoju.github.io/spatialforge/demo.html)
- [GitHub](https://github.com/maruyamakoju/spatialforge)
