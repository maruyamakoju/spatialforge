# SpatialForge

**Any Camera. Instant 3D. One API.**

Spatial intelligence API platform powered by Depth Anything 3. Convert any camera image or video into metric depth maps, camera poses, 3D reconstructions, real-world measurements, floor plans, and 3D segmentation.

## Quick Start

```bash
pip install -e ".[dev]"
spatialforge  # starts on http://localhost:8000
```

## Docker

```bash
cp .env.example .env
docker compose up -d
```

## API Endpoints

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/v1/depth` | POST | Monocular depth estimation | Production |
| `/v1/measure` | POST | Real-world distance measurement | Production |
| `/v1/pose` | POST | Camera pose estimation | Production |
| `/v1/reconstruct` | POST | 3D reconstruction (async) | Production |
| `/v1/floorplan` | POST | Floor plan generation (async) | Beta |
| `/v1/segment-3d` | POST | 3D segmentation (async) | Beta |

## License

MIT
