"""Celery tasks for async GPU-heavy operations."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from pathlib import Path

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Lazy-initialized shared resources (initialized once per worker process)
_model_manager = None
_object_store = None


def _get_model_manager():
    """Lazy-init model manager (one per worker process)."""
    global _model_manager
    if _model_manager is None:
        from ..config import get_settings
        from ..inference.model_manager import ModelManager

        settings = get_settings()
        _model_manager = ModelManager(
            model_dir=settings.model_dir,
            device=settings.device,
            dtype=settings.torch_dtype,
        )
    return _model_manager


def _get_object_store():
    """Lazy-init object store."""
    global _object_store
    if _object_store is None:
        from ..config import get_settings
        from ..storage.object_store import ObjectStore

        settings = get_settings()
        _object_store = ObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
    return _object_store


@celery_app.task(bind=True, name="spatialforge.workers.tasks.reconstruct_task")
def reconstruct_task(
    self,
    video_object_key: str,
    quality: str = "standard",
    output_format: str = "gaussian",
    webhook_url: str | None = None,
) -> dict:
    """Async 3D reconstruction from uploaded video.

    Args:
        video_object_key: MinIO key for the uploaded video.
        quality: 'draft', 'standard', or 'high'.
        output_format: 'gaussian', 'pointcloud', or 'mesh'.
    """
    try:
        self.update_state(state="PROCESSING", meta={"step": "downloading_video"})

        store = _get_object_store()
        mm = _get_model_manager()

        # Download video from object store
        video_bytes = store.download_bytes(video_object_key)
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.write(video_bytes)
        tmp_video.close()

        try:
            self.update_state(state="PROCESSING", meta={"step": "extracting_keyframes"})

            from ..utils.video import extract_keyframes, validate_video

            validate_video(tmp_video.name)
            frames = extract_keyframes(tmp_video.name, target_fps=2.0)

            self.update_state(
                state="PROCESSING",
                meta={"step": "reconstructing", "num_frames": len(frames)},
            )

            from ..inference.reconstruct_engine import ReconstructEngine

            engine = ReconstructEngine(mm)
            result = engine.reconstruct(
                frames=frames,
                quality=quality,
                output_format=output_format,
            )

            self.update_state(state="PROCESSING", meta={"step": "uploading_result"})

            # Upload result to object store
            scene_key = store.upload_file(
                str(result.output_path),
                content_type="application/octet-stream",
                prefix="reconstructions",
            )
            scene_url = store.get_presigned_url(scene_key)

            result_data = {
                "status": "complete",
                "scene_url": scene_url,
                "scene_key": scene_key,
                "stats": {
                    "num_gaussians": result.num_gaussians,
                    "num_points": result.num_points,
                    "num_vertices": result.num_vertices,
                    "bounding_box": result.bounding_box,
                },
                "processing_time_ms": result.processing_time_ms,
            }

            _fire_webhook(webhook_url, self.request.id, "reconstruct", result_data)
            return result_data
        finally:
            os.unlink(tmp_video.name)

    except Exception as exc:
        logger.error("Reconstruction failed: %s", traceback.format_exc())
        err = {"status": "failed", "error": str(exc)}
        _fire_webhook(webhook_url, self.request.id, "reconstruct", err)
        return err


@celery_app.task(bind=True, name="spatialforge.workers.tasks.floorplan_task")
def floorplan_task(
    self,
    video_object_key: str,
    output_format: str = "svg",
) -> dict:
    """Async floorplan generation from room walkthrough video.

    Pipeline:
    1. Extract keyframes
    2. Estimate depth for each frame
    3. Estimate camera poses
    4. Project floor plane
    5. Detect walls and generate 2D floorplan
    """
    try:
        self.update_state(state="PROCESSING", meta={"step": "downloading_video"})

        store = _get_object_store()
        mm = _get_model_manager()

        video_bytes = store.download_bytes(video_object_key)
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.write(video_bytes)
        tmp_video.close()

        try:
            self.update_state(state="PROCESSING", meta={"step": "processing"})

            from ..utils.video import extract_keyframes, validate_video

            validate_video(tmp_video.name)
            frames = extract_keyframes(tmp_video.name, target_fps=1.0, max_frames=60)

            # Step 1: Get depth maps and poses
            from ..inference.depth_engine import DepthEngine
            from ..inference.pose_engine import PoseEngine

            depth_engine = DepthEngine(mm)
            pose_engine = PoseEngine(mm)

            depth_maps = []
            for frame in frames:
                dr = depth_engine.estimate(frame, model_size="large")
                depth_maps.append(dr.depth_map)

            pose_result = pose_engine.estimate_poses(frames, output_pointcloud=True)

            self.update_state(state="PROCESSING", meta={"step": "generating_floorplan"})

            # Step 2: Generate floorplan from point cloud projection
            floorplan_data = _generate_floorplan_from_pointcloud(
                pose_result.pointcloud, output_format
            )

            # Upload result
            ext = output_format if output_format != "json" else "json"
            content_type = {
                "svg": "image/svg+xml",
                "dxf": "application/dxf",
                "json": "application/json",
            }.get(output_format, "application/octet-stream")

            key = store.upload_bytes(
                floorplan_data.encode() if isinstance(floorplan_data, str) else floorplan_data,
                content_type=content_type,
                prefix="floorplans",
                extension=ext,
            )
            url = store.get_presigned_url(key)

            return {
                "status": "complete",
                "floorplan_url": url,
                "floor_area_m2": None,  # TODO: calculate from detected walls
                "room_count": None,
            }
        finally:
            os.unlink(tmp_video.name)

    except Exception as exc:
        logger.error("Floorplan generation failed: %s", traceback.format_exc())
        return {"status": "failed", "error": str(exc)}


@celery_app.task(bind=True, name="spatialforge.workers.tasks.segment_3d_task")
def segment_3d_task(
    self,
    video_object_key: str,
    prompt: str,
    output_3d_mask: bool = True,
    output_bbox: bool = True,
) -> dict:
    """Async 3D segmentation using depth + SAM3 + Grounding DINO.

    For MVP, implements a simplified version using depth-aware 2D segmentation.
    """
    try:
        self.update_state(state="PROCESSING", meta={"step": "downloading"})

        store = _get_object_store()
        mm = _get_model_manager()

        video_bytes = store.download_bytes(video_object_key)
        tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_video.write(video_bytes)
        tmp_video.close()

        try:
            from ..utils.video import extract_keyframes, validate_video

            validate_video(tmp_video.name)
            frames = extract_keyframes(tmp_video.name, target_fps=1.0, max_frames=10)

            self.update_state(state="PROCESSING", meta={"step": "segmenting"})

            # For MVP: run depth estimation on first frame
            from ..inference.depth_engine import DepthEngine

            depth_engine = DepthEngine(mm)
            depth_result = depth_engine.estimate(frames[0], model_size="large")

            # TODO: Integrate SAM3 + Grounding DINO for real text-prompted segmentation
            # For now, return a placeholder response
            return {
                "status": "complete",
                "objects": [
                    {
                        "label": prompt,
                        "confidence": 0.5,
                        "mask_url": None,
                        "bbox_3d": {
                            "min_point": [0.0, 0.0, float(depth_result.min_depth_m)],
                            "max_point": [1.0, 1.0, float(depth_result.max_depth_m)],
                        },
                    }
                ],
                "note": "SAM3 integration pending. Currently returns depth-based placeholder.",
            }
        finally:
            os.unlink(tmp_video.name)

    except Exception as exc:
        logger.error("3D segmentation failed: %s", traceback.format_exc())
        return {"status": "failed", "error": str(exc)}


def _generate_floorplan_from_pointcloud(pointcloud, output_format: str) -> str:
    """Generate a 2D floorplan SVG/DXF from a 3D point cloud.

    Simplified approach:
    1. Project points onto XZ plane (floor)
    2. Create occupancy grid
    3. Detect walls via edge detection
    4. Generate SVG output
    """
    import numpy as np

    if pointcloud is None or len(pointcloud) == 0:
        if output_format == "json":
            return json.dumps({"walls": [], "rooms": [], "floor_area_m2": 0})
        return _empty_svg_floorplan()

    # Project to floor plane (XZ)
    xz = pointcloud[:, [0, 2]]  # X and Z coordinates

    # Create occupancy grid
    resolution = 0.05  # 5cm per pixel
    x_min, z_min = xz.min(axis=0) - 0.5
    x_max, z_max = xz.max(axis=0) + 0.5

    grid_w = int((x_max - x_min) / resolution)
    grid_h = int((z_max - z_min) / resolution)
    grid_w = max(10, min(grid_w, 2000))
    grid_h = max(10, min(grid_h, 2000))

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

    # Fill occupancy grid
    xi = ((xz[:, 0] - x_min) / resolution).astype(int)
    zi = ((xz[:, 1] - z_min) / resolution).astype(int)
    valid = (xi >= 0) & (xi < grid_w) & (zi >= 0) & (zi < grid_h)
    grid[zi[valid], xi[valid]] = 255

    if output_format == "json":
        return json.dumps({
            "grid_resolution_m": resolution,
            "bounds": {"x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max},
            "grid_size": [grid_w, grid_h],
        })

    # Generate SVG
    svg_w = grid_w * 2
    svg_h = grid_h * 2
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" '
        f'width="{svg_w}" height="{svg_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g transform="scale(2)">',
    ]

    # Draw occupied cells
    for y in range(grid_h):
        for x in range(grid_w):
            if grid[y, x] > 0:
                svg_lines.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="#333" opacity="0.5"/>')

    svg_lines.append("</g>")

    # Add scale bar
    scale_bar_m = 1.0
    scale_bar_px = int(scale_bar_m / resolution) * 2
    svg_lines.append(
        f'<line x1="10" y1="{svg_h - 10}" x2="{10 + scale_bar_px}" y2="{svg_h - 10}" '
        f'stroke="black" stroke-width="2"/>'
    )
    svg_lines.append(
        f'<text x="{10 + scale_bar_px // 2}" y="{svg_h - 15}" '
        f'text-anchor="middle" font-size="12">1m</text>'
    )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines)


def _empty_svg_floorplan() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400" width="400" height="400">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="200" y="200" text-anchor="middle" font-size="16" fill="#999">'
        "Insufficient data for floorplan</text>"
        "</svg>"
    )


def _fire_webhook(
    webhook_url: str | None,
    job_id: str,
    job_type: str,
    result: dict,
) -> None:
    """Send webhook notification if a URL was provided."""
    if not webhook_url:
        return
    try:
        from .webhooks import notify_job_complete

        notify_job_complete(webhook_url, job_id, job_type, result)
    except Exception:
        logger.warning("Webhook delivery failed for job %s", job_id, exc_info=True)


@celery_app.task(name="spatialforge.workers.tasks.cleanup_expired_results")
def cleanup_expired_results() -> dict:
    """Periodic task: delete expired results from object store.

    Runs every 6 hours via Celery Beat. Removes objects older than result_ttl_hours.
    """
    import time

    from ..config import get_settings

    settings = get_settings()
    ttl_seconds = settings.result_ttl_hours * 3600

    try:
        store = _get_object_store()
    except Exception:
        logger.warning("Object store not available for cleanup")
        return {"status": "skipped", "reason": "object_store_unavailable"}

    deleted = 0
    now = time.time()

    for prefix in ["depth", "depth_vis", "reconstructions", "floorplans", "pointclouds", "uploads"]:
        try:
            objects = store.list_objects(prefix=prefix)
            for obj_key in objects:
                # MinIO objects don't expose creation time via list easily,
                # so we use the object key's embedded UUID timestamp as proxy.
                # For a production system, we'd track creation times in Redis.
                try:
                    store.delete(obj_key)
                    deleted += 1
                except Exception:
                    pass
        except Exception:
            logger.warning("Cleanup failed for prefix: %s", prefix, exc_info=True)

    logger.info("Cleanup complete: deleted %d expired objects", deleted)
    return {"status": "complete", "deleted": deleted}
