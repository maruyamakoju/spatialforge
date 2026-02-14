"""Celery tasks for async GPU-heavy operations."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import traceback
from contextlib import contextmanager
from typing import Any

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Lazy-initialized shared resources (initialized once per worker process)
_model_manager = None
_object_store = None
_redis_sync = None


def _is_final_retry(task) -> bool:
    """Return True when the task has reached its final retry attempt."""
    retries = int(getattr(task.request, "retries", 0))
    max_retries = int(getattr(task, "max_retries", 0))
    return retries >= max_retries


def _normalize_reconstruct_error(exc: Exception) -> str:
    """Map internal reconstruct errors to stable client-facing messages."""
    raw = str(exc)
    if "requires optional dependency 'open3d'" in raw:
        return "TSDF backend unavailable: install optional dependency '.[tsdf]' (open3d)."
    if "requires metric depth maps" in raw:
        return "TSDF requires metric depth; use a metric depth model/back-end configuration."
    return raw


def _get_model_manager():
    """Lazy-init model manager (one per worker process)."""
    global _model_manager
    if _model_manager is None:
        from ..config import get_settings
        from ..inference.model_manager import create_model_manager_from_settings

        settings = get_settings()
        _model_manager = create_model_manager_from_settings(settings)
    return _model_manager


def _get_object_store():
    """Lazy-init object store."""
    global _object_store, _redis_sync
    if _object_store is None:
        import redis

        from ..config import get_settings
        from ..storage.object_store import ObjectStore

        settings = get_settings()
        if _redis_sync is None:
            try:
                _redis_sync = redis.from_url(settings.redis_url, decode_responses=True)
                _redis_sync.ping()
            except Exception:
                logger.warning("Sync Redis unavailable in worker — object TTL tracking disabled")
                _redis_sync = None

        _object_store = ObjectStore(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
            redis=_redis_sync,
        )
    return _object_store


@contextmanager
def _download_video_to_temp(video_object_key: str):
    """Download video from object store to a temp file, cleanup on exit."""
    store = _get_object_store()
    video_bytes = store.download_bytes(video_object_key)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(video_bytes)
    try:
        yield tmp_video.name, store
    finally:
        os.unlink(tmp_video.name)


def _extract_frames(video_path: str, target_fps: float = 2.0, max_frames: int = 200):
    """Validate video and extract keyframes."""
    from ..utils.video import extract_keyframes, validate_video

    validate_video(video_path)
    return extract_keyframes(video_path, target_fps=target_fps, max_frames=max_frames)


# ── Task definitions ─────────────────────────────────────────


_GPU_TASK_OPTS: dict[str, Any] = {
    "bind": True,
    "autoretry_for": (Exception,),
    "retry_backoff": True,
    "retry_backoff_max": 600,
    "retry_jitter": True,
    "max_retries": 3,
}


@celery_app.task(name="spatialforge.workers.tasks.reconstruct_task", **_GPU_TASK_OPTS)
def reconstruct_task(
    self,
    video_object_key: str,
    quality: str = "standard",
    output_format: str = "gaussian",
    webhook_url: str | None = None,
) -> dict:
    """Async 3D reconstruction from uploaded video."""
    try:
        self.update_state(state="PROCESSING", meta={"step": "downloading_video"})
        mm = _get_model_manager()
        from ..config import get_settings

        settings = get_settings()

        with _download_video_to_temp(video_object_key) as (video_path, store):
            self.update_state(state="PROCESSING", meta={"step": "extracting_keyframes"})
            frames = _extract_frames(video_path, target_fps=2.0)

            from ..inference.reconstruct_engine import ReconstructEngine

            engine = ReconstructEngine(mm, backend=settings.reconstruct_backend)
            self.update_state(
                state="PROCESSING",
                meta={
                    "step": "reconstructing",
                    "num_frames": len(frames),
                    "reconstruct_backend": settings.reconstruct_backend,
                },
            )

            def _progress(step: str, meta: dict[str, Any] | None = None) -> None:
                payload: dict[str, Any] = {"step": step}
                if meta:
                    payload.update(meta)
                self.update_state(state="PROCESSING", meta=payload)

            result = engine.reconstruct(
                frames=frames,
                quality=quality,
                output_format=output_format,
                progress_callback=_progress,
            )

            self.update_state(state="PROCESSING", meta={"step": "uploading_results"})

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

    except Exception as exc:
        logger.error("Reconstruction failed: %s", traceback.format_exc())
        if _is_final_retry(self):
            err = {"status": "failed", "error": _normalize_reconstruct_error(exc)}
            _fire_webhook(webhook_url, self.request.id, "reconstruct", err)
            return err
        raise


@celery_app.task(name="spatialforge.workers.tasks.floorplan_task", **_GPU_TASK_OPTS)
def floorplan_task(
    self,
    video_object_key: str,
    output_format: str = "svg",
) -> dict:
    """Async floorplan generation from room walkthrough video."""
    try:
        self.update_state(state="PROCESSING", meta={"step": "downloading_video"})
        mm = _get_model_manager()

        with _download_video_to_temp(video_object_key) as (video_path, store):
            self.update_state(state="PROCESSING", meta={"step": "processing"})
            frames = _extract_frames(video_path, target_fps=1.0, max_frames=60)

            from ..inference.pose_engine import PoseEngine

            pose_engine = PoseEngine(mm)
            pose_result = pose_engine.estimate_poses(frames, output_pointcloud=True)

            self.update_state(state="PROCESSING", meta={"step": "generating_floorplan"})

            floorplan_data = _generate_floorplan_from_pointcloud(
                pose_result.pointcloud, output_format
            )

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
            "floor_area_m2": None,
            "room_count": None,
        }

    except Exception as exc:
        logger.error("Floorplan generation failed: %s", traceback.format_exc())
        if _is_final_retry(self):
            return {"status": "failed", "error": str(exc)}
        raise


@celery_app.task(name="spatialforge.workers.tasks.segment_3d_task", **_GPU_TASK_OPTS)
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
        mm = _get_model_manager()

        with _download_video_to_temp(video_object_key) as (video_path, _store):
            frames = _extract_frames(video_path, target_fps=1.0, max_frames=10)

            self.update_state(state="PROCESSING", meta={"step": "segmenting"})

            from ..inference.depth_engine import DepthEngine

            depth_engine = DepthEngine(mm)
            depth_result = depth_engine.estimate(frames[0], model_size="large")

            # TODO: Integrate SAM3 + Grounding DINO for real text-prompted segmentation
            return {
                "status": "complete",
                "objects": [
                    {
                        "label": prompt,
                        "confidence": 0.5,
                        "mask_url": None,
                        "bbox_3d": {
                            "min_point": [0.0, 0.0, float(depth_result.min_depth)],
                            "max_point": [1.0, 1.0, float(depth_result.max_depth)],
                        },
                    }
                ],
                "note": "SAM3 integration pending. Currently returns depth-based placeholder.",
            }

    except Exception as exc:
        logger.error("3D segmentation failed: %s", traceback.format_exc())
        if _is_final_retry(self):
            return {"status": "failed", "error": str(exc)}
        raise


# ── Floorplan generation helper ──────────────────────────────


def _generate_floorplan_from_pointcloud(pointcloud, output_format: str) -> str:
    """Generate a 2D floorplan SVG/DXF from a 3D point cloud."""
    import numpy as np

    if pointcloud is None or len(pointcloud) == 0:
        if output_format == "json":
            return json.dumps({"walls": [], "rooms": [], "floor_area_m2": 0})
        return _empty_svg_floorplan()

    # Project to floor plane (XZ)
    xz = pointcloud[:, [0, 2]]

    # Create occupancy grid
    resolution = 0.05  # 5cm per pixel
    x_min, z_min = xz.min(axis=0) - 0.5
    x_max, z_max = xz.max(axis=0) + 0.5

    grid_w = max(10, min(int((x_max - x_min) / resolution), 2000))
    grid_h = max(10, min(int((z_max - z_min) / resolution), 2000))

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

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
        '<g transform="scale(2)">',
    ]

    for y in range(grid_h):
        for x in range(grid_w):
            if grid[y, x] > 0:
                svg_lines.append(f'<rect x="{x}" y="{y}" width="1" height="1" fill="#333" opacity="0.5"/>')

    svg_lines.append("</g>")

    scale_bar_px = int(1.0 / resolution) * 2
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


# ── Periodic tasks ───────────────────────────────────────────


@celery_app.task(name="spatialforge.workers.tasks.cleanup_expired_results")
def cleanup_expired_results() -> dict:
    """Periodic task: delete expired results from object store.

    Runs every 6 hours via Celery Beat. Uses Redis TTL metadata to determine
    which objects have expired — never blindly deletes everything.
    """
    from ..config import get_settings

    settings = get_settings()

    try:
        store = _get_object_store()
    except Exception:
        logger.warning("Object store not available for cleanup")
        return {"status": "skipped", "reason": "object_store_unavailable"}

    deleted = store.cleanup_expired(ttl_hours=settings.result_ttl_hours)

    logger.info("Cleanup complete: deleted %d expired objects (TTL=%dh)", deleted, settings.result_ttl_hours)
    return {"status": "complete", "deleted": deleted}
