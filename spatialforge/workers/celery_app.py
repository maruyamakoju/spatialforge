"""Celery application configuration for async job processing."""

from __future__ import annotations

from celery import Celery
from celery.schedules import crontab

from ..config import get_settings

settings = get_settings()

celery_app = Celery(
    "spatialforge",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Results
    result_expires=3600 * 24,  # 24 hours
    # Task routing
    task_routes={
        "spatialforge.workers.tasks.reconstruct_task": {"queue": "gpu_heavy"},
        "spatialforge.workers.tasks.floorplan_task": {"queue": "gpu_heavy"},
        "spatialforge.workers.tasks.segment_3d_task": {"queue": "gpu_heavy"},
        "spatialforge.workers.tasks.cleanup_expired_results": {"queue": "default"},
    },
    # Default queue for light tasks
    task_default_queue="default",
    # Celery Beat schedule (periodic tasks)
    beat_schedule={
        "cleanup-expired-results": {
            "task": "spatialforge.workers.tasks.cleanup_expired_results",
            "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["spatialforge.workers"])
