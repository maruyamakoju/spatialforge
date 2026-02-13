"""Celery application configuration for async job processing."""

from __future__ import annotations

import logging
import os

from celery import Celery
from celery.schedules import crontab
from celery.signals import celeryd_init

from .. import __version__

logger = logging.getLogger(__name__)


@celeryd_init.connect
def init_sentry_for_celery(**_kwargs) -> None:
    """Initialize Sentry for Celery workers."""
    sentry_dsn = os.getenv("SENTRY_DSN", "")
    if not sentry_dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.celery import CeleryIntegration

        sentry_environment = os.getenv("SENTRY_ENVIRONMENT", "production")
        try:
            traces_sample_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
        except ValueError:
            traces_sample_rate = 0.1

        debug = os.getenv("DEBUG", "false").lower() == "true"

        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=sentry_environment,
            traces_sample_rate=traces_sample_rate,
            integrations=[CeleryIntegration()],
            release=f"spatialforge@{__version__}",
            send_default_pii=False,
            attach_stacktrace=True,
            max_breadcrumbs=50,
            debug=debug,
        )
        logger.info("Sentry enabled for Celery worker (environment: %s)", sentry_environment)
    except ImportError:
        logger.warning("Sentry SDK not installed for Celery worker")


redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "spatialforge",
    broker=redis_url,
    backend=redis_url,
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
