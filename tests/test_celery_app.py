"""Celery app config tests."""

from __future__ import annotations

import importlib


def test_celery_app_uses_redis_url_from_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis.example:6380/9")
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    import spatialforge.workers.celery_app as celery_app_module

    celery_app_module = importlib.reload(celery_app_module)

    assert celery_app_module.celery_app.conf.broker_url == "redis://redis.example:6380/9"
    assert celery_app_module.celery_app.conf.result_backend == "redis://redis.example:6380/9"


def test_init_sentry_for_celery_no_dsn_noop(monkeypatch):
    monkeypatch.delenv("SENTRY_DSN", raising=False)

    import spatialforge.workers.celery_app as celery_app_module

    celery_app_module.init_sentry_for_celery()
