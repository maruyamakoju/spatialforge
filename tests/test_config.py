"""Configuration validation tests."""

from __future__ import annotations

import pytest

from spatialforge.config import get_settings


def _set_production_secrets(monkeypatch):
    monkeypatch.setenv("API_KEY_SECRET", "x" * 48)
    monkeypatch.setenv("ADMIN_API_KEY", "sf_" + "a" * 32)
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio_access_key_strong")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio_secret_key_strong")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cors_wildcard_rejected_in_production(monkeypatch):
    _set_production_secrets(monkeypatch)
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOWED_ORIGINS", "[\"*\"]")

    with pytest.raises(RuntimeError):
        get_settings()


def test_cors_wildcard_allowed_in_demo(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", "[\"*\"]")

    s = get_settings()
    assert s.demo_mode is True
    assert s.allowed_origins == ["*"]


def test_security_fields_can_be_overridden(monkeypatch):
    _set_production_secrets(monkeypatch)
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOWED_ORIGINS", "[\"https://api.example.org\"]")
    monkeypatch.setenv("SECURITY_CONTACT", "mailto:security@cam.ac.uk")
    monkeypatch.setenv("SECURITY_CANONICAL_URL", "https://api.example.org/.well-known/security.txt")
    monkeypatch.setenv("SECURITY_EXPIRES", "2027-12-31T00:00:00Z")
    monkeypatch.setenv("SECURITY_PREFERRED_LANGUAGES", "en,ja")
    monkeypatch.setenv("SECURITY_ENCRYPTION_URL", "https://api.example.org/.well-known/pgp-key.txt")

    s = get_settings()

    assert s.security_contact == "mailto:security@cam.ac.uk"
    assert s.security_canonical_url == "https://api.example.org/.well-known/security.txt"
    assert s.security_expires == "2027-12-31T00:00:00Z"
    assert s.security_preferred_languages == "en,ja"
    assert s.security_encryption_url == "https://api.example.org/.well-known/pgp-key.txt"


def test_depth_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEPTH_BACKEND", "da3")

    s = get_settings()
    assert s.depth_backend == "da3"


def test_depth_backend_invalid_rejected(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("DEPTH_BACKEND", "invalid-backend")

    with pytest.raises(ValueError, match="DEPTH_BACKEND"):
        get_settings()


def test_reconstruct_backend_can_be_overridden(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("RECONSTRUCT_BACKEND", "tsdf")

    s = get_settings()
    assert s.reconstruct_backend == "tsdf"


def test_reconstruct_backend_invalid_rejected(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("RECONSTRUCT_BACKEND", "invalid-backend")

    with pytest.raises(ValueError, match="RECONSTRUCT_BACKEND"):
        get_settings()
