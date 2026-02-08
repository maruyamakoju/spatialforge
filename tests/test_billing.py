"""Tests for billing endpoints."""

from __future__ import annotations


def test_list_plans(client):
    """GET /v1/billing/plans returns plan list without auth."""
    resp = client.get("/v1/billing/plans")
    assert resp.status_code == 200
    plans = resp.json()
    assert len(plans) == 4
    names = {p["plan_id"] for p in plans}
    assert names == {"free", "builder", "pro", "enterprise"}

    # Verify pricing
    by_id = {p["plan_id"]: p for p in plans}
    assert by_id["free"]["price_usd"] == 0
    assert by_id["builder"]["price_usd"] == 29
    assert by_id["pro"]["price_usd"] == 99
    assert by_id["enterprise"]["price_usd"] == 499


def test_plan_info_has_features(client):
    """Each plan should have at least 3 features listed."""
    resp = client.get("/v1/billing/plans")
    for plan in resp.json():
        assert len(plan["features"]) >= 3, f"Plan {plan['plan_id']} has too few features"


def test_usage_returns_user_info(client, api_key):
    """GET /v1/billing/usage returns usage data for authenticated user."""
    resp = client.get("/v1/billing/usage", headers={"X-API-Key": api_key})
    assert resp.status_code == 200
    data = resp.json()
    assert data["plan"] == "admin"  # Test fixture returns admin user
    assert data["owner"] == "test"
    assert data["monthly_calls"] == 0
    assert data["monthly_limit"] == 999999999
    assert data["usage_pct"] == 0.0


def test_checkout_no_stripe(client, api_key):
    """POST /v1/billing/checkout returns 503 when Stripe not configured."""
    resp = client.post(
        "/v1/billing/checkout",
        headers={"X-API-Key": api_key},
        json={"plan": "pro", "email": "test@example.com"},
    )
    assert resp.status_code == 503
    assert "Billing not configured" in resp.json()["detail"]


def test_portal_no_stripe(client, api_key):
    """POST /v1/billing/portal returns 503 when Stripe not configured."""
    resp = client.post(
        "/v1/billing/portal",
        headers={"X-API-Key": api_key},
        json={"email": "test@example.com"},
    )
    assert resp.status_code == 503


def test_webhook_no_stripe(client):
    """POST /v1/billing/webhooks returns 503 when Stripe not configured."""
    resp = client.post(
        "/v1/billing/webhooks",
        content=b"test",
        headers={"stripe-signature": "test"},
    )
    assert resp.status_code == 503


def test_checkout_invalid_plan(client, api_key, app):
    """POST /v1/billing/checkout rejects invalid plan names when Stripe configured."""
    from unittest.mock import MagicMock

    # Temporarily set stripe_billing so we get past the 503 check
    app.state.stripe_billing = MagicMock()

    resp = client.post(
        "/v1/billing/checkout",
        headers={"X-API-Key": api_key},
        json={"plan": "invalid_plan", "email": "test@example.com"},
    )
    assert resp.status_code == 400
    assert "Invalid plan" in resp.json()["detail"]

    # Clean up
    app.state.stripe_billing = None


def test_checkout_rejects_free_plan(client, api_key, app):
    """POST /v1/billing/checkout rejects free plan (no checkout needed)."""
    from unittest.mock import MagicMock

    app.state.stripe_billing = MagicMock()

    resp = client.post(
        "/v1/billing/checkout",
        headers={"X-API-Key": api_key},
        json={"plan": "free", "email": "test@example.com"},
    )
    assert resp.status_code == 400
    assert "Free plan" in resp.json()["detail"]

    app.state.stripe_billing = None


def test_checkout_rejects_admin_plan(client, api_key, app):
    """POST /v1/billing/checkout rejects admin plan (not purchasable)."""
    from unittest.mock import MagicMock

    app.state.stripe_billing = MagicMock()

    resp = client.post(
        "/v1/billing/checkout",
        headers={"X-API-Key": api_key},
        json={"plan": "admin", "email": "test@example.com"},
    )
    assert resp.status_code == 403

    app.state.stripe_billing = None
