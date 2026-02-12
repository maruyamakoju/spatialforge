"""/v1/billing — Self-service billing endpoints.

Endpoints:
  POST /v1/billing/checkout     — Create Stripe Checkout session for plan upgrade
  POST /v1/billing/portal       — Create Stripe Customer Portal session
  GET  /v1/billing/plans        — List available plans and pricing
  GET  /v1/billing/usage        — Get current usage stats for authenticated user
  POST /v1/billing/webhooks     — Stripe webhook receiver (no auth)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ...auth.api_keys import APIKeyRecord, Plan, get_current_user
from ...models.requests import CheckoutRequest, PortalRequest
from ...models.responses import CheckoutResponse, PlanInfo, PortalResponse, UsageResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Endpoints ─────────────────────────────────────────────────

@router.get("/billing/plans", response_model=list[PlanInfo])
async def list_plans():
    """List available plans and pricing. No authentication required."""
    return [
        PlanInfo(
            name="Free",
            plan_id="free",
            price_usd=0,
            monthly_limit=100,
            features=[
                "100 API calls/month",
                "Depth estimation (all models)",
                "Community support",
            ],
        ),
        PlanInfo(
            name="Builder",
            plan_id="builder",
            price_usd=29,
            monthly_limit=5_000,
            features=[
                "5,000 API calls/month",
                "All endpoints (depth, pose, measure)",
                "Priority inference queue",
                "Email support",
            ],
        ),
        PlanInfo(
            name="Pro",
            plan_id="pro",
            price_usd=99,
            monthly_limit=50_000,
            features=[
                "50,000 API calls/month",
                "All endpoints including async (reconstruct, floorplan, segment-3d)",
                "Batch processing",
                "Priority support",
            ],
        ),
        PlanInfo(
            name="Enterprise",
            plan_id="enterprise",
            price_usd=499,
            monthly_limit=999_999_999,
            features=[
                "Unlimited API calls",
                "All endpoints + GPU priority",
                "Custom model deployment",
                "SLA guarantee",
                "Dedicated support",
            ],
        ),
    ]


@router.get("/billing/usage", response_model=UsageResponse)
async def get_usage(user: APIKeyRecord = Depends(get_current_user)):
    """Get current billing period usage for the authenticated API key."""
    limit = user.monthly_limit
    usage_pct = (user.monthly_calls / limit * 100) if limit > 0 else 0

    return UsageResponse(
        plan=user.plan.value,
        monthly_calls=user.monthly_calls,
        monthly_limit=limit,
        usage_pct=round(min(usage_pct, 100.0), 1),
        owner=user.owner,
    )


@router.post("/billing/checkout", response_model=CheckoutResponse)
async def create_checkout(
    body: CheckoutRequest,
    request: Request,
    user: APIKeyRecord = Depends(get_current_user),
):
    """Create a Stripe Checkout session to upgrade plan.

    After successful payment, the API key will be upgraded automatically
    via webhook. The checkout URL is valid for 24 hours.
    """
    billing = getattr(request.app.state, "stripe_billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing not configured. Set STRIPE_SECRET_KEY env var.")

    try:
        plan_enum = Plan(body.plan)
    except ValueError:
        valid = ["builder", "pro", "enterprise"]
        raise HTTPException(status_code=400, detail=f"Invalid plan: {body.plan}. Use: {valid}") from None

    if plan_enum == Plan.FREE:
        raise HTTPException(status_code=400, detail="Free plan does not require checkout. Use the portal to downgrade.")

    if plan_enum == Plan.ADMIN:
        raise HTTPException(status_code=403, detail="Admin plan is not purchasable")

    try:
        url = await billing.create_checkout_session(
            plan=plan_enum,
            owner_email=body.email,
            api_key_hash=user.key_hash,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )
    except Exception as e:
        logger.error("Stripe checkout creation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to create checkout session") from None

    return CheckoutResponse(checkout_url=url, plan=body.plan)


@router.post("/billing/portal", response_model=PortalResponse)
async def create_portal(
    body: PortalRequest,
    request: Request,
    _user: APIKeyRecord = Depends(get_current_user),
):
    """Create a Stripe Customer Portal session for billing management.

    Allows customers to update payment methods, view invoices, and cancel.
    """
    billing = getattr(request.app.state, "stripe_billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing not configured. Set STRIPE_SECRET_KEY env var.")

    try:
        url = await billing.create_portal_session(owner_email=body.email)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        logger.error("Stripe portal creation failed: %s", e)
        raise HTTPException(status_code=502, detail="Failed to create portal session") from None

    return PortalResponse(portal_url=url)


@router.post("/billing/webhooks")
async def stripe_webhook(request: Request):
    """Receive Stripe webhook events. No API key auth — verified by Stripe signature.

    Configure this URL in your Stripe Dashboard → Webhooks:
      https://your-domain/v1/billing/webhooks

    Required events:
      - checkout.session.completed
      - customer.subscription.updated
      - customer.subscription.deleted
      - invoice.paid
      - invoice.payment_failed
    """
    billing = getattr(request.app.state, "stripe_billing", None)
    if billing is None:
        raise HTTPException(status_code=503, detail="Billing not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    try:
        event = billing.construct_webhook_event(payload, sig_header)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload") from None
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature") from None

    result = await billing.handle_webhook_event(event)
    logger.info("Webhook processed: %s → %s", event.type, result)

    return {"status": "ok", "event_type": event.type, "result": result}
