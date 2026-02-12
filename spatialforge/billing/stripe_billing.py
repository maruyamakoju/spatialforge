"""Stripe billing — subscription management & usage-based billing.

Architecture:
  - Stripe is the source of truth for subscriptions and payment status.
  - Redis caches active subscription state for low-latency quota checks.
  - Webhook events sync Stripe state → Redis.
  - Checkout Sessions handle self-service plan upgrades.
  - Customer Portal lets users manage payment methods and cancel.

Stripe Products/Prices are created once via `ensure_stripe_products()` on startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import stripe
from starlette.concurrency import run_in_threadpool

from ..auth.api_keys import Plan

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger(__name__)

# Redis key prefix for subscription state
SUB_PREFIX = "stripe_sub:"
CUSTOMER_PREFIX = "stripe_cust:"

# Plan → Stripe price mapping (monthly USD cents)
PLAN_PRICES: dict[Plan, dict[str, Any]] = {
    Plan.FREE: {
        "amount": 0,
        "name": "Free",
        "monthly_limit": 100,
        "lookup_key": "spatialforge_free",
    },
    Plan.BUILDER: {
        "amount": 2900,  # $29/mo
        "name": "Builder",
        "monthly_limit": 5_000,
        "lookup_key": "spatialforge_builder",
    },
    Plan.PRO: {
        "amount": 9900,  # $99/mo
        "name": "Pro",
        "monthly_limit": 50_000,
        "lookup_key": "spatialforge_pro",
    },
    Plan.ENTERPRISE: {
        "amount": 49900,  # $499/mo
        "name": "Enterprise",
        "monthly_limit": 999_999_999,
        "lookup_key": "spatialforge_enterprise",
    },
}


class StripeBilling:
    """Manages Stripe subscriptions and syncs state to Redis."""

    def __init__(self, secret_key: str, webhook_secret: str, redis: Redis | None = None) -> None:
        stripe.api_key = secret_key
        self._webhook_secret = webhook_secret
        self._redis = redis
        self._price_ids: dict[str, str] = {}  # lookup_key → price_id

    async def ensure_products(self) -> None:
        """Create or fetch Stripe products/prices for each plan.

        Uses lookup_keys so prices are idempotent across deploys.
        """
        # Create the SpatialForge product (idempotent via metadata search)
        products = await run_in_threadpool(
            stripe.Product.search, query="metadata['app']:'spatialforge'", limit=1,
        )
        if products.data:
            product = products.data[0]
        else:
            product = await run_in_threadpool(
                stripe.Product.create,
                name="SpatialForge API",
                description="Spatial intelligence API — depth estimation, 3D reconstruction, and more.",
                metadata={"app": "spatialforge"},
            )
            logger.info("Created Stripe product: %s", product.id)

        # Create prices for each paid plan
        for plan, config in PLAN_PRICES.items():
            if plan == Plan.FREE:
                continue  # Free plan has no Stripe price

            lookup_key = config["lookup_key"]

            # Check if price already exists
            existing = await run_in_threadpool(
                stripe.Price.list, lookup_keys=[lookup_key], limit=1,
            )
            if existing.data:
                self._price_ids[lookup_key] = existing.data[0].id
                logger.debug("Found existing price for %s: %s", lookup_key, existing.data[0].id)
                continue

            price = await run_in_threadpool(
                stripe.Price.create,
                product=product.id,
                unit_amount=config["amount"],
                currency="usd",
                recurring={"interval": "month"},
                lookup_key=lookup_key,
                metadata={"plan": plan.value, "monthly_limit": str(config["monthly_limit"])},
            )
            self._price_ids[lookup_key] = price.id
            logger.info("Created Stripe price for %s: %s ($%s/mo)", plan.value, price.id, config["amount"] / 100)

    def get_price_id(self, plan: Plan) -> str | None:
        """Get the Stripe price ID for a plan."""
        config = PLAN_PRICES.get(plan)
        if config is None:
            return None
        return self._price_ids.get(config["lookup_key"])

    async def create_checkout_session(
        self,
        plan: Plan,
        owner_email: str,
        api_key_hash: str,
        success_url: str,
        cancel_url: str,
    ) -> str:
        """Create a Stripe Checkout Session for plan subscription.

        Returns the checkout session URL.
        """
        price_id = self.get_price_id(plan)
        if price_id is None:
            raise ValueError(f"No Stripe price configured for plan: {plan.value}")

        # Find or create Stripe customer
        customer_id = await self._get_or_create_customer(owner_email, api_key_hash)

        session = await run_in_threadpool(
            stripe.checkout.Session.create,
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={"api_key_hash": api_key_hash, "plan": plan.value},
            subscription_data={"metadata": {"api_key_hash": api_key_hash, "plan": plan.value}},
        )
        return session.url

    async def create_portal_session(self, owner_email: str) -> str:
        """Create a Stripe Customer Portal session for self-service management.

        Returns the portal URL.
        """
        # Find customer by email
        customers = await run_in_threadpool(
            stripe.Customer.search, query=f"email:'{owner_email}'", limit=1,
        )
        if not customers.data:
            raise ValueError(f"No Stripe customer found for email: {owner_email}")

        session = await run_in_threadpool(
            stripe.billing_portal.Session.create,
            customer=customers.data[0].id,
            return_url="https://spatialforge-demo.fly.dev/docs",
        )
        return session.url

    def construct_webhook_event(self, payload: bytes, sig_header: str) -> stripe.Event:
        """Verify and construct a Stripe webhook event."""
        return stripe.Webhook.construct_event(payload, sig_header, self._webhook_secret)

    async def handle_webhook_event(self, event: stripe.Event) -> str:
        """Process a Stripe webhook event and sync state to Redis.

        Returns a description of what was processed.
        """
        event_type = event.type
        data = event.data.object

        if event_type == "checkout.session.completed":
            return await self._handle_checkout_completed(data)
        elif event_type == "customer.subscription.updated":
            return await self._handle_subscription_updated(data)
        elif event_type == "customer.subscription.deleted":
            return await self._handle_subscription_deleted(data)
        elif event_type == "invoice.paid":
            return await self._handle_invoice_paid(data)
        elif event_type == "invoice.payment_failed":
            return await self._handle_payment_failed(data)
        else:
            return f"Unhandled event type: {event_type}"

    # ── Webhook handlers ──────────────────────────────────────

    async def _handle_checkout_completed(self, session: Any) -> str:
        """New subscription created via checkout."""
        api_key_hash = session.get("metadata", {}).get("api_key_hash")
        plan = session.get("metadata", {}).get("plan")
        subscription_id = session.get("subscription")
        customer_id = session.get("customer")

        if not api_key_hash or not plan:
            return "Checkout completed but missing metadata"

        # Update API key plan in Redis
        await self._sync_api_key_plan(api_key_hash, plan, subscription_id, customer_id)
        return f"Checkout completed: {plan} plan for key {api_key_hash[:8]}..."

    async def _handle_subscription_updated(self, subscription: Any) -> str:
        """Subscription plan changed or renewed."""
        api_key_hash = subscription.get("metadata", {}).get("api_key_hash")
        if not api_key_hash:
            return "Subscription updated but missing api_key_hash metadata"

        status = subscription.get("status")
        plan = subscription.get("metadata", {}).get("plan", "free")

        if status == "active":
            await self._sync_api_key_plan(
                api_key_hash, plan, subscription.get("id"), subscription.get("customer"),
            )
            return f"Subscription active: {plan} for {api_key_hash[:8]}..."
        elif status in ("past_due", "unpaid"):
            # Downgrade to free on payment issues
            await self._sync_api_key_plan(api_key_hash, "free", None, subscription.get("customer"))
            return f"Subscription {status}: downgraded {api_key_hash[:8]}... to free"
        else:
            return f"Subscription status: {status} for {api_key_hash[:8]}..."

    async def _handle_subscription_deleted(self, subscription: Any) -> str:
        """Subscription canceled."""
        api_key_hash = subscription.get("metadata", {}).get("api_key_hash")
        if not api_key_hash:
            return "Subscription deleted but missing metadata"

        # Downgrade to free
        await self._sync_api_key_plan(api_key_hash, "free", None, subscription.get("customer"))
        return f"Subscription canceled: downgraded {api_key_hash[:8]}... to free"

    async def _handle_invoice_paid(self, invoice: Any) -> str:
        """Invoice paid — reset monthly usage counter."""
        subscription_id = invoice.get("subscription")
        if not subscription_id:
            return "Invoice paid (no subscription)"

        # Find the API key hash from subscription metadata
        try:
            sub = await run_in_threadpool(stripe.Subscription.retrieve, subscription_id)
            api_key_hash = sub.metadata.get("api_key_hash")
            if api_key_hash and self._redis:
                # Reset monthly call counter
                await self._redis.hset(f"apikey:{api_key_hash}", "monthly_calls", "0")
                return f"Invoice paid: reset usage for {api_key_hash[:8]}..."
        except Exception:
            logger.warning("Failed to reset usage on invoice.paid", exc_info=True)

        return "Invoice paid"

    async def _handle_payment_failed(self, invoice: Any) -> str:
        """Payment failed — log warning but don't immediately downgrade."""
        customer_id = invoice.get("customer")
        return f"Payment failed for customer {customer_id} — Stripe will retry"

    # ── Internal helpers ──────────────────────────────────────

    async def _get_or_create_customer(self, email: str, api_key_hash: str) -> str:
        """Find existing Stripe customer by email, or create a new one."""
        # Check Redis cache first
        if self._redis:
            cached = await self._redis.get(f"{CUSTOMER_PREFIX}{email}")
            if cached:
                return cached

        # Search Stripe
        customers = await run_in_threadpool(
            stripe.Customer.search, query=f"email:'{email}'", limit=1,
        )
        if customers.data:
            customer_id = customers.data[0].id
        else:
            customer = await run_in_threadpool(
                stripe.Customer.create,
                email=email,
                metadata={"api_key_hash": api_key_hash, "source": "spatialforge"},
            )
            customer_id = customer.id

        # Cache in Redis
        if self._redis:
            await self._redis.set(f"{CUSTOMER_PREFIX}{email}", customer_id, ex=86400)

        return customer_id

    async def _sync_api_key_plan(
        self,
        api_key_hash: str,
        plan: str,
        subscription_id: str | None,
        customer_id: str | None,
    ) -> None:
        """Sync subscription state to the API key record in Redis."""
        if self._redis is None:
            logger.warning("Cannot sync plan — Redis not available")
            return

        from ..config import get_settings

        settings = get_settings()
        limits = {
            "free": settings.rate_limit_free,
            "builder": settings.rate_limit_builder,
            "pro": settings.rate_limit_pro,
            "enterprise": 999_999_999,
        }

        redis_key = f"apikey:{api_key_hash}"

        # Check the key exists
        exists = await self._redis.exists(redis_key)
        if not exists:
            logger.warning("API key hash not found in Redis: %s", api_key_hash[:8])
            return

        # Update plan, monthly limit, and reset counter
        mapping: dict[str, str] = {
            "plan": plan,
            "monthly_limit": str(limits.get(plan, 100)),
            "monthly_calls": "0",  # Reset on plan change
        }
        if subscription_id:
            mapping["stripe_subscription_id"] = subscription_id
        if customer_id:
            mapping["stripe_customer_id"] = customer_id

        await self._redis.hset(redis_key, mapping=mapping)
        logger.info("Synced plan=%s for key %s...", plan, api_key_hash[:8])
