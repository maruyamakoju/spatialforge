"""Webhook notification for async job completion."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = 10.0  # seconds
MAX_RETRIES = 3


def send_webhook(
    url: str,
    payload: dict,
    secret: str | None = None,
    retries: int = MAX_RETRIES,
) -> bool:
    """Send a webhook notification to the given URL.

    Args:
        url: The webhook URL to POST to.
        payload: JSON-serializable dict with job result data.
        secret: Optional secret for HMAC-SHA256 signature.
        retries: Number of retry attempts on failure.

    Returns:
        True if webhook was delivered successfully.
    """
    body = json.dumps(payload, default=str)

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": "SpatialForge-Webhook/0.1.0",
        "X-SpatialForge-Event": payload.get("event", "job.completed"),
        "X-SpatialForge-Timestamp": str(int(time.time())),
    }

    # Sign the payload if secret is provided
    if secret:
        signature = hmac.new(
            secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()
        headers["X-SpatialForge-Signature"] = f"sha256={signature}"

    for attempt in range(retries):
        try:
            with httpx.Client(timeout=WEBHOOK_TIMEOUT) as client:
                resp = client.post(url, content=body, headers=headers)

            if 200 <= resp.status_code < 300:
                logger.info("Webhook delivered: %s (status=%d)", url, resp.status_code)
                return True

            logger.warning(
                "Webhook %s returned %d (attempt %d/%d)",
                url, resp.status_code, attempt + 1, retries,
            )
        except httpx.TimeoutException:
            logger.warning("Webhook timeout: %s (attempt %d/%d)", url, attempt + 1, retries)
        except Exception as e:
            logger.warning("Webhook error: %s — %s (attempt %d/%d)", url, e, attempt + 1, retries)

        # Exponential backoff
        if attempt < retries - 1:
            time.sleep(2 ** attempt)

    logger.error("Webhook delivery failed after %d attempts: %s", retries, url)
    return False


def notify_job_complete(
    webhook_url: str,
    job_id: str,
    job_type: str,
    result: dict,
    secret: str | None = None,
) -> bool:
    """Send a job completion webhook."""
    payload = {
        "event": "job.completed",
        "job_id": job_id,
        "job_type": job_type,
        "status": result.get("status", "complete"),
        "data": result,
        "timestamp": time.time(),
    }
    return send_webhook(webhook_url, payload, secret=secret)
