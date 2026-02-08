"""/v1/admin — Admin endpoints for API key management and system status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from ...auth.api_keys import APIKeyRecord, Plan, get_current_user

router = APIRouter()


def _require_admin(user: APIKeyRecord = Depends(get_current_user)) -> APIKeyRecord:
    if user.plan != Plan.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


@router.post("/keys")
async def create_api_key(
    request: Request,
    owner: str,
    plan: str = "free",
    admin: APIKeyRecord = Depends(_require_admin),
):
    """Create a new API key for a user."""
    try:
        plan_enum = Plan(plan)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {plan}. Use: {[p.value for p in Plan]}")

    manager = request.app.state.key_manager
    raw_key = await manager.create_key(owner=owner, plan=plan_enum)

    return {
        "api_key": raw_key,
        "owner": owner,
        "plan": plan_enum.value,
        "message": "Store this key securely — it cannot be retrieved later.",
    }


@router.get("/status")
async def system_status(
    request: Request,
    admin: APIKeyRecord = Depends(_require_admin),
):
    """Get system status including GPU info and loaded models."""
    mm = request.app.state.model_manager
    gpu_info = mm.gpu_status()

    import torch

    return {
        "version": "0.1.0",
        "gpu": gpu_info,
        "loaded_models": mm.loaded_models,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
