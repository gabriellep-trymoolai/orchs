"""Legacy Metrics API endpoints - replaced by Langfuse analytics."""

from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/metrics/users/{user_id}/realtime")
async def get_user_realtime_metrics(user_id: str):
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "user_id": user_id,
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/overview?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/metrics/users/{user_id}/history")
async def get_user_historical_metrics(user_id: str):
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "user_id": user_id,
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/provider-breakdown?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/metrics/users/{user_id}/requests")
async def get_user_request_details(user_id: str):
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "user_id": user_id,
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/time-series?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/metrics/organization/realtime")
async def get_organization_realtime_metrics():
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/overview?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/metrics/organization/users")
async def get_organization_user_metrics():
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/provider-breakdown?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/metrics/organization/summary")
async def get_organization_summary():
    """Legacy endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/overview?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }