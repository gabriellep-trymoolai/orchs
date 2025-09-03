"""Legacy Dashboard API endpoints - replaced by Langfuse analytics."""

from datetime import datetime, timezone
from fastapi import APIRouter

router = APIRouter()


@router.get("/dashboard/overview")
async def get_dashboard_overview():
    """Legacy dashboard endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/overview?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/dashboard/users")
async def get_dashboard_users():
    """Legacy dashboard endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/provider-breakdown?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/dashboard/costs")
async def get_dashboard_costs():
    """Legacy dashboard endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/time-series?metric=cost&use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/dashboard/performance")
async def get_dashboard_performance():
    """Legacy dashboard endpoint - replaced by Langfuse analytics."""
    return {
        "message": "Legacy monitoring replaced with Langfuse. Use /analytics endpoints instead.",
        "redirect": "/analytics/cache-performance?use_langfuse=true",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }