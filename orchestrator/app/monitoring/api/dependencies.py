"""FastAPI dependencies."""

import os
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..models import get_db
from ..middleware import SystemPerformanceMiddleware
# Note: PromptResponseAgent moved to orchestrator service


# Global app state - will be set by main.py
app_state = None


def set_app_state(state):
    """Set the global app state reference."""
    global app_state
    app_state = state



async def get_system_monitoring_middleware(
    db: AsyncSession = Depends(get_db)
) -> SystemPerformanceMiddleware:
    """Get system performance monitoring middleware instance."""
    if app_state and hasattr(app_state, 'system_middleware') and app_state.system_middleware:
        # Update database session for the middleware
        app_state.system_middleware.db_session = db
        return app_state.system_middleware
    
    # Fallback: create new instance
    redis_client = getattr(app_state, 'redis', None) if app_state else None
    organization_id = os.getenv("DEFAULT_ORGANIZATION_ID", "default-org")
    
    return SystemPerformanceMiddleware(
        redis_client=redis_client,
        db_session=db,
        organization_id=organization_id,
        collection_interval=int(os.getenv("SYSTEM_METRICS_INTERVAL", "60")),
        enable_realtime_redis=True
    )


# Note: get_agent function removed - PromptResponseAgent moved to orchestrator service