"""SSE streaming endpoints for real-time metrics."""

import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ...config.database import get_db
from ...config.settings import get_config
import sys
import os

# Add common directory to path (container environment)
sys.path.append('/app/common')

from realtime import SSEManager, EventBus, EventType
from ..dependencies import get_system_monitoring_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])

# Global SSE manager instance
sse_manager = SSEManager(heartbeat_interval=30)
event_bus = None


async def get_sse_manager() -> SSEManager:
	"""Get SSE manager instance."""
	return sse_manager


async def get_event_bus() -> EventBus:
	"""Get event bus instance."""
	global event_bus
	if not event_bus:
		config = get_config()
		if config.redis_url:
			redis_client = await redis.from_url(config.redis_url)
			event_bus = EventBus(
				redis_client=redis_client,
				service_name="monitoring",
				organization_id=config.get_organization_id()
			)
			await event_bus.start()
	return event_bus


@router.on_event("startup")
async def startup_event():
	"""Initialize SSE manager on startup."""
	await sse_manager.start()
	logger.info("SSE streaming service started")


@router.on_event("shutdown")
async def shutdown_event():
	"""Cleanup SSE manager on shutdown."""
	await sse_manager.stop()
	if event_bus:
		await event_bus.stop()
	logger.info("SSE streaming service stopped")


@router.get("/metrics/users/{user_id}")
async def stream_user_metrics(
	user_id: str,
	request: Request,
	organization_id: Optional[str] = Query(None),
	department: Optional[str] = Query(None),
	db: AsyncSession = Depends(get_db),
	manager: SSEManager = Depends(get_sse_manager)
):
	"""
	Stream real-time metrics for a specific user.
	
	This endpoint provides Server-Sent Events (SSE) stream of user metrics including:
	- Real-time LLM usage
	- Cost tracking
	- Performance metrics
	- Error rates
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Create SSE connection
	connection = await manager.connect(
		organization_id=org_id,
		user_id=user_id,
		channels={f"metrics:user:{user_id}"},
		metadata={"department": department}
	)
	
	async def generate():
		"""Generate SSE stream."""
		try:
			# Register event listeners
			bus = await get_event_bus()
			
			async def on_user_metrics(event):
				"""Handle user metrics events."""
				if event.user_id == user_id:
					await manager.publish(
						f"metrics:user:{user_id}",
						"user_metrics",
						event.data
					)
			
			bus.register_listener(EventType.METRICS_USER_UPDATE, on_user_metrics)
			bus.register_listener(EventType.METRICS_REALTIME, on_user_metrics)
			
			# Subscribe to user-specific channel
			await bus.subscribe_channel(f"user:{org_id}:{user_id}")
			
			# Stream messages
			async for message in manager.stream(connection.connection_id):
				yield message
				
				# Check if client disconnected
				if await request.is_disconnected():
					break
					
		except asyncio.CancelledError:
			logger.info(f"User metrics stream cancelled for {user_id}")
		except Exception as e:
			logger.error(f"Error in user metrics stream: {e}")
		finally:
			# Cleanup
			await manager.disconnect(connection.connection_id)
			if bus:
				bus.unregister_listener(EventType.METRICS_USER_UPDATE, on_user_metrics)
				bus.unregister_listener(EventType.METRICS_REALTIME, on_user_metrics)
	
	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no"  # Disable nginx buffering
		}
	)


@router.get("/metrics/organization")
async def stream_organization_metrics(
	request: Request,
	organization_id: Optional[str] = Query(None),
	time_window: str = Query("1h", regex="^(1h|6h|24h|7d|30d)$"),
	db: AsyncSession = Depends(get_db),
	manager: SSEManager = Depends(get_sse_manager)
):
	"""
	Stream real-time organization-level metrics.
	
	Provides aggregated metrics for the entire organization including:
	- Total usage across all users
	- Cost breakdown by department
	- System performance metrics
	- Active users and sessions
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Create SSE connection
	connection = await manager.connect(
		organization_id=org_id,
		channels={f"metrics:org:{org_id}"},
		metadata={"time_window": time_window}
	)
	
	async def generate():
		"""Generate SSE stream."""
		try:
			# Register event listeners
			bus = await get_event_bus()
			
			async def on_org_metrics(event):
				"""Handle organization metrics events."""
				if event.organization_id == org_id:
					await manager.publish(
						f"metrics:org:{org_id}",
						"org_metrics",
						event.data
					)
			
			bus.register_listener(EventType.METRICS_ORG_UPDATE, on_org_metrics)
			
			# Subscribe to organization channel
			await bus.subscribe_channel(f"org:{org_id}")
			
			# Send initial metrics snapshot
			initial_metrics = await get_organization_snapshot(db, org_id, time_window)
			await manager.publish(
				f"metrics:org:{org_id}",
				"metrics_snapshot",
				initial_metrics
			)
			
			# Stream messages
			async for message in manager.stream(connection.connection_id):
				yield message
				
				# Check if client disconnected
				if await request.is_disconnected():
					break
					
		except asyncio.CancelledError:
			logger.info(f"Organization metrics stream cancelled for {org_id}")
		except Exception as e:
			logger.error(f"Error in organization metrics stream: {e}")
		finally:
			# Cleanup
			await manager.disconnect(connection.connection_id)
			if bus:
				bus.unregister_listener(EventType.METRICS_ORG_UPDATE, on_org_metrics)
	
	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no"
		}
	)


@router.get("/system/health")
async def stream_system_health(
	request: Request,
	organization_id: Optional[str] = Query(None),
	components: Optional[str] = Query(None, description="Comma-separated list of components"),
	db: AsyncSession = Depends(get_db),
	manager: SSEManager = Depends(get_sse_manager),
	system_monitoring = Depends(get_system_monitoring_middleware)
):
	"""
	Stream real-time system health and performance metrics.
	
	Monitors:
	- Service health status
	- CPU and memory usage
	- Database performance
	- Redis connectivity
	- API response times
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Parse components filter
	component_list = components.split(",") if components else []
	
	# Create SSE connection
	connection = await manager.connect(
		organization_id=org_id,
		channels={f"health:org:{org_id}"},
		metadata={"components": component_list}
	)
	
	async def generate():
		"""Generate SSE stream."""
		try:
			# Register event listeners
			bus = await get_event_bus()
			
			async def on_health_event(event):
				"""Handle system health events."""
				if event.organization_id == org_id:
					# Filter by components if specified
					if component_list:
						component = event.data.get("component")
						if component and component not in component_list:
							return
					
					await manager.publish(
						f"health:org:{org_id}",
						"health_update",
						event.data
					)
			
			bus.register_listener(EventType.SYSTEM_HEALTH, on_health_event)
			bus.register_listener(EventType.SYSTEM_PERFORMANCE, on_health_event)
			bus.register_listener(EventType.SYSTEM_ALERT, on_health_event)
			
			# Start periodic health checks
			health_task = asyncio.create_task(
				periodic_health_check(manager, system_monitoring, org_id, connection.connection_id)
			)
			
			# Stream messages
			async for message in manager.stream(connection.connection_id):
				yield message
				
				# Check if client disconnected
				if await request.is_disconnected():
					break
					
		except asyncio.CancelledError:
			logger.info(f"System health stream cancelled for {org_id}")
		except Exception as e:
			logger.error(f"Error in system health stream: {e}")
		finally:
			# Cleanup
			health_task.cancel()
			await manager.disconnect(connection.connection_id)
			if bus:
				bus.unregister_listener(EventType.SYSTEM_HEALTH, on_health_event)
				bus.unregister_listener(EventType.SYSTEM_PERFORMANCE, on_health_event)
				bus.unregister_listener(EventType.SYSTEM_ALERT, on_health_event)
	
	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no"
		}
	)


@router.get("/llm/responses/{request_id}")
async def stream_llm_response(
	request_id: str,
	request: Request,
	user_id: Optional[str] = Query(None),
	organization_id: Optional[str] = Query(None),
	manager: SSEManager = Depends(get_sse_manager)
):
	"""
	Stream LLM response chunks in real-time.
	
	This endpoint streams the LLM response as it's being generated,
	allowing for immediate display in the UI.
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Create SSE connection
	connection = await manager.connect(
		organization_id=org_id,
		user_id=user_id,
		channels={f"llm:request:{request_id}"},
		metadata={"request_id": request_id}
	)
	
	async def generate():
		"""Generate SSE stream."""
		try:
			# Register event listeners
			bus = await get_event_bus()
			
			async def on_llm_chunk(event):
				"""Handle LLM stream chunks."""
				if event.data.get("request_id") == request_id:
					await manager.publish(
						f"llm:request:{request_id}",
						"llm_chunk",
						event.data
					)
			
			bus.register_listener(EventType.LLM_STREAM_CHUNK, on_llm_chunk)
			bus.register_listener(EventType.LLM_REQUEST_COMPLETE, on_llm_chunk)
			bus.register_listener(EventType.LLM_REQUEST_ERROR, on_llm_chunk)
			
			# Subscribe to request-specific channel
			await bus.subscribe_channel(f"llm:request:{request_id}")
			
			# Stream messages
			async for message in manager.stream(connection.connection_id):
				yield message
				
				# Check if client disconnected
				if await request.is_disconnected():
					break
					
		except asyncio.CancelledError:
			logger.info(f"LLM response stream cancelled for {request_id}")
		except Exception as e:
			logger.error(f"Error in LLM response stream: {e}")
		finally:
			# Cleanup
			await manager.disconnect(connection.connection_id)
			if bus:
				bus.unregister_listener(EventType.LLM_STREAM_CHUNK, on_llm_chunk)
				bus.unregister_listener(EventType.LLM_REQUEST_COMPLETE, on_llm_chunk)
				bus.unregister_listener(EventType.LLM_REQUEST_ERROR, on_llm_chunk)
	
	return StreamingResponse(
		generate(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no"
		}
	)


@router.get("/connections/stats")
async def get_connection_statistics(
	manager: SSEManager = Depends(get_sse_manager)
):
	"""Get statistics about active SSE connections."""
	return manager.get_connection_stats()


# Helper functions

async def get_organization_snapshot(db: AsyncSession, org_id: str, time_window: str) -> dict:
	"""Get initial organization metrics snapshot."""
	# Calculate time range
	now = datetime.utcnow()
	time_ranges = {
		"1h": timedelta(hours=1),
		"6h": timedelta(hours=6),
		"24h": timedelta(days=1),
		"7d": timedelta(days=7),
		"30d": timedelta(days=30)
	}
	start_time = now - time_ranges.get(time_window, timedelta(hours=1))
	
	# TODO: Query actual metrics from database
	# For now, return sample data
	return {
		"organization_id": org_id,
		"timestamp": now.isoformat(),
		"time_window": time_window,
		"metrics": {
			"total_requests": 0,
			"total_cost": 0.0,
			"active_users": 0,
			"error_rate": 0.0,
			"avg_latency": 0.0
		}
	}


async def periodic_health_check(
	manager: SSEManager,
	system_monitoring,
	org_id: str,
	connection_id: str,
	interval: int = 30
):
	"""Periodically check and broadcast system health."""
	while True:
		try:
			await asyncio.sleep(interval)
			
			# Get current system metrics
			metrics = await system_monitoring.get_current_metrics()
			
			# Determine health status
			health_status = "healthy"
			if metrics.get("cpu_usage", 0) > 80:
				health_status = "degraded"
			if metrics.get("memory_usage", 0) > 90:
				health_status = "critical"
			
			# Publish health update
			await manager.publish(
				f"health:org:{org_id}",
				"health_status",
				{
					"status": health_status,
					"metrics": metrics,
					"timestamp": datetime.utcnow().isoformat()
				}
			)
			
		except asyncio.CancelledError:
			break
		except Exception as e:
			logger.error(f"Error in periodic health check: {e}")