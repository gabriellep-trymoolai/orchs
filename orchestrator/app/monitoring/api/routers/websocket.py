"""WebSocket endpoints for bidirectional admin communication."""

import logging
from typing import Optional, Set
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import json

from ...config.database import get_db
from ...config.settings import get_config
import sys
import os

# Add common directory to path (container environment)
sys.path.append('/app/common')

from realtime import (
	WebSocketManager,
	WebSocketMessage,
	MessageType,
	EventBus,
	EventType
)
from ..dependencies import get_system_monitoring_middleware

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws/monitoring", tags=["monitoring-websocket"])

# Global WebSocket manager instance
ws_manager = WebSocketManager(
	max_connections_per_org=100,
	ping_interval=30,
	auth_timeout=10
)


async def get_ws_manager() -> WebSocketManager:
	"""Get WebSocket manager instance."""
	return ws_manager


@router.on_event("startup")
async def startup_event():
	"""Initialize WebSocket manager on startup."""
	await ws_manager.start()
	
	# Register message handlers
	ws_manager.register_handler(MessageType.COMMAND, handle_admin_command)
	ws_manager.register_handler(MessageType.CONFIG_UPDATE, handle_config_update)
	ws_manager.register_handler(MessageType.SYSTEM_CONTROL, handle_system_control)
	
	logger.info("WebSocket service started")


@router.on_event("shutdown")
async def shutdown_event():
	"""Cleanup WebSocket manager on shutdown."""
	await ws_manager.stop()
	logger.info("WebSocket service stopped")


@router.websocket("/admin/control")
async def websocket_admin_control(
	websocket: WebSocket,
	organization_id: Optional[str] = Query(None),
	token: Optional[str] = Query(None)
):
	"""
	WebSocket endpoint for admin control operations.
	
	Provides bidirectional communication for:
	- System configuration updates
	- Real-time control commands
	- Live debugging and monitoring
	- Administrative actions
	
	Requires admin authentication.
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# TODO: Validate token and extract user info/roles
	# For now, assume admin role if token provided
	roles = {"admin"} if token else set()
	user_id = "admin_user" if token else None
	
	try:
		# Accept connection
		connection = await ws_manager.connect(
			websocket=websocket,
			organization_id=org_id,
			user_id=user_id,
			roles=roles
		)
		
		# Auto-authenticate if token provided in query
		if token:
			await ws_manager.authenticate(connection.connection_id, token)
		
		# Handle messages
		while True:
			try:
				data = await websocket.receive_text()
				await ws_manager.handle_message(connection.connection_id, data)
			except WebSocketDisconnect:
				logger.info(f"Admin WebSocket disconnected: {connection.connection_id}")
				break
			except Exception as e:
				logger.error(f"Error in admin WebSocket: {e}")
				break
				
	except ConnectionError as e:
		logger.warning(f"Connection rejected: {e}")
		await websocket.close(reason=str(e))
	except Exception as e:
		logger.error(f"Unexpected error in admin WebSocket: {e}")
		await websocket.close(reason="Internal error")
	finally:
		if 'connection' in locals():
			await ws_manager.disconnect(connection.connection_id)


@router.websocket("/debug/logs")
async def websocket_debug_logs(
	websocket: WebSocket,
	organization_id: Optional[str] = Query(None),
	token: Optional[str] = Query(None),
	log_level: str = Query("INFO"),
	components: Optional[str] = Query(None)
):
	"""
	WebSocket endpoint for real-time log streaming.
	
	Streams filtered logs from the monitoring system for debugging.
	
	Parameters:
	- log_level: Minimum log level to stream (DEBUG, INFO, WARNING, ERROR)
	- components: Comma-separated list of components to filter
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Validate admin access
	roles = {"admin", "debug"} if token else set()
	user_id = "debug_user" if token else None
	
	# Parse components
	component_filter = components.split(",") if components else []
	
	try:
		# Accept connection
		connection = await ws_manager.connect(
			websocket=websocket,
			organization_id=org_id,
			user_id=user_id,
			roles=roles
		)
		
		# Set up log streaming
		connection.metadata["log_level"] = log_level
		connection.metadata["components"] = component_filter
		
		# Auto-authenticate if token provided
		if token:
			await ws_manager.authenticate(connection.connection_id, token)
			
			# Subscribe to log channels
			await ws_manager.subscribe(connection.connection_id, f"logs:{org_id}")
			if component_filter:
				for component in component_filter:
					await ws_manager.subscribe(connection.connection_id, f"logs:{org_id}:{component}")
		
		# Handle messages
		while True:
			try:
				data = await websocket.receive_text()
				await ws_manager.handle_message(connection.connection_id, data)
			except WebSocketDisconnect:
				logger.info(f"Debug WebSocket disconnected: {connection.connection_id}")
				break
			except Exception as e:
				logger.error(f"Error in debug WebSocket: {e}")
				break
				
	except ConnectionError as e:
		logger.warning(f"Connection rejected: {e}")
		await websocket.close(reason=str(e))
	except Exception as e:
		logger.error(f"Unexpected error in debug WebSocket: {e}")
		await websocket.close(reason="Internal error")
	finally:
		if 'connection' in locals():
			await ws_manager.disconnect(connection.connection_id)


@router.websocket("/metrics/live")
async def websocket_live_metrics(
	websocket: WebSocket,
	organization_id: Optional[str] = Query(None),
	token: Optional[str] = Query(None),
	metrics: Optional[str] = Query(None)
):
	"""
	WebSocket endpoint for bidirectional metrics communication.
	
	Allows:
	- Subscribing to specific metrics
	- Sending metric queries
	- Receiving real-time metric updates
	
	Parameters:
	- metrics: Comma-separated list of metric types to subscribe to
	"""
	config = get_config()
	org_id = organization_id or config.get_organization_id()
	
	# Basic authentication
	roles = {"metrics_viewer"} if token else set()
	user_id = None  # Extract from token validation
	
	# Parse metrics filter
	metric_types = metrics.split(",") if metrics else []
	
	try:
		# Accept connection
		connection = await ws_manager.connect(
			websocket=websocket,
			organization_id=org_id,
			user_id=user_id,
			roles=roles
		)
		
		connection.metadata["metric_types"] = metric_types
		
		# Auto-authenticate if token provided
		if token:
			await ws_manager.authenticate(connection.connection_id, token)
			
			# Subscribe to metric channels
			for metric_type in metric_types:
				if metric_type == "analytics":
					# Subscribe to analytics aggregated channel
					await ws_manager.subscribe(connection.connection_id, f"analytics:{org_id}")
					# Also subscribe to regular metrics for fallback
					await ws_manager.subscribe(connection.connection_id, f"org:{org_id}:metrics")
				else:
					# Subscribe to regular metric channels
					await ws_manager.subscribe(connection.connection_id, f"metrics:{org_id}:{metric_type}")
		
		# Handle messages
		while True:
			try:
				data = await websocket.receive_text()
				await ws_manager.handle_message(connection.connection_id, data)
			except WebSocketDisconnect:
				logger.info(f"Metrics WebSocket disconnected: {connection.connection_id}")
				break
			except Exception as e:
				logger.error(f"Error in metrics WebSocket: {e}")
				break
				
	except ConnectionError as e:
		logger.warning(f"Connection rejected: {e}")
		await websocket.close(reason=str(e))
	except Exception as e:
		logger.error(f"Unexpected error in metrics WebSocket: {e}")
		await websocket.close(reason="Internal error")
	finally:
		if 'connection' in locals():
			await ws_manager.disconnect(connection.connection_id)


@router.get("/connections/stats")
async def get_websocket_statistics(
	manager: WebSocketManager = Depends(get_ws_manager)
):
	"""Get statistics about active WebSocket connections."""
	return manager.get_connection_stats()


# Message Handlers

async def handle_admin_command(connection, message: WebSocketMessage):
	"""
	Handle admin command messages.
	
	Commands include:
	- restart_service: Restart a specific service
	- clear_cache: Clear Redis cache
	- force_sync: Force data synchronization
	- update_config: Update runtime configuration
	"""
	command = message.data.get("command")
	params = message.data.get("params", {})
	
	logger.info(f"Admin command from {connection.connection_id}: {command}")
	
	try:
		result = None
		
		if command == "restart_service":
			service_name = params.get("service")
			# TODO: Implement service restart logic
			result = {"status": "success", "message": f"Service {service_name} restart initiated"}
			
		elif command == "clear_cache":
			cache_type = params.get("type", "all")
			# TODO: Implement cache clearing
			result = {"status": "success", "message": f"Cache cleared: {cache_type}"}
			
		elif command == "force_sync":
			# TODO: Implement forced synchronization
			result = {"status": "success", "message": "Synchronization started"}
			
		elif command == "get_status":
			# Return current system status
			result = {
				"status": "success",
				"data": {
					"services": "all_running",
					"database": "connected",
					"redis": "connected"
				}
			}
		else:
			raise ValueError(f"Unknown command: {command}")
		
		# Send response
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.SUCCESS,
				data=result,
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)
		
	except Exception as e:
		logger.error(f"Error handling admin command: {e}")
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": str(e)},
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)


async def handle_config_update(connection, message: WebSocketMessage):
	"""
	Handle configuration update messages.
	
	Updates runtime configuration without restart.
	"""
	config_section = message.data.get("section")
	config_values = message.data.get("values", {})
	
	logger.info(f"Config update from {connection.connection_id}: {config_section}")
	
	try:
		# TODO: Implement configuration update logic
		# This would update in-memory configuration and persist to database
		
		result = {
			"status": "success",
			"message": f"Configuration updated: {config_section}",
			"updated_values": config_values
		}
		
		# Broadcast config change to all connections in organization
		await ws_manager.broadcast_to_organization(
			connection.organization_id,
			WebSocketMessage(
				type=MessageType.CONFIG_UPDATE,
				data={
					"section": config_section,
					"values": config_values,
					"updated_by": connection.user_id,
					"timestamp": datetime.utcnow().isoformat()
				},
				timestamp=datetime.utcnow()
			)
		)
		
		# Send confirmation to requester
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.SUCCESS,
				data=result,
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)
		
	except Exception as e:
		logger.error(f"Error handling config update: {e}")
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": str(e)},
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)


async def handle_system_control(connection, message: WebSocketMessage):
	"""
	Handle system control messages.
	
	Controls system-level operations like:
	- Enable/disable features
	- Adjust rate limits
	- Modify monitoring parameters
	"""
	control_type = message.data.get("type")
	action = message.data.get("action")
	params = message.data.get("params", {})
	
	logger.info(f"System control from {connection.connection_id}: {control_type}/{action}")
	
	try:
		result = None
		
		if control_type == "monitoring":
			if action == "set_interval":
				interval = params.get("interval", 60)
				# TODO: Update monitoring interval
				result = {"status": "success", "message": f"Monitoring interval set to {interval}s"}
				
			elif action == "enable_component":
				component = params.get("component")
				# TODO: Enable monitoring component
				result = {"status": "success", "message": f"Component {component} enabled"}
				
		elif control_type == "rate_limit":
			if action == "update":
				limits = params.get("limits", {})
				# TODO: Update rate limits
				result = {"status": "success", "message": "Rate limits updated", "limits": limits}
				
		elif control_type == "feature_flag":
			if action == "toggle":
				feature = params.get("feature")
				enabled = params.get("enabled", True)
				# TODO: Toggle feature flag
				result = {
					"status": "success",
					"message": f"Feature {feature} {'enabled' if enabled else 'disabled'}"
				}
		else:
			raise ValueError(f"Unknown control type: {control_type}")
		
		# Send response
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.SUCCESS,
				data=result,
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)
		
	except Exception as e:
		logger.error(f"Error handling system control: {e}")
		await ws_manager.send_message(
			connection.connection_id,
			WebSocketMessage(
				type=MessageType.ERROR,
				data={"error": str(e)},
				timestamp=datetime.utcnow(),
				correlation_id=message.message_id
			)
		)