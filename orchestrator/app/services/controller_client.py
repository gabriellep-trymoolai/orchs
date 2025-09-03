"""
Controller Registration Client
Handles orchestrator registration and communication with the central controller
"""

import os
import httpx
import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class ControllerClient:
	"""Client for communicating with the MoolAI Controller."""
	
	def __init__(self):
		# Check if we're in development mode
		self.environment = os.getenv("ENVIRONMENT", "production")
		self.is_development = self.environment.lower() == "development"
		
		self.controller_url = os.getenv("CONTROLLER_URL", "http://controller:8002")
		self.orchestrator_id = os.getenv("ORCHESTRATOR_ID")
		self.organization_id = os.getenv("ORGANIZATION_ID")
		self.orchestrator_name = os.getenv("ORCHESTRATOR_NAME", f"Orchestrator {self.organization_id}")
		
		# Service URLs
		self.internal_url = os.getenv("INTERNAL_URL", f"http://{self.orchestrator_id}:8000")
		self.database_url = os.getenv("DATABASE_URL")
		self.redis_url = os.getenv("REDIS_URL")
		
		# Container information
		self.container_id = os.getenv("HOSTNAME")  # Docker sets this to container ID
		self.image_name = os.getenv("IMAGE_NAME", "moolai/orchestrator")
		
		# Registration state
		self.is_registered = False
		self.registration_retries = 0
		self.max_retries = 10
		self.retry_delay = 5  # seconds
		
		# Heartbeat configuration
		self.heartbeat_interval = 30  # seconds
		self.heartbeat_task: Optional[asyncio.Task] = None
		
		self._validate_configuration()
	
	def _validate_configuration(self):
		"""Validate required configuration for controller registration."""
		required_vars = {
			"ORCHESTRATOR_ID": self.orchestrator_id,
			"ORGANIZATION_ID": self.organization_id,
			"DATABASE_URL": self.database_url
		}
		
		missing_vars = [name for name, value in required_vars.items() if not value]
		
		if missing_vars:
			raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
		
		logger.info(f"Controller client configured for orchestrator: {self.orchestrator_id}")
	
	async def register_with_controller(self) -> bool:
		"""
		Register this orchestrator with the controller.
		Retries automatically if registration fails.
		"""
		# Use same database for monitoring (Phoenix handles LLM monitoring)
		monitoring_database_url = self.database_url
		
		registration_data = {
			"orchestrator_id": self.orchestrator_id,
			"organization_id": self.organization_id,
			"name": self.orchestrator_name,
			"internal_url": self.internal_url,
			"database_url": self.database_url,
			"redis_url": self.redis_url,
			"container_id": self.container_id,
			"image_name": self.image_name,
			"monitoring": {
				"enabled": True,
				"embedded": True,
				"database_url": monitoring_database_url,
				"endpoints": {
					"metrics": f"{self.internal_url}/api/v1/system/metrics",
					"health": f"{self.internal_url}/api/v1/system/status",
					"stream": f"{self.internal_url}/api/v1/stream",
					"websocket": f"{self.internal_url.replace('http', 'ws')}/ws"
				}
			},
			"environment_variables": {
				"ORCHESTRATOR_ID": self.orchestrator_id,
				"ORGANIZATION_ID": self.organization_id,
				"INTERNAL_URL": self.internal_url,
				"MONITORING_DATABASE_URL": monitoring_database_url
			}
		}
		
		for attempt in range(self.max_retries):
			try:
				logger.info(f"Attempting to register with controller (attempt {attempt + 1}/{self.max_retries})")
				logger.info(f"Controller URL: {self.controller_url}")
				logger.info(f"Orchestrator ID: {self.orchestrator_id}")
				
				async with httpx.AsyncClient(timeout=30.0) as client:
					response = await client.post(
						f"{self.controller_url}/api/v1/internal/orchestrators/register",
						json=registration_data
					)
					
					if response.status_code == 200:
						result = response.json()
						logger.info(f"Successfully registered with controller: {result}")
						self.is_registered = True
						self.registration_retries = 0
						
						# Start heartbeat task
						await self._start_heartbeat()
						
						return True
					else:
						logger.error(f"Registration failed with status {response.status_code}: {response.text}")
				
			except httpx.RequestError as e:
				logger.error(f"Network error during registration: {e}")
			except Exception as e:
				logger.error(f"Unexpected error during registration: {e}")
			
			# Wait before retry (unless it's the last attempt)
			if attempt < self.max_retries - 1:
				logger.info(f"Retrying registration in {self.retry_delay} seconds...")
				await asyncio.sleep(self.retry_delay)
			
			self.registration_retries += 1
		
		# All registration attempts failed
		logger.error(f"Failed to register with controller after {self.max_retries} attempts")
		return False
	
	async def send_heartbeat(self) -> bool:
		"""Send heartbeat to controller to maintain registration."""
		# Skip heartbeat in development mode if not registered
		if self.is_development and not self.is_registered:
			logger.debug("Development mode: Skipping heartbeat (not registered)")
			return True  # Return success to avoid warnings
		
		if not self.is_registered:
			logger.warning("Cannot send heartbeat - not registered with controller")
			return False
		
		heartbeat_data = {
			"orchestrator_id": self.orchestrator_id,
			"status": "active",
			"health_status": "healthy"
		}
		
		try:
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.post(
					f"{self.controller_url}/api/v1/internal/orchestrators/heartbeat",
					json=heartbeat_data
				)
				
				if response.status_code == 200:
					logger.debug(f"Heartbeat sent successfully")
					return True
				else:
					logger.warning(f"Heartbeat failed with status {response.status_code}: {response.text}")
					return False
		
		except httpx.RequestError as e:
			logger.error(f"Network error during heartbeat: {e}")
			return False
		except Exception as e:
			logger.error(f"Unexpected error during heartbeat: {e}")
			return False
	
	async def deregister_from_controller(self) -> bool:
		"""Deregister this orchestrator from the controller."""
		if not self.is_registered:
			logger.info("Not registered with controller - skipping deregistration")
			return True
		
		try:
			logger.info("Deregistering from controller...")
			
			# Stop heartbeat task
			await self._stop_heartbeat()
			
			async with httpx.AsyncClient(timeout=10.0) as client:
				response = await client.delete(
					f"{self.controller_url}/api/v1/internal/orchestrators/{self.orchestrator_id}/deregister"
				)
				
				if response.status_code == 200:
					logger.info("Successfully deregistered from controller")
					self.is_registered = False
					return True
				else:
					logger.error(f"Deregistration failed with status {response.status_code}: {response.text}")
					return False
		
		except httpx.RequestError as e:
			logger.error(f"Network error during deregistration: {e}")
			return False
		except Exception as e:
			logger.error(f"Unexpected error during deregistration: {e}")
			return False
	
	async def _start_heartbeat(self):
		"""Start periodic heartbeat task."""
		if self.heartbeat_task and not self.heartbeat_task.done():
			logger.warning("Heartbeat task already running")
			return
		
		logger.info(f"Starting heartbeat task (interval: {self.heartbeat_interval}s)")
		self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
	
	async def _stop_heartbeat(self):
		"""Stop heartbeat task."""
		if self.heartbeat_task:
			logger.info("Stopping heartbeat task")
			self.heartbeat_task.cancel()
			try:
				await self.heartbeat_task
			except asyncio.CancelledError:
				pass
			self.heartbeat_task = None
	
	async def _heartbeat_loop(self):
		"""Main heartbeat loop."""
		try:
			while True:
				await asyncio.sleep(self.heartbeat_interval)
				
				success = await self.send_heartbeat()
				if not success and not self.is_development:
					# Only warn in production mode
					logger.warning("Heartbeat failed - controller may be unavailable")
					# Could implement re-registration logic here if needed
				elif not success and self.is_development:
					# In development, just log at debug level
					logger.debug("Heartbeat skipped in development mode")
		
		except asyncio.CancelledError:
			logger.info("Heartbeat loop cancelled")
		except Exception as e:
			logger.error(f"Heartbeat loop error: {e}")
	
	async def check_controller_connectivity(self) -> bool:
		"""Check if the controller is reachable."""
		try:
			async with httpx.AsyncClient(timeout=5.0) as client:
				response = await client.get(f"{self.controller_url}/api/v1/internal/health")
				return response.status_code == 200
		except:
			return False
	
	def get_registration_status(self) -> Dict[str, Any]:
		"""Get current registration status information."""
		return {
			"is_registered": self.is_registered,
			"orchestrator_id": self.orchestrator_id,
			"organization_id": self.organization_id,
			"controller_url": self.controller_url,
			"registration_retries": self.registration_retries,
			"heartbeat_active": self.heartbeat_task is not None and not self.heartbeat_task.done()
		}


# Global controller client instance
controller_client: Optional[ControllerClient] = None

def get_controller_client() -> ControllerClient:
	"""Get the global controller client instance."""
	global controller_client
	if controller_client is None:
		controller_client = ControllerClient()
	return controller_client

async def ensure_controller_registration() -> bool:
	"""
	Ensure orchestrator is registered with controller.
	This function should be called during application startup.
	In development mode, controller registration is optional.
	In production mode, orchestrator will not start if registration fails.
	"""
	client = get_controller_client()
	
	# Development mode bypass
	if client.is_development:
		logger.info("DEVELOPMENT MODE: Controller registration is optional")
		
		# Check if controller is reachable
		logger.info("Checking controller connectivity...")
		controller_reachable = await client.check_controller_connectivity()
		
		if not controller_reachable:
			logger.warning(f"Controller not reachable at {client.controller_url}")
			logger.warning("Running in development mode without controller")
			# Mark as registered to allow heartbeat bypass
			client.is_registered = False
			return True  # Allow startup to continue in development
		
		logger.info("Controller is reachable - proceeding with registration")
		
		# Attempt registration but don't fail if it doesn't work
		registration_success = await client.register_with_controller()
		
		if not registration_success:
			logger.warning("Failed to register with controller in development mode")
			logger.warning("Continuing without controller registration")
			return True  # Allow startup to continue
		
		logger.info("Successfully registered with controller")
		return True
	
	# Production mode - original strict behavior
	logger.info("PRODUCTION MODE: Controller registration is required")
	
	# Check if controller is reachable
	logger.info("Checking controller connectivity...")
	controller_reachable = await client.check_controller_connectivity()
	
	if not controller_reachable:
		logger.error(f"Controller not reachable at {client.controller_url}")
		logger.error("Orchestrator cannot start without controller connection")
		raise RuntimeError("Controller connection required but not available")
	
	logger.info("Controller is reachable - proceeding with registration")
	
	# Attempt registration
	registration_success = await client.register_with_controller()
	
	if not registration_success:
		logger.error("Failed to register with controller")
		logger.error("Orchestrator cannot start without successful registration")
		raise RuntimeError("Controller registration required but failed")
	
	logger.info("Successfully registered with controller")
	return True

async def cleanup_controller_registration():
	"""
	Clean up controller registration during shutdown.
	This function should be called during application shutdown.
	"""
	client = get_controller_client()
	await client.deregister_from_controller()