"""
API utilities for generating standardized responses
"""

import uuid
from datetime import datetime
from typing import Any, Optional, Dict, List
from .models import APIResponse, ResponseMetadata, HealthResponse

def generate_request_id() -> str:
	"""Generate a unique request ID"""
	return str(uuid.uuid4())

def create_metadata(
	service: str,
	organization_id: Optional[str] = None,
	user_id: Optional[str] = None,
	version: str = "1.0.0"
) -> ResponseMetadata:
	"""Create standard response metadata"""
	return ResponseMetadata(
		request_id=generate_request_id(),
		timestamp=datetime.utcnow(),
		version=version,
		service=service,
		organization_id=organization_id,
		user_id=user_id
	)

def create_success_response(
	data: Any = None,
	service: str = "moolai-api",
	message: str = "Operation completed successfully",
	organization_id: Optional[str] = None,
	user_id: Optional[str] = None
) -> APIResponse:
	"""Create a successful API response"""
	return APIResponse(
		success=True,
		data=data,
		metadata=create_metadata(service, organization_id, user_id),
		message=message
	)

def create_error_response(
	errors: List[str],
	service: str = "moolai-api",
	message: str = "Operation failed",
	organization_id: Optional[str] = None,
	user_id: Optional[str] = None
) -> APIResponse:
	"""Create an error API response"""
	return APIResponse(
		success=False,
		data=None,
		metadata=create_metadata(service, organization_id, user_id),
		message=message,
		errors=errors
	)

def create_health_response(
	service: str,
	status: str = "healthy",
	version: str = "1.0.0",
	uptime_seconds: Optional[int] = None,
	dependencies: Optional[Dict[str, str]] = None
) -> HealthResponse:
	"""Create a health check response"""
	return HealthResponse(
		status=status,
		service=service,
		version=version,
		timestamp=datetime.utcnow(),
		uptime_seconds=uptime_seconds,
		dependencies=dependencies
	)

# Placeholder data generators for foundation endpoints
def create_placeholder_user(user_id: str = "user-123", org_id: str = "org-123") -> Dict[str, Any]:
	"""Create placeholder user data"""
	return {
		"user_id": user_id,
		"username": "foundation-user",
		"email": "user@foundation.test",
		"organization_id": org_id,
		"role": "org_user",
		"active": True,
		"created_at": datetime.utcnow().isoformat(),
		"last_login": None
	}

def create_placeholder_organization(org_id: str = "org-123") -> Dict[str, Any]:
	"""Create placeholder organization data"""
	return {
		"organization_id": org_id,
		"name": "Foundation Organization",
		"status": "active",
		"created_at": datetime.utcnow().isoformat(),
		"updated_at": datetime.utcnow().isoformat(),
		"settings": {"api_enabled": True, "monitoring_enabled": True}
	}

def create_placeholder_orchestrator(orchestrator_id: str = "orch-123", org_id: str = "org-123") -> Dict[str, Any]:
	"""Create placeholder orchestrator data"""
	return {
		"orchestrator_id": orchestrator_id,
		"organization_id": org_id,
		"status": "running",
		"endpoint_url": f"https://api.foundation.test/orchestrators/{orchestrator_id}",
		"version": "1.0.0",
		"deployed_at": datetime.utcnow().isoformat(),
		"last_heartbeat": datetime.utcnow().isoformat()
	}

def create_placeholder_prompt_response(prompt_id: str = "prompt-123") -> Dict[str, Any]:
	"""Create placeholder prompt response"""
	return {
		"prompt_id": prompt_id,
		"response": "This is a foundation response. Functionality will be implemented later.",
		"model": "gpt-4",
		"tokens_used": 25,
		"cost": 0.001,
		"latency_ms": 150,
		"created_at": datetime.utcnow().isoformat()
	}

def create_placeholder_task(task_id: str = "task-123") -> Dict[str, Any]:
	"""Create placeholder task data"""
	return {
		"task_id": task_id,
		"task_type": "foundation-task",
		"status": "completed",
		"progress": 100.0,
		"result": {"message": "Foundation task completed successfully"},
		"error": None,
		"created_at": datetime.utcnow().isoformat(),
		"started_at": datetime.utcnow().isoformat(),
		"completed_at": datetime.utcnow().isoformat()
	}

def create_placeholder_metrics() -> List[Dict[str, Any]]:
	"""Create placeholder metrics data"""
	now = datetime.utcnow()
	return [
		{
			"name": "total_requests",
			"unit": "count",
			"data_points": [
				{
					"timestamp": now.isoformat(),
					"value": 1250.0,
					"labels": {"service": "foundation"}
				}
			],
			"aggregation": "sum"
		},
		{
			"name": "total_cost",
			"unit": "usd",
			"data_points": [
				{
					"timestamp": now.isoformat(),
					"value": 125.50,
					"labels": {"service": "foundation"}
				}
			],
			"aggregation": "sum"
		}
	]

def create_placeholder_configuration(org_id: str = "org-123") -> Dict[str, Any]:
	"""Create placeholder configuration"""
	return {
		"organization_id": org_id,
		"items": [
			{
				"key": "api_enabled",
				"value": True,
				"type": "boolean",
				"description": "Enable API access",
				"sensitive": False
			},
			{
				"key": "max_requests_per_minute",
				"value": 100,
				"type": "integer", 
				"description": "Rate limit for API requests",
				"sensitive": False
			}
		],
		"version": "1.0.0",
		"updated_at": datetime.utcnow().isoformat()
	}

def create_placeholder_export_job(job_id: str = "export-123") -> Dict[str, Any]:
	"""Create placeholder export job"""
	return {
		"job_id": job_id,
		"status": "completed",
		"progress": 100.0,
		"created_at": datetime.utcnow().isoformat(),
		"completed_at": datetime.utcnow().isoformat(),
		"download_url": f"https://api.foundation.test/exports/{job_id}/download",
		"error": None
	}