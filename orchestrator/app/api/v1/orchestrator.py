"""
Orchestrator API Endpoints - Foundation
All endpoints return healthy responses for integration foundation
"""

from fastapi import APIRouter, Query, Path, Depends, HTTPException, Request
from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

# Import common models and utilities
import sys
import os
# Add common directory to path (located at /app/common in container)
sys.path.append('/app/common')

from common.api.models import (
	APIResponse, HealthResponse, PromptRequest, PromptResponse, 
	TaskRequest, Task, Configuration, ConfigurationItem, User,
	PaginationParams, PaginatedResponse
)
from common.api.utils import (
	create_success_response, create_health_response,
	create_placeholder_prompt_response, create_placeholder_task,
	create_placeholder_configuration, create_placeholder_user,
	generate_request_id
)

# Import controller client
from ...services.controller_client import get_controller_client

# Import database and models
from ...db.database import get_db
from ...models.user import User as UserModel
from ...core.logging_config import get_logger

# Import agent models from main_response.py
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
main_response_path = os.path.join(current_dir, '../../agents/Prompt Response')
sys.path.insert(0, main_response_path)

try:
    from main_response import QueryRequest as AgentPromptRequest
finally:
    if main_response_path in sys.path:
        sys.path.remove(main_response_path)

router = APIRouter(prefix="/orchestrators/{organization_id}", tags=["Orchestrator"])

# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health", response_model=HealthResponse)
async def get_orchestrator_health(organization_id: str = Path(...)):
	"""Orchestrator service health check"""
	return create_health_response(
		service=f"orchestrator-{organization_id}",
		dependencies={
			"database": "healthy", 
			"redis": "healthy",
			"controller": "healthy"
		}
	)

@router.get("/controller-status", response_model=APIResponse)
async def get_controller_registration_status(organization_id: str = Path(...)):
	"""Get controller registration status for this orchestrator"""
	try:
		client = get_controller_client()
		status = client.get_registration_status()
		
		return create_success_response(
			data=status,
			service=f"orchestrator-{organization_id}",
			message="Controller registration status retrieved successfully",
			organization_id=organization_id
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to get controller status: {str(e)}")

# ============================================================================
# AUTHENTICATION & USER MANAGEMENT
# ============================================================================

@router.get("/users", response_model=APIResponse[PaginatedResponse[User]])
async def list_users(
	organization_id: str = Path(...),
	pagination: PaginationParams = Depends(),
	role: Optional[str] = Query(None),
	active: Optional[bool] = Query(None),
	db: AsyncSession = Depends(get_db)
):
	"""List users in organization from database"""
	logger = get_logger(__name__)
	
	try:
		# Build query with filters
		query = select(UserModel)
		
		if active is not None:
			query = query.where(UserModel.is_active == active)
			
		# Count total items
		count_query = select(func.count(UserModel.user_id))
		if active is not None:
			count_query = count_query.where(UserModel.is_active == active)
			
		total_result = await db.execute(count_query)
		total_items = total_result.scalar() or 0
		
		# Calculate pagination
		total_pages = (total_items + pagination.page_size - 1) // pagination.page_size
		offset = (pagination.page - 1) * pagination.page_size
		
		# Get paginated results
		query = query.offset(offset).limit(pagination.page_size).order_by(UserModel.username)
		result = await db.execute(query)
		db_users = result.scalars().all()
		
		# Convert to API User model format
		users = []
		for db_user in db_users:
			user_dict = {
				"user_id": db_user.user_id,
				"username": db_user.username,
				"email": db_user.email,
				"organization_id": organization_id,
				"role": "org_admin" if db_user.is_admin else "org_user",
				"active": db_user.is_active,
				"created_at": db_user.created_at.isoformat() if db_user.created_at else None,
				"last_login": db_user.last_login.isoformat() if db_user.last_login else None,
				# Additional fields for frontend display
				"full_name": db_user.full_name,
				"department": db_user.department,
				"job_title": db_user.job_title,
				"preferred_model": db_user.preferred_model
			}
			users.append(user_dict)
		
		paginated_data = PaginatedResponse(
			items=users,
			page=pagination.page,
			page_size=pagination.page_size,
			total_items=total_items,
			total_pages=total_pages,
			has_next=pagination.page < total_pages,
			has_prev=pagination.page > 1
		)
		
		logger.info(f"📋 Listed {len(users)} users for organization {organization_id}")
		
		return create_success_response(
			data=paginated_data,
			service=f"orchestrator-{organization_id}",
			message="Users listed successfully",
			organization_id=organization_id
		)
			
	except Exception as e:
		logger.error(f"❌ Failed to list users: {e}")
		raise HTTPException(status_code=500, detail=f"Failed to list users: {str(e)}")

@router.post("/users", response_model=APIResponse[User])
async def create_user(
	organization_id: str = Path(...),
	username: str = Query(...),
	email: str = Query(...),
	role: str = Query(...)
):
	"""Create new user in organization"""
	user_data = create_placeholder_user(generate_request_id(), organization_id)
	user_data.update({
		"username": username,
		"email": email,
		"role": role
	})
	
	return create_success_response(
		data=user_data,
		service=f"orchestrator-{organization_id}",
		message="User created successfully",
		organization_id=organization_id
	)

@router.get("/users/{user_id}", response_model=APIResponse[User])
async def get_user(
	organization_id: str = Path(...),
	user_id: str = Path(...)
):
	"""Get user details"""
	user_data = create_placeholder_user(user_id, organization_id)
	
	return create_success_response(
		data=user_data,
		service=f"orchestrator-{organization_id}",
		message="User details retrieved successfully",
		organization_id=organization_id,
		user_id=user_id
	)

@router.put("/users/{user_id}", response_model=APIResponse[User])
async def update_user(
	organization_id: str = Path(...),
	user_id: str = Path(...),
	email: Optional[str] = Query(None),
	role: Optional[str] = Query(None),
	active: Optional[bool] = Query(None)
):
	"""Update user details"""
	user_data = create_placeholder_user(user_id, organization_id)
	if email:
		user_data["email"] = email
	if role:
		user_data["role"] = role
	if active is not None:
		user_data["active"] = active
	
	return create_success_response(
		data=user_data,
		service=f"orchestrator-{organization_id}",
		message="User updated successfully",
		organization_id=organization_id,
		user_id=user_id
	)

@router.delete("/users/{user_id}", response_model=APIResponse)
async def delete_user(
	organization_id: str = Path(...),
	user_id: str = Path(...)
):
	"""Delete user from organization"""
	return create_success_response(
		data={"user_id": user_id, "status": "deleted"},
		service=f"orchestrator-{organization_id}",
		message="User deleted successfully",
		organization_id=organization_id,
		user_id=user_id
	)

# ============================================================================
# PROMPT EXECUTION
# ============================================================================

@router.post("/prompts", response_model=APIResponse[PromptResponse])
async def execute_prompt(
	prompt_request: PromptRequest,
	organization_id: str = Path(...),
	request: Request = None,
	db: AsyncSession = Depends(get_db)
):
	"""Execute LLM prompt using the integrated prompt-response agent"""
	try:
		# Get the prompt agent from app state
		if not hasattr(request.app.state, 'prompt_agent') or request.app.state.prompt_agent is None:
			raise HTTPException(status_code=503, detail="Prompt-Response Agent not available")
		
		agent = request.app.state.prompt_agent
		
		# Convert API request to agent request
		agent_request = AgentPromptRequest(
			query=prompt_request.prompt,  # Map 'prompt' to 'query' for QueryRequest
			session_id=f"{organization_id}_api_session"
		)
		
		# Process the prompt
		agent_response = await agent.process_prompt(agent_request, db)
		
		# Convert agent response to API response
		response_data = {
			"prompt_id": agent_response.prompt_id,
			"response": agent_response.response,
			"model": agent_response.model,
			"tokens_used": agent_response.total_tokens,
			"cost": agent_response.cost,
			"latency_ms": agent_response.latency_ms,
			"created_at": agent_response.timestamp,
			"from_cache": getattr(agent_response, "from_cache", False),
			"cache_similarity": getattr(agent_response, "cache_similarity", None)
		}
		
		return create_success_response(
			data=response_data,
			service=f"orchestrator-{organization_id}",
			message="Prompt executed successfully using integrated agent",
			organization_id=organization_id
		)
		
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Prompt execution failed: {str(e)}")

@router.get("/prompts/{prompt_id}", response_model=APIResponse[PromptResponse])
async def get_prompt_result(
	organization_id: str = Path(...),
	prompt_id: str = Path(...),
	db: AsyncSession = Depends(get_db)
):
	"""Get prompt execution result"""
	try:
		from sqlalchemy import select
		from ...models.prompt_execution import PromptExecution
		
		query = select(PromptExecution).where(
			PromptExecution.prompt_id == prompt_id,
			PromptExecution.organization_id == organization_id
		)
		
		result = await db.execute(query)
		prompt_execution = result.scalar_one_or_none()
		
		if not prompt_execution:
			raise HTTPException(status_code=404, detail="Prompt execution not found")
		
		response_data = {
			"prompt_id": prompt_execution.prompt_id,
			"response": prompt_execution.response_text,
			"model": prompt_execution.model,
			"tokens_used": prompt_execution.total_tokens,
			"cost": prompt_execution.cost,
			"latency_ms": prompt_execution.latency_ms,
			"created_at": prompt_execution.timestamp
		}
		
		return create_success_response(
			data=response_data,
			service=f"orchestrator-{organization_id}",
			message="Prompt result retrieved successfully",
			organization_id=organization_id
		)
		
	except HTTPException:
		raise
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to retrieve prompt result: {str(e)}")

@router.get("/prompts/{prompt_id}/stream")
async def stream_prompt_response(
	organization_id: str = Path(...),
	prompt_id: str = Path(...)
):
	"""Stream prompt response (Server-Sent Events)"""
	# This would typically be a streaming response
	# For foundation, return a simple message
	return {
		"message": "Streaming endpoint foundation - SSE implementation pending",
		"prompt_id": prompt_id,
		"organization_id": organization_id
	}

@router.get("/prompts", response_model=APIResponse[PaginatedResponse[PromptResponse]])
async def list_prompts(
	organization_id: str = Path(...),
	pagination: PaginationParams = Depends(),
	user_id: Optional[str] = Query(None),
	model: Optional[str] = Query(None),
	start_date: Optional[datetime] = Query(None),
	end_date: Optional[datetime] = Query(None)
):
	"""List prompt execution history"""
	total_items = 150
	total_pages = (total_items + pagination.page_size - 1) // pagination.page_size
	
	prompts = [
		create_placeholder_prompt_response(f"prompt-{i:03d}")
		for i in range(1, min(pagination.page_size + 1, total_items + 1))
	]
	
	paginated_data = PaginatedResponse(
		items=prompts,
		page=pagination.page,
		page_size=pagination.page_size,
		total_items=total_items,
		total_pages=total_pages,
		has_next=pagination.page < total_pages,
		has_prev=pagination.page > 1
	)
	
	return create_success_response(
		data=paginated_data,
		service=f"orchestrator-{organization_id}",
		message="Prompt history retrieved successfully",
		organization_id=organization_id
	)

# ============================================================================
# TASK EXECUTION
# ============================================================================

@router.post("/tasks", response_model=APIResponse[Task])
async def create_task(
	task_request: TaskRequest,
	organization_id: str = Path(...)
):
	"""Create and execute AI agent task"""
	task_id = generate_request_id()
	task_data = create_placeholder_task(task_id)
	task_data.update({
		"task_type": task_request.task_type,
		"priority": task_request.priority,
		"status": "pending"
	})
	
	return create_success_response(
		data=task_data,
		service=f"orchestrator-{organization_id}",
		message="Task created successfully",
		organization_id=organization_id
	)

@router.get("/tasks/{task_id}", response_model=APIResponse[Task])
async def get_task(
	organization_id: str = Path(...),
	task_id: str = Path(...)
):
	"""Get task status and result"""
	task_data = create_placeholder_task(task_id)
	
	return create_success_response(
		data=task_data,
		service=f"orchestrator-{organization_id}",
		message="Task details retrieved successfully",
		organization_id=organization_id
	)

@router.put("/tasks/{task_id}/cancel", response_model=APIResponse[Task])
async def cancel_task(
	organization_id: str = Path(...),
	task_id: str = Path(...)
):
	"""Cancel running task"""
	task_data = create_placeholder_task(task_id)
	task_data["status"] = "cancelled"
	
	return create_success_response(
		data=task_data,
		service=f"orchestrator-{organization_id}",
		message="Task cancelled successfully",
		organization_id=organization_id
	)

@router.get("/tasks", response_model=APIResponse[PaginatedResponse[Task]])
async def list_tasks(
	organization_id: str = Path(...),
	pagination: PaginationParams = Depends(),
	status: Optional[str] = Query(None),
	task_type: Optional[str] = Query(None),
	user_id: Optional[str] = Query(None)
):
	"""List tasks for organization"""
	total_items = 75
	total_pages = (total_items + pagination.page_size - 1) // pagination.page_size
	
	tasks = [
		create_placeholder_task(f"task-{i:03d}")
		for i in range(1, min(pagination.page_size + 1, total_items + 1))
	]
	
	if status:
		for task in tasks:
			task["status"] = status
	
	paginated_data = PaginatedResponse(
		items=tasks,
		page=pagination.page,
		page_size=pagination.page_size,
		total_items=total_items,
		total_pages=total_pages,
		has_next=pagination.page < total_pages,
		has_prev=pagination.page > 1
	)
	
	return create_success_response(
		data=paginated_data,
		service=f"orchestrator-{organization_id}",
		message="Tasks listed successfully",
		organization_id=organization_id
	)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

@router.get("/config", response_model=APIResponse[Configuration])
async def get_configuration(
	organization_id: str = Path(...)
):
	"""Get organization configuration"""
	config_data = create_placeholder_configuration(organization_id)
	
	return create_success_response(
		data=config_data,
		service=f"orchestrator-{organization_id}",
		message="Configuration retrieved successfully",
		organization_id=organization_id
	)

@router.put("/config", response_model=APIResponse[Configuration])
async def update_configuration(
	items: List[ConfigurationItem],
	organization_id: str = Path(...)
):
	"""Update organization configuration"""
	config_data = create_placeholder_configuration(organization_id)
	config_data["items"] = [item.dict() for item in items]
	config_data["updated_at"] = datetime.utcnow().isoformat()
	
	return create_success_response(
		data=config_data,
		service=f"orchestrator-{organization_id}",
		message="Configuration updated successfully",
		organization_id=organization_id
	)

@router.post("/config/validate", response_model=APIResponse)
async def validate_configuration(
	items: List[ConfigurationItem],
	organization_id: str = Path(...)
):
	"""Validate configuration before applying"""
	validation_result = {
		"valid": True,
		"warnings": [],
		"errors": [],
		"validated_items": len(items)
	}
	
	return create_success_response(
		data=validation_result,
		service=f"orchestrator-{organization_id}",
		message="Configuration validation completed",
		organization_id=organization_id
	)

# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

@router.get("/api-keys", response_model=APIResponse[List[dict]])
async def list_api_keys(
	organization_id: str = Path(...),
	active: Optional[bool] = Query(None)
):
	"""List API keys for organization"""
	api_keys = [
		{
			"key_id": f"key-{i:03d}",
			"name": f"API Key {i}",
			"active": True,
			"created_at": datetime.utcnow().isoformat(),
			"last_used": datetime.utcnow().isoformat(),
			"permissions": ["prompts", "tasks", "monitoring"]
		}
		for i in range(1, 6)
	]
	
	if active is not None:
		api_keys = [key for key in api_keys if key["active"] == active]
	
	return create_success_response(
		data=api_keys,
		service=f"orchestrator-{organization_id}",
		message="API keys listed successfully",
		organization_id=organization_id
	)

@router.post("/api-keys", response_model=APIResponse[dict])
async def create_api_key(
	organization_id: str = Path(...),
	name: str = Query(...),
	permissions: List[str] = Query(...)
):
	"""Create new API key"""
	api_key_data = {
		"key_id": generate_request_id(),
		"name": name,
		"key": f"moolai_{'*' * 32}",  # Masked for security
		"active": True,
		"permissions": permissions,
		"created_at": datetime.utcnow().isoformat()
	}
	
	return create_success_response(
		data=api_key_data,
		service=f"orchestrator-{organization_id}",
		message="API key created successfully",
		organization_id=organization_id
	)

@router.delete("/api-keys/{key_id}", response_model=APIResponse)
async def delete_api_key(
	organization_id: str = Path(...),
	key_id: str = Path(...)
):
	"""Delete API key"""
	return create_success_response(
		data={"key_id": key_id, "status": "deleted"},
		service=f"orchestrator-{organization_id}",
		message="API key deleted successfully",
		organization_id=organization_id
	)

# ============================================================================
# AGENT MANAGEMENT
# ============================================================================

@router.get("/agents", response_model=APIResponse[List[dict]])
async def list_agents(
	organization_id: str = Path(...),
	agent_type: Optional[str] = Query(None)
):
	"""List available AI agents"""
	agents = [
		{
			"agent_id": "prompt-response-agent",
			"name": "Prompt Response Agent",
			"type": "prompt_response",
			"status": "active",
			"description": "Handles simple prompt-response interactions"
		},
		{
			"agent_id": "task-agent",
			"name": "Task Agent", 
			"type": "task",
			"status": "active",
			"description": "Executes complex multi-step tasks"
		},
		{
			"agent_id": "evaluation-agent",
			"name": "Evaluation Agent",
			"type": "evaluation", 
			"status": "active",
			"description": "Evaluates and scores AI responses"
		},
		{
			"agent_id": "rag-agent",
			"name": "Agentic RAG Agent",
			"type": "rag",
			"status": "active",
			"description": "Retrieval-augmented generation with agentic capabilities"
		}
	]
	
	if agent_type:
		agents = [agent for agent in agents if agent["type"] == agent_type]
	
	return create_success_response(
		data=agents,
		service=f"orchestrator-{organization_id}",
		message="Agents listed successfully",
		organization_id=organization_id
	)

@router.get("/agents/{agent_id}/status", response_model=APIResponse[dict])
async def get_agent_status(
	organization_id: str = Path(...),
	agent_id: str = Path(...)
):
	"""Get agent status and health"""
	agent_status = {
		"agent_id": agent_id,
		"status": "active",
		"health": "healthy",
		"last_heartbeat": datetime.utcnow().isoformat(),
		"active_tasks": 3,
		"completed_tasks": 1247,
		"error_rate": 0.02
	}
	
	return create_success_response(
		data=agent_status,
		service=f"orchestrator-{organization_id}",
		message="Agent status retrieved successfully",
		organization_id=organization_id
	)