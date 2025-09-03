"""
Common API models for MoolAI services
These models provide a foundation for all API responses and requests
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from enum import Enum

# Type variable for generic responses
T = TypeVar('T')

# Standard Response Models
class ResponseMetadata(BaseModel):
	"""Standard metadata for all API responses"""
	request_id: str = Field(..., description="Unique request identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
	version: str = Field(default="1.0.0", description="API version")
	service: str = Field(..., description="Service that generated the response")
	organization_id: Optional[str] = Field(None, description="Organization context")
	user_id: Optional[str] = Field(None, description="User context")

class APIResponse(BaseModel, Generic[T]):
	"""Standard envelope for all API responses"""
	success: bool = Field(..., description="Whether the request was successful")
	data: Optional[T] = Field(None, description="Response data")
	metadata: ResponseMetadata = Field(..., description="Response metadata")
	message: str = Field(default="Operation completed successfully", description="Human-readable message")
	errors: Optional[List[str]] = Field(None, description="List of errors if any")

class HealthResponse(BaseModel):
	"""Standard health check response"""
	status: str = Field(default="healthy", description="Service health status")
	service: str = Field(..., description="Service name")
	version: str = Field(default="1.0.0", description="Service version")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	uptime_seconds: Optional[int] = Field(None, description="Service uptime in seconds")
	dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency health status")

# Authentication Models
class UserRole(str, Enum):
	"""User roles in the system"""
	SUPER_ADMIN = "super_admin"
	CONTROLLER_ADMIN = "controller_admin" 
	ORG_ADMIN = "org_admin"
	ORG_USER = "org_user"
	READONLY = "readonly"

class User(BaseModel):
	"""User model"""
	user_id: str = Field(..., description="Unique user identifier")
	username: str = Field(..., description="Username")
	email: str = Field(..., description="User email")
	organization_id: str = Field(..., description="Organization ID")
	role: UserRole = Field(..., description="User role")
	active: bool = Field(default=True, description="Whether user is active")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_login: Optional[datetime] = Field(None)

class LoginRequest(BaseModel):
	"""Login request model"""
	username: str = Field(..., description="Username or email")
	password: str = Field(..., description="User password")
	organization_id: Optional[str] = Field(None, description="Organization context")

class LoginResponse(BaseModel):
	"""Login response model"""
	access_token: str = Field(..., description="JWT access token")
	refresh_token: str = Field(..., description="JWT refresh token")
	token_type: str = Field(default="bearer", description="Token type")
	expires_in: int = Field(default=3600, description="Token expiration in seconds")
	user: User = Field(..., description="User information")

# Organization Models
class OrganizationStatus(str, Enum):
	"""Organization status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	SUSPENDED = "suspended"
	PROVISIONING = "provisioning"

class Organization(BaseModel):
	"""Organization model"""
	organization_id: str = Field(..., description="Unique organization identifier")
	name: str = Field(..., description="Organization name")
	status: OrganizationStatus = Field(..., description="Organization status")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	settings: Optional[Dict[str, Any]] = Field(None, description="Organization settings")

# Orchestrator Models
class OrchestratorStatus(str, Enum):
	"""Orchestrator deployment status"""
	RUNNING = "running"
	STOPPED = "stopped"
	PROVISIONING = "provisioning"
	ERROR = "error"
	MAINTENANCE = "maintenance"

class Orchestrator(BaseModel):
	"""Orchestrator instance model"""
	orchestrator_id: str = Field(..., description="Unique orchestrator identifier")
	organization_id: str = Field(..., description="Associated organization")
	status: OrchestratorStatus = Field(..., description="Orchestrator status")
	endpoint_url: str = Field(..., description="Orchestrator API endpoint")
	version: str = Field(..., description="Orchestrator version")
	deployed_at: datetime = Field(default_factory=datetime.utcnow)
	last_heartbeat: Optional[datetime] = Field(None, description="Last heartbeat timestamp")

# Prompt & Task Models
class PromptRequest(BaseModel):
	"""Prompt execution request"""
	prompt: str = Field(..., description="The prompt text")
	model: str = Field(default="gpt-4", description="LLM model to use")
	temperature: float = Field(default=0.7, ge=0, le=2, description="Model temperature")
	max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens")
	context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
	stream: bool = Field(default=False, description="Whether to stream response")

class PromptResponse(BaseModel):
	"""Prompt execution response"""
	prompt_id: str = Field(..., description="Unique prompt execution ID")
	response: str = Field(..., description="LLM response")
	model: str = Field(..., description="Model used")
	tokens_used: int = Field(..., description="Tokens consumed")
	cost: float = Field(..., description="Cost of execution")
	latency_ms: int = Field(..., description="Execution latency in milliseconds")
	created_at: datetime = Field(default_factory=datetime.utcnow)

class TaskRequest(BaseModel):
	"""Task execution request"""
	task_type: str = Field(..., description="Type of task to execute")
	parameters: Dict[str, Any] = Field(..., description="Task parameters")
	priority: int = Field(default=5, ge=1, le=10, description="Task priority")
	timeout_seconds: Optional[int] = Field(None, description="Task timeout")

class TaskStatus(str, Enum):
	"""Task execution status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

class Task(BaseModel):
	"""Task execution model"""
	task_id: str = Field(..., description="Unique task identifier")
	task_type: str = Field(..., description="Type of task")
	status: TaskStatus = Field(..., description="Task status")
	progress: float = Field(default=0.0, ge=0, le=100, description="Task progress percentage")
	result: Optional[Dict[str, Any]] = Field(None, description="Task result")
	error: Optional[str] = Field(None, description="Error message if failed")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	started_at: Optional[datetime] = Field(None)
	completed_at: Optional[datetime] = Field(None)

# Monitoring Models
class MetricType(str, Enum):
	"""Types of metrics"""
	COST = "cost"
	USAGE = "usage"
	PERFORMANCE = "performance"
	SECURITY = "security"
	SYSTEM = "system"

class TimeRange(BaseModel):
	"""Time range for queries"""
	start: datetime = Field(..., description="Start time")
	end: datetime = Field(..., description="End time")
	timezone: str = Field(default="UTC", description="Timezone")

class MetricQuery(BaseModel):
	"""Metric query parameters"""
	metrics: List[str] = Field(..., description="Metric names to query")
	time_range: TimeRange = Field(..., description="Time range")
	filters: Optional[Dict[str, Any]] = Field(None, description="Query filters")
	group_by: Optional[List[str]] = Field(None, description="Group by fields")
	aggregation: str = Field(default="sum", description="Aggregation method")
	limit: int = Field(default=100, ge=1, le=10000, description="Result limit")

class MetricPoint(BaseModel):
	"""Single metric data point"""
	timestamp: datetime = Field(..., description="Metric timestamp")
	value: float = Field(..., description="Metric value")
	labels: Optional[Dict[str, str]] = Field(None, description="Metric labels")

class Metric(BaseModel):
	"""Metric response"""
	name: str = Field(..., description="Metric name")
	unit: str = Field(..., description="Metric unit")
	data_points: List[MetricPoint] = Field(..., description="Metric data points")
	aggregation: str = Field(..., description="Aggregation method used")

# Configuration Models
class ConfigurationItem(BaseModel):
	"""Configuration item"""
	key: str = Field(..., description="Configuration key")
	value: Any = Field(..., description="Configuration value")
	type: str = Field(..., description="Value type")
	description: Optional[str] = Field(None, description="Configuration description")
	sensitive: bool = Field(default=False, description="Whether value is sensitive")

class Configuration(BaseModel):
	"""Configuration collection"""
	organization_id: str = Field(..., description="Organization ID")
	items: List[ConfigurationItem] = Field(..., description="Configuration items")
	version: str = Field(..., description="Configuration version")
	updated_at: datetime = Field(default_factory=datetime.utcnow)

# Real-time Models
class StreamSubscription(BaseModel):
	"""Real-time stream subscription"""
	subscribe_to: List[str] = Field(..., description="Metric types to subscribe to")
	organization_id: Optional[str] = Field(None, description="Organization filter")
	user_id: Optional[str] = Field(None, description="User filter")
	filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")

class StreamMessage(BaseModel):
	"""Real-time stream message"""
	event_type: str = Field(..., description="Event type")
	data: Dict[str, Any] = Field(..., description="Event data")
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	message_id: str = Field(..., description="Unique message ID")
	organization_id: Optional[str] = Field(None, description="Organization context")

# Export Models
class ExportFormat(str, Enum):
	"""Export format options"""
	CSV = "csv"
	JSON = "json"
	PARQUET = "parquet"
	XLSX = "xlsx"

class ExportRequest(BaseModel):
	"""Data export request"""
	export_type: ExportFormat = Field(..., description="Export format")
	metrics: List[str] = Field(..., description="Metrics to export")
	time_range: TimeRange = Field(..., description="Time range")
	filters: Optional[Dict[str, Any]] = Field(None, description="Export filters")
	destination: Optional[str] = Field(None, description="Export destination")

class ExportJob(BaseModel):
	"""Export job status"""
	job_id: str = Field(..., description="Unique job identifier")
	status: str = Field(..., description="Job status")
	progress: float = Field(default=0.0, description="Job progress percentage")
	created_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = Field(None)
	download_url: Optional[str] = Field(None, description="Download URL when ready")
	error: Optional[str] = Field(None, description="Error message if failed")

# Pagination Models
class PaginationParams(BaseModel):
	"""Pagination parameters"""
	page: int = Field(default=1, ge=1, description="Page number")
	page_size: int = Field(default=100, ge=1, le=1000, description="Items per page")

class PaginatedResponse(BaseModel, Generic[T]):
	"""Paginated response wrapper"""
	items: List[T] = Field(..., description="Page items")
	page: int = Field(..., description="Current page")
	page_size: int = Field(..., description="Items per page")
	total_items: int = Field(..., description="Total items available")
	total_pages: int = Field(..., description="Total pages available")
	has_next: bool = Field(..., description="Whether next page exists")
	has_prev: bool = Field(..., description="Whether previous page exists")