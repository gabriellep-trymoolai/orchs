"""Data models for real-time communication."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class MetricType(Enum):
	"""Types of metrics that can be streamed."""
	USER_ACTIVITY = "user_activity"
	ORGANIZATION_SUMMARY = "organization_summary"
	LLM_USAGE = "llm_usage"
	SYSTEM_PERFORMANCE = "system_performance"
	COST_TRACKING = "cost_tracking"
	ERROR_RATE = "error_rate"


class HealthStatus(Enum):
	"""System health status levels."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	CRITICAL = "critical"
	UNKNOWN = "unknown"


@dataclass
class StreamingMetric:
	"""Represents a metric that can be streamed in real-time."""
	metric_type: MetricType
	organization_id: str
	timestamp: datetime
	value: Any
	metadata: Dict[str, Any]
	user_id: Optional[str] = None
	department: Optional[str] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		return {
			"metric_type": self.metric_type.value,
			"organization_id": self.organization_id,
			"timestamp": self.timestamp.isoformat(),
			"value": self.value,
			"metadata": self.metadata,
			"user_id": self.user_id,
			"department": self.department
		}


@dataclass
class SystemHealthEvent:
	"""Represents a system health event."""
	status: HealthStatus
	organization_id: str
	component: str
	timestamp: datetime
	message: str
	metrics: Dict[str, Any]
	affected_services: List[str]
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary for JSON serialization."""
		return {
			"status": self.status.value,
			"organization_id": self.organization_id,
			"component": self.component,
			"timestamp": self.timestamp.isoformat(),
			"message": self.message,
			"metrics": self.metrics,
			"affected_services": self.affected_services
		}


@dataclass
class OrganizationChannel:
	"""Represents an organization-level channel for broadcasting."""
	organization_id: str
	channel_name: str
	created_at: datetime
	subscriber_count: int
	last_activity: datetime
	metadata: Dict[str, Any]
	
	@property
	def redis_channel(self) -> str:
		"""Get the Redis channel name."""
		return f"org:{self.organization_id}:{self.channel_name}"
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"organization_id": self.organization_id,
			"channel_name": self.channel_name,
			"created_at": self.created_at.isoformat(),
			"subscriber_count": self.subscriber_count,
			"last_activity": self.last_activity.isoformat(),
			"metadata": self.metadata
		}


@dataclass
class UserChannel:
	"""Represents a user-specific channel."""
	organization_id: str
	user_id: str
	channel_name: str
	created_at: datetime
	permissions: List[str]
	metadata: Dict[str, Any]
	
	@property
	def redis_channel(self) -> str:
		"""Get the Redis channel name."""
		return f"user:{self.organization_id}:{self.user_id}:{self.channel_name}"
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"organization_id": self.organization_id,
			"user_id": self.user_id,
			"channel_name": self.channel_name,
			"created_at": self.created_at.isoformat(),
			"permissions": self.permissions,
			"metadata": self.metadata
		}


@dataclass
class LLMStreamChunk:
	"""Represents a chunk of streaming LLM response."""
	request_id: str
	organization_id: str
	user_id: str
	chunk_index: int
	content: str
	is_final: bool
	timestamp: datetime
	model: str
	tokens_used: Optional[int] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"request_id": self.request_id,
			"organization_id": self.organization_id,
			"user_id": self.user_id,
			"chunk_index": self.chunk_index,
			"content": self.content,
			"is_final": self.is_final,
			"timestamp": self.timestamp.isoformat(),
			"model": self.model,
			"tokens_used": self.tokens_used
		}


@dataclass
class RealtimeMetricUpdate:
	"""Real-time metric update for dashboards."""
	organization_id: str
	metric_name: str
	current_value: float
	previous_value: float
	change_percent: float
	timestamp: datetime
	time_window: str  # e.g., "1h", "24h", "7d"
	breakdown: Optional[Dict[str, Any]] = None
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"organization_id": self.organization_id,
			"metric_name": self.metric_name,
			"current_value": self.current_value,
			"previous_value": self.previous_value,
			"change_percent": self.change_percent,
			"timestamp": self.timestamp.isoformat(),
			"time_window": self.time_window,
			"breakdown": self.breakdown
		}


@dataclass
class ConnectionInfo:
	"""Information about a real-time connection."""
	connection_id: str
	connection_type: str  # "sse" or "websocket"
	organization_id: str
	user_id: Optional[str]
	established_at: datetime
	last_activity: datetime
	channels: List[str]
	metadata: Dict[str, Any]
	
	def to_dict(self) -> Dict[str, Any]:
		"""Convert to dictionary."""
		return {
			"connection_id": self.connection_id,
			"connection_type": self.connection_type,
			"organization_id": self.organization_id,
			"user_id": self.user_id,
			"established_at": self.established_at.isoformat(),
			"last_activity": self.last_activity.isoformat(),
			"channels": self.channels,
			"metadata": self.metadata
		}