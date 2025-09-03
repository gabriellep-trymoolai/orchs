"""Event bus for distributing real-time events across services using Redis PubSub."""

import asyncio
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class EventType(Enum):
	"""Types of events in the system."""
	# Metrics events
	METRICS_USER_UPDATE = "metrics.user.update"
	METRICS_ORG_UPDATE = "metrics.org.update"
	METRICS_REALTIME = "metrics.realtime"
	
	# System events
	SYSTEM_HEALTH = "system.health"
	SYSTEM_ALERT = "system.alert"
	SYSTEM_PERFORMANCE = "system.performance"
	
	# LLM events
	LLM_REQUEST_START = "llm.request.start"
	LLM_REQUEST_COMPLETE = "llm.request.complete"
	LLM_REQUEST_ERROR = "llm.request.error"
	LLM_STREAM_CHUNK = "llm.stream.chunk"
	
	# Admin events
	ADMIN_COMMAND = "admin.command"
	ADMIN_CONFIG_UPDATE = "admin.config.update"
	ADMIN_USER_ACTION = "admin.user.action"
	
	# Connection events
	CONNECTION_ESTABLISHED = "connection.established"
	CONNECTION_CLOSED = "connection.closed"


@dataclass
class Event:
	"""Represents an event in the system."""
	type: EventType
	organization_id: str
	data: Dict[str, Any]
	timestamp: datetime
	source: str
	event_id: Optional[str] = None
	user_id: Optional[str] = None
	correlation_id: Optional[str] = None
	
	def to_json(self) -> str:
		"""Convert event to JSON string."""
		event_dict = {
			"type": self.type.value,
			"organization_id": self.organization_id,
			"data": self.data,
			"timestamp": self.timestamp.isoformat(),
			"source": self.source,
			"event_id": self.event_id,
			"user_id": self.user_id,
			"correlation_id": self.correlation_id
		}
		return json.dumps(event_dict)
	
	@classmethod
	def from_json(cls, json_str: str) -> "Event":
		"""Create event from JSON string."""
		event_dict = json.loads(json_str)
		return cls(
			type=EventType(event_dict["type"]),
			organization_id=event_dict["organization_id"],
			data=event_dict["data"],
			timestamp=datetime.fromisoformat(event_dict["timestamp"]),
			source=event_dict["source"],
			event_id=event_dict.get("event_id"),
			user_id=event_dict.get("user_id"),
			correlation_id=event_dict.get("correlation_id")
		)


class EventBus:
	"""
	Event bus for distributing events across services using Redis PubSub.
	Supports multi-tenant isolation through channel namespacing.
	"""
	
	def __init__(
		self,
		redis_client: redis.Redis,
		service_name: str,
		organization_id: Optional[str] = None
	):
		"""
		Initialize event bus.
		
		Args:
			redis_client: Redis async client
			service_name: Name of the service using this bus
			organization_id: Optional organization ID for tenant isolation
		"""
		self.redis = redis_client
		self.service_name = service_name
		self.organization_id = organization_id
		self.pubsub = None
		self.listeners: Dict[EventType, Set[Callable]] = {}
		self._running = False
		self._listen_task = None
		self._subscribed_channels: Set[str] = set()
		
	async def start(self):
		"""Start the event bus and begin listening for events."""
		if self._running:
			return
		
		self._running = True
		self.pubsub = self.redis.pubsub()
		
		# Start listening task
		self._listen_task = asyncio.create_task(self._listen_loop())
		
		# Subscribe to global system channel
		await self.subscribe_channel("system:global")
		
		# Subscribe to organization channel if specified
		if self.organization_id:
			await self.subscribe_channel(f"org:{self.organization_id}")
		
		logger.info(f"Event bus started for service: {self.service_name}")
	
	async def stop(self):
		"""Stop the event bus and cleanup."""
		self._running = False
		
		if self._listen_task:
			self._listen_task.cancel()
			try:
				await self._listen_task
			except asyncio.CancelledError:
				pass
		
		if self.pubsub:
			await self.pubsub.unsubscribe()
			await self.pubsub.close()
		
		logger.info(f"Event bus stopped for service: {self.service_name}")
	
	async def subscribe_channel(self, channel: str):
		"""
		Subscribe to a Redis channel.
		
		Args:
			channel: Channel name to subscribe to
		"""
		if not self.pubsub:
			raise RuntimeError("Event bus not started")
		
		if channel not in self._subscribed_channels:
			await self.pubsub.subscribe(channel)
			self._subscribed_channels.add(channel)
			logger.debug(f"Subscribed to channel: {channel}")
	
	async def unsubscribe_channel(self, channel: str):
		"""
		Unsubscribe from a Redis channel.
		
		Args:
			channel: Channel name to unsubscribe from
		"""
		if not self.pubsub:
			return
		
		if channel in self._subscribed_channels:
			await self.pubsub.unsubscribe(channel)
			self._subscribed_channels.discard(channel)
			logger.debug(f"Unsubscribed from channel: {channel}")
	
	async def publish(self, event: Event):
		"""
		Publish an event to appropriate channels.
		
		Args:
			event: Event to publish
		"""
		# Determine channels based on event
		channels = self._get_channels_for_event(event)
		
		# Publish to all relevant channels
		event_json = event.to_json()
		for channel in channels:
			await self.redis.publish(channel, event_json)
			logger.debug(f"Published {event.type.value} to {channel}")
	
	async def publish_to_user(
		self,
		organization_id: str,
		user_id: str,
		event_type: EventType,
		data: Dict[str, Any]
	):
		"""
		Publish an event to a specific user.
		
		Args:
			organization_id: Organization ID
			user_id: User ID
			event_type: Type of event
			data: Event data
		"""
		event = Event(
			type=event_type,
			organization_id=organization_id,
			user_id=user_id,
			data=data,
			timestamp=datetime.utcnow(),
			source=self.service_name
		)
		
		# Publish to user-specific channel
		channel = f"user:{organization_id}:{user_id}"
		await self.redis.publish(channel, event.to_json())
		logger.debug(f"Published {event_type.value} to user {user_id}")
	
	async def publish_to_organization(
		self,
		organization_id: str,
		event_type: EventType,
		data: Dict[str, Any]
	):
		"""
		Publish an event to all users in an organization.
		
		Args:
			organization_id: Organization ID
			event_type: Type of event
			data: Event data
		"""
		event = Event(
			type=event_type,
			organization_id=organization_id,
			data=data,
			timestamp=datetime.utcnow(),
			source=self.service_name
		)
		
		# Publish to organization channel
		channel = f"org:{organization_id}"
		await self.redis.publish(channel, event.to_json())
		logger.debug(f"Published {event_type.value} to org {organization_id}")
	
	def register_listener(self, event_type: EventType, callback: Callable):
		"""
		Register a callback for a specific event type.
		
		Args:
			event_type: Type of event to listen for
			callback: Async function to call when event occurs
		"""
		if event_type not in self.listeners:
			self.listeners[event_type] = set()
		self.listeners[event_type].add(callback)
		logger.debug(f"Registered listener for {event_type.value}")
	
	def unregister_listener(self, event_type: EventType, callback: Callable):
		"""
		Unregister a callback for an event type.
		
		Args:
			event_type: Type of event
			callback: Callback to remove
		"""
		if event_type in self.listeners:
			self.listeners[event_type].discard(callback)
	
	async def _listen_loop(self):
		"""Main loop for listening to Redis PubSub messages."""
		try:
			async for message in self.pubsub.listen():
				if not self._running:
					break
				
				# Skip non-message types
				if message["type"] not in ["message", "pmessage"]:
					continue
				
				try:
					# Parse event from message
					event = Event.from_json(message["data"])
					
					# Call registered listeners
					if event.type in self.listeners:
						for callback in self.listeners[event.type]:
							try:
								asyncio.create_task(callback(event))
							except Exception as e:
								logger.error(f"Error in event listener: {e}")
								
				except json.JSONDecodeError:
					logger.warning(f"Invalid event data received: {message['data']}")
				except Exception as e:
					logger.error(f"Error processing event: {e}")
					
		except asyncio.CancelledError:
			logger.debug("Listen loop cancelled")
		except Exception as e:
			logger.error(f"Error in listen loop: {e}")
	
	def _get_channels_for_event(self, event: Event) -> Set[str]:
		"""
		Determine which channels an event should be published to.
		
		Args:
			event: Event to analyze
			
		Returns:
			Set of channel names
		"""
		channels = set()
		
		# Organization channel
		if event.organization_id:
			channels.add(f"org:{event.organization_id}")
		
		# User-specific channel
		if event.user_id and event.organization_id:
			channels.add(f"user:{event.organization_id}:{event.user_id}")
		
		# System-wide events
		if event.type in [EventType.SYSTEM_HEALTH, EventType.SYSTEM_ALERT]:
			channels.add("system:global")
		
		# Admin events to admin channel
		if event.type.value.startswith("admin."):
			channels.add(f"admin:{event.organization_id}")
		
		return channels
	
	async def wait_for_event(
		self,
		event_type: EventType,
		timeout: Optional[float] = None,
		filter_func: Optional[Callable[[Event], bool]] = None
	) -> Optional[Event]:
		"""
		Wait for a specific event type to occur.
		
		Args:
			event_type: Type of event to wait for
			timeout: Optional timeout in seconds
			filter_func: Optional function to filter events
			
		Returns:
			Event if received, None if timeout
		"""
		event_future = asyncio.Future()
		
		async def event_handler(event: Event):
			if filter_func and not filter_func(event):
				return
			if not event_future.done():
				event_future.set_result(event)
		
		self.register_listener(event_type, event_handler)
		
		try:
			return await asyncio.wait_for(event_future, timeout=timeout)
		except asyncio.TimeoutError:
			return None
		finally:
			self.unregister_listener(event_type, event_handler)