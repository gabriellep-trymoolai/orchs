"""WebSocket manager for bidirectional real-time communication."""

import asyncio
import json
import logging
import uuid
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(Enum):
	"""Types of WebSocket messages."""
	# Control messages
	PING = "ping"
	PONG = "pong"
	SUBSCRIBE = "subscribe"
	UNSUBSCRIBE = "unsubscribe"
	AUTH = "auth"
	
	# Admin messages
	COMMAND = "command"
	CONFIG_UPDATE = "config_update"
	SYSTEM_CONTROL = "system_control"
	
	# Data messages
	METRICS = "metrics"
	LOGS = "logs"
	ALERTS = "alerts"
	
	# Response messages
	SUCCESS = "success"
	ERROR = "error"
	DATA = "data"


@dataclass
class WebSocketConnection:
	"""Represents a WebSocket connection."""
	connection_id: str
	websocket: WebSocket
	organization_id: str
	user_id: Optional[str]
	roles: Set[str]
	channels: Set[str]
	created_at: datetime
	last_activity: datetime
	metadata: Dict[str, Any]
	is_authenticated: bool = False


@dataclass
class WebSocketMessage:
	"""Represents a WebSocket message."""
	type: MessageType
	data: Any
	timestamp: datetime
	message_id: Optional[str] = None
	correlation_id: Optional[str] = None
	
	def to_json(self) -> str:
		"""Convert message to JSON."""
		return json.dumps({
			"type": self.type.value,
			"data": self.data,
			"timestamp": self.timestamp.isoformat(),
			"message_id": self.message_id,
			"correlation_id": self.correlation_id
		})
	
	@classmethod
	def from_json(cls, json_str: str) -> "WebSocketMessage":
		"""Create message from JSON."""
		msg_dict = json.loads(json_str)
		return cls(
			type=MessageType(msg_dict["type"]),
			data=msg_dict["data"],
			timestamp=datetime.fromisoformat(msg_dict["timestamp"]),
			message_id=msg_dict.get("message_id"),
			correlation_id=msg_dict.get("correlation_id")
		)


class WebSocketManager:
	"""
	Manages WebSocket connections for bidirectional communication.
	Primarily used for admin features requiring real-time control.
	"""
	
	def __init__(
		self,
		max_connections_per_org: int = 100,
		ping_interval: int = 30,
		auth_timeout: int = 10
	):
		"""
		Initialize WebSocket Manager.
		
		Args:
			max_connections_per_org: Maximum connections per organization
			ping_interval: Seconds between ping messages
			auth_timeout: Seconds to wait for authentication
		"""
		self.connections: Dict[str, WebSocketConnection] = {}
		self.org_connections: Dict[str, Set[str]] = {}
		self.channel_connections: Dict[str, Set[str]] = {}
		self.message_handlers: Dict[MessageType, Callable] = {}
		self.max_connections_per_org = max_connections_per_org
		self.ping_interval = ping_interval
		self.auth_timeout = auth_timeout
		self._running = False
		self._ping_task = None
		
	async def start(self):
		"""Start the WebSocket manager."""
		self._running = True
		self._ping_task = asyncio.create_task(self._ping_loop())
		logger.info("WebSocket Manager started")
		
	async def stop(self):
		"""Stop the WebSocket manager."""
		self._running = False
		if self._ping_task:
			self._ping_task.cancel()
			try:
				await self._ping_task
			except asyncio.CancelledError:
				pass
		
		# Close all connections
		for conn_id in list(self.connections.keys()):
			await self.disconnect(conn_id)
		
		logger.info("WebSocket Manager stopped")
	
	async def connect(
		self,
		websocket: WebSocket,
		organization_id: str,
		user_id: Optional[str] = None,
		roles: Optional[Set[str]] = None
	) -> WebSocketConnection:
		"""
		Accept a new WebSocket connection.
		
		Args:
			websocket: FastAPI WebSocket instance
			organization_id: Organization identifier
			user_id: Optional user identifier
			roles: User roles for authorization
			
		Returns:
			WebSocketConnection object
			
		Raises:
			ConnectionError: If organization has too many connections
		"""
		# Check connection limit
		if organization_id in self.org_connections:
			if len(self.org_connections[organization_id]) >= self.max_connections_per_org:
				raise ConnectionError(f"Organization {organization_id} has reached connection limit")
		
		# Accept WebSocket connection
		await websocket.accept()
		
		# Create connection object
		connection_id = str(uuid.uuid4())
		now = datetime.utcnow()
		
		connection = WebSocketConnection(
			connection_id=connection_id,
			websocket=websocket,
			organization_id=organization_id,
			user_id=user_id,
			roles=roles or set(),
			channels=set(),
			created_at=now,
			last_activity=now,
			metadata={},
			is_authenticated=False
		)
		
		# Store connection
		self.connections[connection_id] = connection
		
		# Track organization connections
		if organization_id not in self.org_connections:
			self.org_connections[organization_id] = set()
		self.org_connections[organization_id].add(connection_id)
		
		# Send connection confirmation
		await self.send_message(
			connection_id,
			WebSocketMessage(
				type=MessageType.SUCCESS,
				data={
					"message": "Connected",
					"connection_id": connection_id,
					"require_auth": True
				},
				timestamp=now
			)
		)
		
		logger.info(f"WebSocket connection established: {connection_id} for org {organization_id}")
		
		# Start authentication timeout
		asyncio.create_task(self._auth_timeout_check(connection_id))
		
		return connection
	
	async def disconnect(self, connection_id: str, reason: str = "Normal closure"):
		"""
		Disconnect a WebSocket connection.
		
		Args:
			connection_id: Connection identifier
			reason: Disconnection reason
		"""
		if connection_id not in self.connections:
			return
		
		connection = self.connections[connection_id]
		
		# Unsubscribe from all channels
		for channel in list(connection.channels):
			await self.unsubscribe(connection_id, channel)
		
		# Remove from organization tracking
		if connection.organization_id in self.org_connections:
			self.org_connections[connection.organization_id].discard(connection_id)
			if not self.org_connections[connection.organization_id]:
				del self.org_connections[connection.organization_id]
		
		# Try to close WebSocket
		try:
			await connection.websocket.close(reason=reason)
		except Exception as e:
			logger.debug(f"Error closing WebSocket: {e}")
		
		# Remove connection
		del self.connections[connection_id]
		
		logger.info(f"WebSocket connection closed: {connection_id} - {reason}")
	
	async def authenticate(
		self,
		connection_id: str,
		auth_token: str
	) -> bool:
		"""
		Authenticate a WebSocket connection.
		
		Args:
			connection_id: Connection identifier
			auth_token: Authentication token
			
		Returns:
			True if authentication successful
		"""
		if connection_id not in self.connections:
			return False
		
		connection = self.connections[connection_id]
		
		# TODO: Implement actual token validation
		# For now, just check if token is provided
		if auth_token:
			connection.is_authenticated = True
			connection.last_activity = datetime.utcnow()
			
			# Subscribe to default channels based on roles
			if "admin" in connection.roles:
				await self.subscribe(connection_id, f"admin:{connection.organization_id}")
			
			await self.subscribe(connection_id, f"org:{connection.organization_id}")
			
			if connection.user_id:
				await self.subscribe(connection_id, f"user:{connection.organization_id}:{connection.user_id}")
			
			# Send success message
			await self.send_message(
				connection_id,
				WebSocketMessage(
					type=MessageType.SUCCESS,
					data={"message": "Authenticated", "channels": list(connection.channels)},
					timestamp=datetime.utcnow()
				)
			)
			
			logger.info(f"WebSocket {connection_id} authenticated")
			return True
		
		return False
	
	async def subscribe(self, connection_id: str, channel: str) -> bool:
		"""
		Subscribe a connection to a channel.
		
		Args:
			connection_id: Connection identifier
			channel: Channel name
			
		Returns:
			True if subscription successful
		"""
		if connection_id not in self.connections:
			return False
		
		connection = self.connections[connection_id]
		
		# Check authorization for channel
		if not self._is_authorized_for_channel(connection, channel):
			logger.warning(f"Connection {connection_id} not authorized for channel {channel}")
			return False
		
		# Add to channel
		connection.channels.add(channel)
		
		if channel not in self.channel_connections:
			self.channel_connections[channel] = set()
		self.channel_connections[channel].add(connection_id)
		
		logger.debug(f"WebSocket {connection_id} subscribed to {channel}")
		return True
	
	async def unsubscribe(self, connection_id: str, channel: str):
		"""
		Unsubscribe a connection from a channel.
		
		Args:
			connection_id: Connection identifier
			channel: Channel name
		"""
		if connection_id not in self.connections:
			return
		
		connection = self.connections[connection_id]
		connection.channels.discard(channel)
		
		if channel in self.channel_connections:
			self.channel_connections[channel].discard(connection_id)
			if not self.channel_connections[channel]:
				del self.channel_connections[channel]
		
		logger.debug(f"WebSocket {connection_id} unsubscribed from {channel}")
	
	async def send_message(
		self,
		connection_id: str,
		message: WebSocketMessage
	) -> bool:
		"""
		Send a message to a specific connection.
		
		Args:
			connection_id: Connection identifier
			message: Message to send
			
		Returns:
			True if message sent successfully
		"""
		if connection_id not in self.connections:
			return False
		
		connection = self.connections[connection_id]
		
		try:
			await connection.websocket.send_text(message.to_json())
			connection.last_activity = datetime.utcnow()
			return True
		except Exception as e:
			logger.error(f"Error sending message to {connection_id}: {e}")
			await self.disconnect(connection_id, "Send error")
			return False
	
	async def broadcast_to_channel(
		self,
		channel: str,
		message: WebSocketMessage
	):
		"""
		Broadcast a message to all connections in a channel.
		
		Args:
			channel: Channel name
			message: Message to broadcast
		"""
		if channel not in self.channel_connections:
			return
		
		# Send to all connections in channel
		disconnected = []
		for conn_id in self.channel_connections[channel]:
			if not await self.send_message(conn_id, message):
				disconnected.append(conn_id)
		
		# Clean up disconnected connections
		for conn_id in disconnected:
			await self.disconnect(conn_id, "Broadcast error")
		
		logger.debug(f"Broadcast to {channel}: {message.type.value}")
	
	async def broadcast_to_organization(
		self,
		organization_id: str,
		message: WebSocketMessage
	):
		"""
		Broadcast a message to all connections in an organization.
		
		Args:
			organization_id: Organization identifier
			message: Message to broadcast
		"""
		channel = f"org:{organization_id}"
		await self.broadcast_to_channel(channel, message)
	
	async def handle_message(
		self,
		connection_id: str,
		raw_message: str
	):
		"""
		Handle an incoming message from a connection.
		
		Args:
			connection_id: Connection identifier
			raw_message: Raw message string
		"""
		if connection_id not in self.connections:
			return
		
		connection = self.connections[connection_id]
		connection.last_activity = datetime.utcnow()
		
		try:
			# Parse message
			message = WebSocketMessage.from_json(raw_message)
			
			# Handle authentication
			if message.type == MessageType.AUTH:
				auth_token = message.data.get("token")
				await self.authenticate(connection_id, auth_token)
				return
			
			# Check if authenticated for other message types
			if not connection.is_authenticated:
				await self.send_message(
					connection_id,
					WebSocketMessage(
						type=MessageType.ERROR,
						data={"error": "Not authenticated"},
						timestamp=datetime.utcnow(),
						correlation_id=message.message_id
					)
				)
				return
			
			# Handle ping/pong
			if message.type == MessageType.PING:
				await self.send_message(
					connection_id,
					WebSocketMessage(
						type=MessageType.PONG,
						data={"timestamp": datetime.utcnow().isoformat()},
						timestamp=datetime.utcnow(),
						correlation_id=message.message_id
					)
				)
				return
			
			# Handle subscription requests
			if message.type == MessageType.SUBSCRIBE:
				channels = message.data.get("channels", [])
				for channel in channels:
					if await self.subscribe(connection_id, channel):
						await self.send_message(
							connection_id,
							WebSocketMessage(
								type=MessageType.SUCCESS,
								data={"subscribed": channel},
								timestamp=datetime.utcnow(),
								correlation_id=message.message_id
							)
						)
				return
			
			# Handle unsubscription requests
			if message.type == MessageType.UNSUBSCRIBE:
				channels = message.data.get("channels", [])
				for channel in channels:
					await self.unsubscribe(connection_id, channel)
				await self.send_message(
					connection_id,
					WebSocketMessage(
						type=MessageType.SUCCESS,
						data={"unsubscribed": channels},
						timestamp=datetime.utcnow(),
						correlation_id=message.message_id
					)
				)
				return
			
			# Call registered handler for message type
			if message.type in self.message_handlers:
				handler = self.message_handlers[message.type]
				try:
					await handler(connection, message)
				except Exception as e:
					logger.error(f"Error in message handler: {e}")
					await self.send_message(
						connection_id,
						WebSocketMessage(
							type=MessageType.ERROR,
							data={"error": str(e)},
							timestamp=datetime.utcnow(),
							correlation_id=message.message_id
						)
					)
			else:
				logger.warning(f"No handler for message type: {message.type.value}")
				
		except json.JSONDecodeError as e:
			logger.error(f"Invalid JSON from {connection_id}: {e}")
			await self.send_message(
				connection_id,
				WebSocketMessage(
					type=MessageType.ERROR,
					data={"error": "Invalid JSON"},
					timestamp=datetime.utcnow()
				)
			)
		except Exception as e:
			logger.error(f"Error handling message from {connection_id}: {e}")
	
	def register_handler(
		self,
		message_type: MessageType,
		handler: Callable
	):
		"""
		Register a handler for a message type.
		
		Args:
			message_type: Type of message to handle
			handler: Async function to handle the message
		"""
		self.message_handlers[message_type] = handler
		logger.info(f"Registered handler for {message_type.value}")
	
	def unregister_handler(self, message_type: MessageType):
		"""
		Unregister a handler for a message type.
		
		Args:
			message_type: Type of message
		"""
		if message_type in self.message_handlers:
			del self.message_handlers[message_type]
	
	def _is_authorized_for_channel(
		self,
		connection: WebSocketConnection,
		channel: str
	) -> bool:
		"""
		Check if a connection is authorized for a channel.
		
		Args:
			connection: WebSocket connection
			channel: Channel name
			
		Returns:
			True if authorized
		"""
		# Organization channels
		if channel.startswith(f"org:{connection.organization_id}"):
			return True
		
		# User channels
		if connection.user_id and channel.startswith(f"user:{connection.organization_id}:{connection.user_id}"):
			return True
		
		# Admin channels
		if channel.startswith("admin:") and "admin" in connection.roles:
			return True
		
		# System channels (admin only)
		if channel.startswith("system:") and "super_admin" in connection.roles:
			return True
		
		return False
	
	async def _auth_timeout_check(self, connection_id: str):
		"""Check if connection authenticated within timeout."""
		await asyncio.sleep(self.auth_timeout)
		
		if connection_id in self.connections:
			connection = self.connections[connection_id]
			if not connection.is_authenticated:
				logger.warning(f"WebSocket {connection_id} authentication timeout")
				await self.disconnect(connection_id, "Authentication timeout")
	
	async def _ping_loop(self):
		"""Send periodic ping messages to keep connections alive."""
		while self._running:
			try:
				await asyncio.sleep(self.ping_interval)
				
				# Send ping to all authenticated connections
				now = datetime.utcnow()
				ping_message = WebSocketMessage(
					type=MessageType.PING,
					data={"timestamp": now.isoformat()},
					timestamp=now
				)
				
				disconnected = []
				for conn_id, connection in list(self.connections.items()):
					if connection.is_authenticated:
						# Check for stale connections
						if (now - connection.last_activity).total_seconds() > self.ping_interval * 3:
							logger.warning(f"Removing stale WebSocket: {conn_id}")
							disconnected.append(conn_id)
						else:
							await self.send_message(conn_id, ping_message)
				
				# Clean up stale connections
				for conn_id in disconnected:
					await self.disconnect(conn_id, "Stale connection")
					
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in ping loop: {e}")
	
	def get_connection_stats(self) -> Dict[str, Any]:
		"""
		Get statistics about current connections.
		
		Returns:
			Dictionary with connection statistics
		"""
		org_stats = {}
		for org_id, conn_ids in self.org_connections.items():
			authenticated = sum(
				1 for conn_id in conn_ids
				if self.connections[conn_id].is_authenticated
			)
			org_stats[org_id] = {
				"total": len(conn_ids),
				"authenticated": authenticated
			}
		
		return {
			"total_connections": len(self.connections),
			"authenticated_connections": sum(
				1 for c in self.connections.values() if c.is_authenticated
			),
			"organizations": org_stats,
			"channels": list(self.channel_connections.keys())
		}