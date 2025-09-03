"""Server-Sent Events (SSE) manager for real-time streaming."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Set, Optional, AsyncGenerator, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SSEConnection:
	"""Represents an SSE connection."""
	connection_id: str
	user_id: Optional[str]
	organization_id: str
	channels: Set[str]
	created_at: datetime
	last_ping: datetime
	metadata: Dict[str, Any]


class SSEManager:
	"""Manages Server-Sent Events connections and message distribution."""
	
	def __init__(self, heartbeat_interval: int = 30):
		"""
		Initialize SSE Manager.
		
		Args:
			heartbeat_interval: Seconds between heartbeat messages
		"""
		self.connections: Dict[str, SSEConnection] = {}
		self.channel_connections: Dict[str, Set[str]] = {}
		self.heartbeat_interval = heartbeat_interval
		self._queues: Dict[str, asyncio.Queue] = {}
		self._running = False
		self._heartbeat_task = None
		
	async def start(self):
		"""Start the SSE manager and heartbeat task."""
		self._running = True
		self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
		logger.info("SSE Manager started")
		
	async def stop(self):
		"""Stop the SSE manager and cleanup."""
		self._running = False
		if self._heartbeat_task:
			self._heartbeat_task.cancel()
			try:
				await self._heartbeat_task
			except asyncio.CancelledError:
				pass
		
		# Close all connections
		for conn_id in list(self.connections.keys()):
			await self.disconnect(conn_id)
		
		logger.info("SSE Manager stopped")
	
	async def connect(
		self,
		organization_id: str,
		user_id: Optional[str] = None,
		channels: Optional[Set[str]] = None,
		metadata: Optional[Dict[str, Any]] = None
	) -> SSEConnection:
		"""
		Create a new SSE connection.
		
		Args:
			organization_id: Organization identifier
			user_id: Optional user identifier
			channels: Initial channels to subscribe to
			metadata: Additional connection metadata
			
		Returns:
			SSEConnection object
		"""
		connection_id = str(uuid.uuid4())
		now = datetime.utcnow()
		
		connection = SSEConnection(
			connection_id=connection_id,
			user_id=user_id,
			organization_id=organization_id,
			channels=channels or set(),
			created_at=now,
			last_ping=now,
			metadata=metadata or {}
		)
		
		self.connections[connection_id] = connection
		self._queues[connection_id] = asyncio.Queue()
		
		# Subscribe to channels
		for channel in connection.channels:
			await self.subscribe(connection_id, channel)
		
		# Subscribe to organization channel by default
		org_channel = f"org:{organization_id}"
		await self.subscribe(connection_id, org_channel)
		
		# Subscribe to user channel if user_id provided
		if user_id:
			user_channel = f"user:{organization_id}:{user_id}"
			await self.subscribe(connection_id, user_channel)
		
		logger.info(f"SSE connection established: {connection_id} for org {organization_id}")
		return connection
	
	async def disconnect(self, connection_id: str):
		"""
		Disconnect an SSE connection.
		
		Args:
			connection_id: Connection identifier
		"""
		if connection_id not in self.connections:
			return
		
		connection = self.connections[connection_id]
		
		# Unsubscribe from all channels
		for channel in list(connection.channels):
			await self.unsubscribe(connection_id, channel)
		
		# Clean up
		del self.connections[connection_id]
		if connection_id in self._queues:
			queue = self._queues[connection_id]
			# Send termination signal
			await queue.put(None)
			del self._queues[connection_id]
		
		logger.info(f"SSE connection closed: {connection_id}")
	
	async def subscribe(self, connection_id: str, channel: str):
		"""
		Subscribe a connection to a channel.
		
		Args:
			connection_id: Connection identifier
			channel: Channel name
		"""
		if connection_id not in self.connections:
			return
		
		connection = self.connections[connection_id]
		connection.channels.add(channel)
		
		if channel not in self.channel_connections:
			self.channel_connections[channel] = set()
		self.channel_connections[channel].add(connection_id)
		
		logger.debug(f"Connection {connection_id} subscribed to {channel}")
	
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
		
		logger.debug(f"Connection {connection_id} unsubscribed from {channel}")
	
	async def publish(
		self,
		channel: str,
		event: str,
		data: Any,
		id: Optional[str] = None
	):
		"""
		Publish a message to a channel.
		
		Args:
			channel: Channel name
			event: Event type
			data: Event data (will be JSON serialized)
			id: Optional event ID
		"""
		if channel not in self.channel_connections:
			return
		
		# Format SSE message
		message = self._format_sse_message(event, data, id)
		
		# Send to all connections in channel
		for conn_id in self.channel_connections[channel]:
			if conn_id in self._queues:
				await self._queues[conn_id].put(message)
		
		logger.debug(f"Published to {channel}: {event}")
	
	async def publish_to_user(
		self,
		organization_id: str,
		user_id: str,
		event: str,
		data: Any,
		id: Optional[str] = None
	):
		"""
		Publish a message to a specific user.
		
		Args:
			organization_id: Organization identifier
			user_id: User identifier
			event: Event type
			data: Event data
			id: Optional event ID
		"""
		channel = f"user:{organization_id}:{user_id}"
		await self.publish(channel, event, data, id)
	
	async def publish_to_organization(
		self,
		organization_id: str,
		event: str,
		data: Any,
		id: Optional[str] = None
	):
		"""
		Publish a message to all connections in an organization.
		
		Args:
			organization_id: Organization identifier
			event: Event type
			data: Event data
			id: Optional event ID
		"""
		channel = f"org:{organization_id}"
		await self.publish(channel, event, data, id)
	
	async def stream(self, connection_id: str) -> AsyncGenerator[str, None]:
		"""
		Stream messages for a connection.
		
		Args:
			connection_id: Connection identifier
			
		Yields:
			SSE formatted messages
		"""
		if connection_id not in self._queues:
			return
		
		queue = self._queues[connection_id]
		
		# Send initial connection message
		yield self._format_sse_message(
			"connected",
			{"connection_id": connection_id, "timestamp": datetime.utcnow().isoformat()}
		)
		
		try:
			while self._running:
				try:
					# Wait for message with timeout for connection check
					message = await asyncio.wait_for(queue.get(), timeout=1.0)
					
					# None signals disconnection
					if message is None:
						break
					
					yield message
					
					# Update last activity
					if connection_id in self.connections:
						self.connections[connection_id].last_ping = datetime.utcnow()
						
				except asyncio.TimeoutError:
					# Continue loop to check if still running
					continue
					
		except asyncio.CancelledError:
			logger.debug(f"Stream cancelled for {connection_id}")
		except Exception as e:
			logger.error(f"Error in SSE stream {connection_id}: {e}")
		finally:
			# Send disconnection message if still connected
			if connection_id in self.connections:
				yield self._format_sse_message(
					"disconnecting",
					{"connection_id": connection_id, "timestamp": datetime.utcnow().isoformat()}
				)
	
	def _format_sse_message(
		self,
		event: str,
		data: Any,
		id: Optional[str] = None
	) -> str:
		"""
		Format a message for SSE transmission.
		
		Args:
			event: Event type
			data: Event data
			id: Optional event ID
			
		Returns:
			SSE formatted message string
		"""
		lines = []
		
		if id:
			lines.append(f"id: {id}")
		
		lines.append(f"event: {event}")
		
		# Convert data to JSON
		if isinstance(data, str):
			json_data = data
		else:
			json_data = json.dumps(data)
		
		# Split data into lines for SSE format
		for line in json_data.split('\n'):
			lines.append(f"data: {line}")
		
		# Add double newline to signal end of message
		return '\n'.join(lines) + '\n\n'
	
	async def _heartbeat_loop(self):
		"""Send periodic heartbeat messages to keep connections alive."""
		while self._running:
			try:
				await asyncio.sleep(self.heartbeat_interval)
				
				# Send heartbeat to all connections
				heartbeat_data = {
					"timestamp": datetime.utcnow().isoformat(),
					"type": "heartbeat"
				}
				
				for conn_id in list(self.connections.keys()):
					if conn_id in self._queues:
						message = self._format_sse_message("heartbeat", heartbeat_data)
						await self._queues[conn_id].put(message)
				
				# Clean up stale connections (no activity for 5 heartbeat intervals)
				stale_threshold = self.heartbeat_interval * 5
				now = datetime.utcnow()
				
				for conn_id in list(self.connections.keys()):
					connection = self.connections[conn_id]
					if (now - connection.last_ping).total_seconds() > stale_threshold:
						logger.warning(f"Removing stale connection: {conn_id}")
						await self.disconnect(conn_id)
						
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error in heartbeat loop: {e}")
	
	def get_connection_stats(self) -> Dict[str, Any]:
		"""
		Get statistics about current connections.
		
		Returns:
			Dictionary with connection statistics
		"""
		org_counts = {}
		for conn in self.connections.values():
			org_id = conn.organization_id
			if org_id not in org_counts:
				org_counts[org_id] = 0
			org_counts[org_id] += 1
		
		return {
			"total_connections": len(self.connections),
			"total_channels": len(self.channel_connections),
			"connections_by_org": org_counts,
			"channels": list(self.channel_connections.keys())
		}