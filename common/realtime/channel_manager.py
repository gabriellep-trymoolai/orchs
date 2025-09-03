"""Multi-tenant channel isolation and management."""

import logging
from typing import Dict, Set, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ChannelType(Enum):
	"""Types of channels in the system."""
	ORGANIZATION = "org"          # Organization-wide channel
	USER = "user"                  # User-specific channel
	DEPARTMENT = "dept"            # Department channel
	ADMIN = "admin"                # Admin-only channel
	SYSTEM = "system"              # System-wide channel
	METRIC = "metric"              # Metric-specific channel
	LOG = "log"                    # Log streaming channel
	CUSTOM = "custom"              # Custom application channel


class ChannelScope(Enum):
	"""Scope of channel access."""
	GLOBAL = "global"              # Available to all organizations
	ORGANIZATION = "organization"   # Limited to specific organization
	DEPARTMENT = "department"       # Limited to specific department
	USER = "user"                  # Limited to specific user
	ADMIN = "admin"                # Admin access only


@dataclass
class ChannelDefinition:
	"""Defines a channel's properties and access rules."""
	name: str
	type: ChannelType
	scope: ChannelScope
	organization_id: Optional[str]
	department_id: Optional[str]
	user_id: Optional[str]
	required_roles: Set[str]
	metadata: Dict
	created_at: datetime
	
	@property
	def full_name(self) -> str:
		"""Get the full channel name with namespace."""
		parts = [self.type.value]
		
		if self.organization_id:
			parts.append(self.organization_id)
		if self.department_id:
			parts.append(self.department_id)
		if self.user_id:
			parts.append(self.user_id)
		if self.name and self.name != self.type.value:
			parts.append(self.name)
		
		return ":".join(parts)


class MultiTenantChannelManager:
	"""
	Manages channel isolation and access control for multi-tenant system.
	Ensures data isolation between organizations while allowing controlled sharing.
	"""
	
	def __init__(self):
		"""Initialize the channel manager."""
		self.channels: Dict[str, ChannelDefinition] = {}
		self.organization_channels: Dict[str, Set[str]] = {}
		self.user_subscriptions: Dict[Tuple[str, str], Set[str]] = {}  # (org_id, user_id) -> channels
		self.department_channels: Dict[Tuple[str, str], Set[str]] = {}  # (org_id, dept_id) -> channels
		
	def create_channel(
		self,
		name: str,
		channel_type: ChannelType,
		organization_id: Optional[str] = None,
		department_id: Optional[str] = None,
		user_id: Optional[str] = None,
		required_roles: Optional[Set[str]] = None,
		metadata: Optional[Dict] = None
	) -> ChannelDefinition:
		"""
		Create a new channel with proper isolation.
		
		Args:
			name: Channel name
			channel_type: Type of channel
			organization_id: Organization for isolation
			department_id: Department for further isolation
			user_id: User for user-specific channels
			required_roles: Roles required to access
			metadata: Additional channel metadata
			
		Returns:
			Created channel definition
		"""
		# Determine scope based on parameters
		if channel_type == ChannelType.SYSTEM:
			scope = ChannelScope.GLOBAL
		elif user_id:
			scope = ChannelScope.USER
		elif department_id:
			scope = ChannelScope.DEPARTMENT
		elif organization_id:
			scope = ChannelScope.ORGANIZATION
		else:
			scope = ChannelScope.GLOBAL
		
		# Create channel definition
		channel = ChannelDefinition(
			name=name,
			type=channel_type,
			scope=scope,
			organization_id=organization_id,
			department_id=department_id,
			user_id=user_id,
			required_roles=required_roles or set(),
			metadata=metadata or {},
			created_at=datetime.utcnow()
		)
		
		# Store channel
		channel_name = channel.full_name
		self.channels[channel_name] = channel
		
		# Track organization channels
		if organization_id:
			if organization_id not in self.organization_channels:
				self.organization_channels[organization_id] = set()
			self.organization_channels[organization_id].add(channel_name)
		
		# Track department channels
		if department_id and organization_id:
			dept_key = (organization_id, department_id)
			if dept_key not in self.department_channels:
				self.department_channels[dept_key] = set()
			self.department_channels[dept_key].add(channel_name)
		
		logger.info(f"Created channel: {channel_name} (scope: {scope.value})")
		return channel
	
	def can_access_channel(
		self,
		channel_name: str,
		organization_id: str,
		user_id: Optional[str] = None,
		department_id: Optional[str] = None,
		roles: Optional[Set[str]] = None
	) -> bool:
		"""
		Check if a user can access a channel.
		
		Args:
			channel_name: Full channel name
			organization_id: User's organization
			user_id: User identifier
			department_id: User's department
			roles: User's roles
			
		Returns:
			True if access is allowed
		"""
		if channel_name not in self.channels:
			return False
		
		channel = self.channels[channel_name]
		roles = roles or set()
		
		# Check scope-based access
		if channel.scope == ChannelScope.GLOBAL:
			# Global channels accessible to all
			pass
		elif channel.scope == ChannelScope.ORGANIZATION:
			# Must be in same organization
			if channel.organization_id != organization_id:
				return False
		elif channel.scope == ChannelScope.DEPARTMENT:
			# Must be in same organization and department
			if channel.organization_id != organization_id:
				return False
			if channel.department_id != department_id:
				return False
		elif channel.scope == ChannelScope.USER:
			# Must be the specific user
			if channel.organization_id != organization_id:
				return False
			if channel.user_id != user_id:
				return False
		elif channel.scope == ChannelScope.ADMIN:
			# Must have admin role
			if "admin" not in roles and "super_admin" not in roles:
				return False
		
		# Check role requirements
		if channel.required_roles:
			if not roles.intersection(channel.required_roles):
				return False
		
		return True
	
	def get_accessible_channels(
		self,
		organization_id: str,
		user_id: Optional[str] = None,
		department_id: Optional[str] = None,
		roles: Optional[Set[str]] = None
	) -> List[str]:
		"""
		Get all channels accessible to a user.
		
		Args:
			organization_id: User's organization
			user_id: User identifier
			department_id: User's department
			roles: User's roles
			
		Returns:
			List of accessible channel names
		"""
		accessible = []
		
		for channel_name, channel in self.channels.items():
			if self.can_access_channel(
				channel_name,
				organization_id,
				user_id,
				department_id,
				roles
			):
				accessible.append(channel_name)
		
		return accessible
	
	def subscribe_user(
		self,
		organization_id: str,
		user_id: str,
		channel_names: List[str],
		department_id: Optional[str] = None,
		roles: Optional[Set[str]] = None
	) -> Tuple[List[str], List[str]]:
		"""
		Subscribe a user to channels.
		
		Args:
			organization_id: User's organization
			user_id: User identifier
			channel_names: Channels to subscribe to
			department_id: User's department
			roles: User's roles
			
		Returns:
			Tuple of (subscribed channels, denied channels)
		"""
		user_key = (organization_id, user_id)
		if user_key not in self.user_subscriptions:
			self.user_subscriptions[user_key] = set()
		
		subscribed = []
		denied = []
		
		for channel_name in channel_names:
			if self.can_access_channel(
				channel_name,
				organization_id,
				user_id,
				department_id,
				roles
			):
				self.user_subscriptions[user_key].add(channel_name)
				subscribed.append(channel_name)
				logger.debug(f"User {user_id} subscribed to {channel_name}")
			else:
				denied.append(channel_name)
				logger.warning(f"User {user_id} denied access to {channel_name}")
		
		return subscribed, denied
	
	def unsubscribe_user(
		self,
		organization_id: str,
		user_id: str,
		channel_names: Optional[List[str]] = None
	):
		"""
		Unsubscribe a user from channels.
		
		Args:
			organization_id: User's organization
			user_id: User identifier
			channel_names: Channels to unsubscribe from (None for all)
		"""
		user_key = (organization_id, user_id)
		if user_key not in self.user_subscriptions:
			return
		
		if channel_names is None:
			# Unsubscribe from all
			self.user_subscriptions[user_key].clear()
		else:
			# Unsubscribe from specific channels
			for channel_name in channel_names:
				self.user_subscriptions[user_key].discard(channel_name)
	
	def get_user_subscriptions(
		self,
		organization_id: str,
		user_id: str
	) -> List[str]:
		"""
		Get all channels a user is subscribed to.
		
		Args:
			organization_id: User's organization
			user_id: User identifier
			
		Returns:
			List of channel names
		"""
		user_key = (organization_id, user_id)
		return list(self.user_subscriptions.get(user_key, set()))
	
	def get_channel_subscribers(
		self,
		channel_name: str
	) -> List[Tuple[str, str]]:
		"""
		Get all users subscribed to a channel.
		
		Args:
			channel_name: Channel name
			
		Returns:
			List of (org_id, user_id) tuples
		"""
		subscribers = []
		for user_key, channels in self.user_subscriptions.items():
			if channel_name in channels:
				subscribers.append(user_key)
		return subscribers
	
	def create_default_channels(self, organization_id: str):
		"""
		Create default channels for an organization.
		
		Args:
			organization_id: Organization identifier
		"""
		# Organization-wide channels
		self.create_channel(
			name="general",
			channel_type=ChannelType.ORGANIZATION,
			organization_id=organization_id,
			metadata={"description": "General organization updates"}
		)
		
		self.create_channel(
			name="metrics",
			channel_type=ChannelType.METRIC,
			organization_id=organization_id,
			metadata={"description": "Real-time metrics"}
		)
		
		self.create_channel(
			name="alerts",
			channel_type=ChannelType.ORGANIZATION,
			organization_id=organization_id,
			metadata={"description": "System alerts and notifications"}
		)
		
		# Admin channel
		self.create_channel(
			name="admin",
			channel_type=ChannelType.ADMIN,
			organization_id=organization_id,
			required_roles={"admin", "super_admin"},
			metadata={"description": "Administrative control"}
		)
		
		# System logs channel
		self.create_channel(
			name="logs",
			channel_type=ChannelType.LOG,
			organization_id=organization_id,
			required_roles={"admin", "debug"},
			metadata={"description": "System logs"}
		)
		
		logger.info(f"Created default channels for organization: {organization_id}")
	
	def validate_channel_isolation(
		self,
		source_org: str,
		target_channel: str
	) -> bool:
		"""
		Validate that cross-organization communication is properly isolated.
		
		Args:
			source_org: Source organization ID
			target_channel: Target channel name
			
		Returns:
			True if communication is allowed
		"""
		if target_channel not in self.channels:
			return False
		
		channel = self.channels[target_channel]
		
		# Global channels allow cross-org communication
		if channel.scope == ChannelScope.GLOBAL:
			return True
		
		# Organization-scoped channels must match org
		if channel.organization_id and channel.organization_id != source_org:
			logger.warning(
				f"Cross-org communication blocked: {source_org} -> {target_channel}"
			)
			return False
		
		return True
	
	def get_organization_stats(self, organization_id: str) -> Dict:
		"""
		Get channel statistics for an organization.
		
		Args:
			organization_id: Organization identifier
			
		Returns:
			Dictionary with channel statistics
		"""
		org_channels = self.organization_channels.get(organization_id, set())
		
		# Count subscribers per channel
		channel_subscribers = {}
		for channel_name in org_channels:
			subscribers = self.get_channel_subscribers(channel_name)
			org_subscribers = [s for s in subscribers if s[0] == organization_id]
			channel_subscribers[channel_name] = len(org_subscribers)
		
		# Count users with subscriptions
		active_users = set()
		for user_key in self.user_subscriptions.keys():
			if user_key[0] == organization_id:
				active_users.add(user_key[1])
		
		return {
			"total_channels": len(org_channels),
			"active_users": len(active_users),
			"channel_subscribers": channel_subscribers,
			"channels": list(org_channels)
		}