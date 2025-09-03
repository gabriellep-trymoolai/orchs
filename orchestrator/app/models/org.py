"""Organization model for orchestrator service."""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON
from datetime import datetime
from ..db.database import Base


class Organization(Base):
	"""Organization table for orchestrator service."""
	__tablename__ = "organizations"
	
	# Primary key
	organization_id = Column(String(255), primary_key=True)  # Format: "org_001"
	
	# Basic organization information
	name = Column(String(255), unique=True, nullable=False, index=True)
	slug = Column(String(100), unique=True, nullable=False, index=True)
	description = Column(Text)
	
	# Organization status
	is_active = Column(Boolean, default=True, nullable=False)
	
	# Subscription and limits
	subscription_tier = Column(String(50), default="free")  # free, pro, enterprise
	monthly_token_limit = Column(Integer, default=10000)
	daily_request_limit = Column(Integer, default=100)
	
	# LLM configuration
	allowed_models = Column(JSON, default=["gpt-3.5-turbo"])  # List of allowed models
	default_model = Column(String(100), default="gpt-3.5-turbo")
	max_tokens_per_request = Column(Integer, default=4000)
	
	# Content filtering and security
	enable_content_filtering = Column(Boolean, default=True)
	content_filter_level = Column(String(20), default="medium")  # low, medium, high
	enable_audit_logging = Column(Boolean, default=True)
	data_retention_days = Column(Integer, default=90)
	
	# Organization settings
	settings = Column(JSON, default={})  # Flexible settings storage
	
	# Contact information
	admin_email = Column(String(255))
	support_email = Column(String(255))
	website = Column(String(255))
	
	# Timestamps
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	
	def __repr__(self):
		return f"<Organization(organization_id={self.organization_id}, name='{self.name}', slug='{self.slug}')>"