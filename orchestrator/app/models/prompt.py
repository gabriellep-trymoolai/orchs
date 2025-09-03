"""Prompt model for orchestrator service."""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON, Float
from sqlalchemy import ForeignKey
from datetime import datetime
from ..db.database import Base


class Prompt(Base):
	"""Prompt template table for orchestrator service."""
	__tablename__ = "prompts"
	
	# Primary key
	prompt_id = Column(String(255), primary_key=True)  # Format: "prompt_001_org_001"
	
	# References
	organization_id = Column(String(255), ForeignKey("organizations.organization_id"), nullable=False, index=True)
	user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False, index=True)
	
	# Prompt metadata
	name = Column(String(255), nullable=False)
	description = Column(Text)
	category = Column(String(100))  # e.g., "code", "writing", "analysis"
	tags = Column(JSON, default=[])  # List of tags for organization
	
	# Prompt content
	system_prompt = Column(Text)
	user_prompt = Column(Text, nullable=False)
	example_input = Column(Text)
	example_output = Column(Text)
	
	# Prompt settings
	is_public = Column(Boolean, default=False)  # Visible to other users in org
	is_template = Column(Boolean, default=False)  # Can be used as template
	is_active = Column(Boolean, default=True)
	
	# Usage and performance
	usage_count = Column(Integer, default=0)
	success_rate = Column(Float, default=0.0)  # Success rate based on user feedback
	avg_tokens_used = Column(Integer, default=0)
	avg_response_time_ms = Column(Integer, default=0)
	
	# Version control
	version = Column(String(20), default="1.0")
	parent_prompt_id = Column(String(255), ForeignKey("prompts.prompt_id"))  # For versioning
	
	# Variables and placeholders
	variables = Column(JSON, default=[])  # List of variable names used in prompt
	required_inputs = Column(JSON, default=[])  # Required input fields
	
	# Timestamps
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	last_used = Column(DateTime)
	
	def __repr__(self):
		return f"<Prompt(prompt_id={self.prompt_id}, name='{self.name}', category='{self.category}')>"