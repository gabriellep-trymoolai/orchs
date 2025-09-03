"""LLM Configuration model for orchestrator service."""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON, Float
from sqlalchemy import ForeignKey
from datetime import datetime
from ..db.database import Base


class LLMConfig(Base):
	"""LLM Configuration table for orchestrator service."""
	__tablename__ = "llm_configs"
	
	# Primary key
	config_id = Column(String(255), primary_key=True)  # Format: "llmconf_001_org_001"
	
	# Reference to organization
	organization_id = Column(String(255), ForeignKey("organizations.organization_id"), nullable=False, index=True)
	
	# Model configuration
	model_name = Column(String(100), nullable=False)  # e.g., "gpt-4", "gpt-3.5-turbo"
	model_version = Column(String(50))  # e.g., "0613", "1106"
	provider = Column(String(50), default="openai")  # openai, anthropic, etc.
	
	# Model parameters
	temperature = Column(Float, default=0.7)
	max_tokens = Column(Integer, default=4000)
	top_p = Column(Float, default=1.0)
	frequency_penalty = Column(Float, default=0.0)
	presence_penalty = Column(Float, default=0.0)
	
	# System prompts and instructions
	system_prompt = Column(Text)
	user_prompt_template = Column(Text)
	
	# Rate limiting and costs
	requests_per_minute = Column(Integer, default=60)
	tokens_per_minute = Column(Integer, default=10000)
	cost_per_1k_tokens = Column(Float, default=0.002)
	
	# Feature flags
	enable_streaming = Column(Boolean, default=True)
	enable_function_calling = Column(Boolean, default=False)
	enable_json_mode = Column(Boolean, default=False)
	
	# Configuration metadata
	name = Column(String(255), nullable=False)  # Human-readable name
	description = Column(Text)
	is_active = Column(Boolean, default=True)
	is_default = Column(Boolean, default=False)
	
	# Advanced configuration
	advanced_settings = Column(JSON, default={})  # Additional provider-specific settings
	
	# Timestamps
	created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
	created_by = Column(String(255), ForeignKey("users.user_id"))
	
	def __repr__(self):
		return f"<LLMConfig(config_id={self.config_id}, name='{self.name}', model='{self.model_name}')>"