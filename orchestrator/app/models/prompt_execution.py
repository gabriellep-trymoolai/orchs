"""Prompt execution model for orchestrator service."""

from sqlalchemy import Column, String, DateTime, Text, Integer, Float
from sqlalchemy import ForeignKey
from datetime import datetime
from ..db.database import Base


class PromptExecution(Base):
    """Prompt execution records for tracking LLM usage."""
    __tablename__ = "prompt_executions"
    
    # Primary key
    prompt_id = Column(String(255), primary_key=True)  # Format: "prompt_123456789_abc12345"
    
    # References
    organization_id = Column(String(255), nullable=False, index=True)
    user_id = Column(String(255), nullable=False, index=True)
    
    # Request details
    prompt_text = Column(Text, nullable=False)
    response_text = Column(Text)
    model = Column(String(100), nullable=False)
    
    # Token usage and cost
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0) 
    total_tokens = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    
    # Performance metrics
    latency_ms = Column(Integer, default=0)
    status = Column(String(50), default="pending")  # pending, success, error
    
    # Session and context
    session_id = Column(String(255), index=True)
    department = Column(String(100))
    
    # LLM parameters
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<PromptExecution(prompt_id={self.prompt_id}, user_id={self.user_id}, model={self.model}, status={self.status})>"