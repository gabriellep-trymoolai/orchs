"""Firewall log model for orchestrator service."""

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer, JSON, Float
from sqlalchemy import ForeignKey
from datetime import datetime
from ..db.database import Base


class FirewallLog(Base):
	"""Firewall and content filtering log table."""
	__tablename__ = "firewall_logs"
	
	# Primary key
	log_id = Column(String(255), primary_key=True)  # Format: "fwlog_001_org_001"
	
	# References
	organization_id = Column(String(255), ForeignKey("organizations.organization_id"), nullable=False, index=True)
	user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False, index=True)
	
	# Request identification
	request_id = Column(String(100), index=True)  # Unique request identifier
	session_id = Column(String(100), index=True)  # User session identifier
	
	# Firewall event details
	event_type = Column(String(50), nullable=False)  # "blocked", "flagged", "allowed", "filtered"
	severity = Column(String(20), default="medium")  # "low", "medium", "high", "critical"
	rule_name = Column(String(255))  # Name of the triggered rule
	rule_category = Column(String(100))  # "pii", "profanity", "toxicity", "policy"
	
	# Content analysis
	original_content = Column(Text)  # Original user input (if appropriate to store)
	filtered_content = Column(Text)  # Content after filtering
	detected_entities = Column(JSON, default=[])  # List of detected PII/sensitive entities
	confidence_score = Column(Float)  # Confidence in the detection (0.0-1.0)
	
	# Firewall decision
	action_taken = Column(String(50), nullable=False)  # "block", "filter", "warn", "allow"
	reason = Column(Text)  # Human-readable reason for the action
	risk_score = Column(Float, default=0.0)  # Overall risk assessment (0.0-1.0)
	
	# Context information
	ip_address = Column(String(45))  # IPv4 or IPv6 address
	user_agent = Column(String(500))  # Browser/client user agent
	endpoint = Column(String(255))  # API endpoint that triggered the check
	method = Column(String(10))  # HTTP method (GET, POST, etc.)
	
	# Performance metrics
	processing_time_ms = Column(Integer)  # Time taken to process the request
	model_used = Column(String(100))  # ML model or service used for detection
	
	# Timestamps
	timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
	resolved_at = Column(DateTime)  # When the issue was resolved (if applicable)
	
	# Administrative
	is_false_positive = Column(Boolean, default=False)  # Marked as false positive
	reviewed_by = Column(String(255), ForeignKey("users.user_id"))  # Admin who reviewed
	notes = Column(Text)  # Administrative notes
	
	def __repr__(self):
		return f"<FirewallLog(log_id={self.log_id}, event_type='{self.event_type}', action='{self.action_taken}')>"