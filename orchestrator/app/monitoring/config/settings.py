"""Configuration settings for embedded monitoring system."""

import os

# Since monitoring is embedded in orchestrator, use orchestrator environment directly

class EmbeddedMonitoringConfig:
	"""Configuration for embedded monitoring system."""
	
	@property
	def organization_id(self) -> str:
		"""Get organization ID from orchestrator environment."""
		return os.getenv("ORGANIZATION_ID", "default-org")
	
	@property
	def system_metrics_interval(self) -> int:
		"""Get metrics collection interval."""
		return int(os.getenv("SYSTEM_METRICS_INTERVAL", "60"))
	
	@property
	def redis_url(self) -> str:
		"""Get Redis URL from orchestrator environment."""
		return os.getenv("REDIS_URL", "redis://localhost:6379/0")
	
	def get_organization_id(self) -> str:
		"""Get organization ID."""
		return self.organization_id

# Global configuration instance
config = EmbeddedMonitoringConfig()

def get_config() -> EmbeddedMonitoringConfig:
	"""Get the global configuration instance."""
	return config