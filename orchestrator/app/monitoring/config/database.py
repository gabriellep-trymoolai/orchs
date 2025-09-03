"""Database configuration for embedded monitoring system."""

# Since monitoring is embedded in the orchestrator, use orchestrator's database manager directly
from ...db.database import db_manager as orchestrator_db_manager, Base

# Compatibility functions for monitoring API
async def get_db():
	"""Dependency to get monitoring database session."""
	# Use the orchestrator's monitoring database
	async for session in orchestrator_db_manager.get_monitoring_session():
		yield session

async def init_db():
	"""Initialize monitoring database tables (handled by orchestrator)."""
	# Tables are initialized by the orchestrator's database manager
	await orchestrator_db_manager.init_monitoring_database()