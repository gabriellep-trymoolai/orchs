"""Database adapter to use orchestrator's monitoring database."""

from ...db.database import db_manager, MonitoringBase

# Redirect monitoring database manager to use orchestrator's monitoring connection
class MonitoringDatabaseAdapter:
    """Adapter to use orchestrator's monitoring database from monitoring code."""
    
    @property
    def async_engine(self):
        """Get monitoring async engine from orchestrator db manager."""
        return db_manager.monitoring_async_engine
    
    @property
    def async_session_factory(self):
        """Get monitoring session factory from orchestrator db manager."""
        return db_manager.monitoring_async_session_factory
    
    async def get_session(self):
        """Get monitoring database session."""
        async for session in db_manager.get_monitoring_session():
            yield session
    
    async def test_connection(self) -> bool:
        """Test monitoring database connection."""
        return await db_manager.test_monitoring_connection()
    
    async def init_database(self):
        """Initialize monitoring database tables."""
        await db_manager.init_monitoring_database()
    
    async def close(self):
        """Close monitoring database connections (handled by orchestrator)."""
        # Don't close as orchestrator manages these connections
        pass

# Global adapter instance to replace monitoring's database manager
monitoring_db_adapter = MonitoringDatabaseAdapter()

# Export the monitoring base for models
Base = MonitoringBase

# Compatibility functions for monitoring code
async def get_db():
    """Dependency to get monitoring database session."""
    async for session in monitoring_db_adapter.get_session():
        yield session

async def init_db():
    """Initialize monitoring database tables."""
    await monitoring_db_adapter.init_database()