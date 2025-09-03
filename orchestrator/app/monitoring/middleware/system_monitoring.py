"""System performance monitoring middleware - independent from LLM tracking."""

import time
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import json
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.system_metrics import UserSystemPerformance
from ..services.system_metrics import system_metrics_service
from ..config.database_adapter import monitoring_db_adapter as db_manager


def safe_uuid(value: Any) -> uuid.UUID:
    """Safely convert a value to UUID, generating a new one if invalid."""
    if value is None:
        return uuid.uuid4()
    
    if isinstance(value, uuid.UUID):
        return value
    
    if isinstance(value, str):
        if value in ["string", "uuid", ""]:
            return uuid.uuid4()
        
        try:
            return uuid.UUID(value)
        except ValueError:
            # Generate a deterministic UUID based on the string for consistency
            # This ensures org_001 always maps to the same UUID
            return uuid.uuid5(uuid.NAMESPACE_DNS, f"moolai-org-{value}")
    
    return uuid.uuid4()


class SystemPerformanceMiddleware:
    """Middleware to monitor and track system performance metrics."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None,
        organization_id: str = None,
        collection_interval: int = 60,  # seconds
        enable_realtime_redis: bool = True
    ):
        self.redis_client = redis_client
        self.db_session = db_session
        self.organization_id = organization_id or str(uuid.uuid4())
        self.collection_interval = collection_interval
        self.enable_realtime_redis = enable_realtime_redis
        self._last_collection = {}  # Per-user last collection time
        self._collection_tasks = {}  # Active collection tasks
        
    async def track_organization_system_performance(
        self,
        force_collection: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Track system performance for the entire organization.
        
        Args:
            force_collection: Force immediate collection regardless of interval
            
        Returns:
            Dict containing system metrics or None if not collected
        """
        current_time = time.time()
        org_last_collection = self._last_collection.get('organization', 0)
        
        # Check if we should collect metrics
        should_collect = (
            force_collection or 
            (current_time - org_last_collection) >= self.collection_interval
        )
        
        if not should_collect:
            # Return cached metrics from Redis if available
            return await self._get_cached_organization_metrics()
        
        # Update last collection time
        self._last_collection['organization'] = current_time
        
        try:
            # Collect organization-wide system metrics
            org_id = self.organization_id  # Use string directly instead of UUID
            print(f"🔍 Attempting to collect metrics for org ID: {org_id}")
            
            # Store in database - create session if needed
            metrics_data = None
            if self.db_session:
                print("📊 Using existing database session for collection")
                # Use existing session if available
                metrics = await system_metrics_service.record_organization_system_metrics(org_id, self.db_session)
            else:
                print("🔗 Creating new database session for background collection")
                # Create new session for background collection using async context manager
                try:
                    async with db_manager.async_session_factory() as session:
                        print("✅ Database session created successfully")
                        metrics = await system_metrics_service.record_organization_system_metrics(org_id, session)
                        print(f"📈 Metrics recorded with ID: {metrics.metric_id if metrics else 'None'}")
                except Exception as e:
                    print(f"❌ Error creating database session: {e}")
                    metrics = None
            
            if metrics:
                print(f"📊 Metrics collection successful - storing data for timestamp: {metrics.timestamp}")
                metrics_data = {
                    'metric_id': metrics.metric_id,
                    'user_id': metrics.user_id,
                    'organization_id': metrics.organization_id,
                    'timestamp': metrics.timestamp,
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'memory_percent': metrics.memory_percent,
                    'storage_percent': metrics.storage_percent,
                    'cpu_load_1min': metrics.cpu_load_1min,
                    'memory_usage_mb': metrics.memory_usage_mb,
                    'storage_usage_gb': metrics.storage_usage_gb,
                    'collection_duration_ms': metrics.collection_duration_ms
                }
            else:
                print("⚠️ Primary metrics collection failed, trying fallback method")
                # Fallback to direct collection
                system_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"system-{self.organization_id}"))
                metrics_data = await system_metrics_service.collector.collect_system_metrics(
                    user_id=system_user_id,
                    organization_id=org_id
                )
                print(f"🔄 Fallback collection result: {bool(metrics_data)}")
            
            # Store in Redis for real-time access
            if self.redis_client and self.enable_realtime_redis:
                print("💾 Storing metrics in Redis for real-time access")
                await self._store_organization_metrics_redis(metrics_data)
            
            return metrics_data
            
        except Exception as e:
            print(f"❌ Error tracking organization system performance: {e}")
            import traceback
            print(f"📋 Full traceback: {traceback.format_exc()}")
            return None
    
    async def start_continuous_organization_monitoring(
        self,
        custom_interval: Optional[int] = None
    ):
        """Start continuous organization-level system monitoring."""
        interval = custom_interval or self.collection_interval
        
        # Stop existing task if running
        await self.stop_continuous_organization_monitoring()
        
        # Start new monitoring task
        task = asyncio.create_task(
            self._continuous_organization_monitoring_loop(interval)
        )
        self._collection_tasks['organization'] = task
        
        print(f"Started continuous organization monitoring with {interval}s interval")
    
    async def stop_continuous_organization_monitoring(self):
        """Stop continuous organization-level system monitoring."""
        if 'organization' in self._collection_tasks:
            task = self._collection_tasks['organization']
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._collection_tasks['organization']
            print("Stopped continuous organization monitoring")
    
    async def _continuous_organization_monitoring_loop(self, interval: int):
        """Continuous organization monitoring loop."""
        collection_count = 0
        while True:
            try:
                collection_count += 1
                print(f"🔄 Starting automatic collection #{collection_count} for org {self.organization_id}")
                
                result = await self.track_organization_system_performance(force_collection=True)
                
                if result:
                    print(f"✅ Automatic collection #{collection_count} successful - recorded metrics at {result.get('timestamp', 'unknown time')}")
                else:
                    print(f"⚠️ Automatic collection #{collection_count} returned no data")
                
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                print(f"🛑 Background collection loop cancelled for org {self.organization_id}")
                break
            except Exception as e:
                print(f"❌ Error in organization monitoring loop #{collection_count}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
                await asyncio.sleep(min(interval, 60))  # Wait before retrying
    
    async def track_system_performance(
        self,
        user_id: str,
        force_collection: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        DEPRECATED: Track system performance for a user.
        
        This method is deprecated. System performance should be tracked at 
        organization level using track_organization_system_performance().
        
        Args:
            user_id: User identifier
            force_collection: Force immediate collection regardless of interval
            
        Returns:
            Dict containing system metrics or None if not collected
        """
        current_time = time.time()
        user_last_collection = self._last_collection.get(user_id, 0)
        
        # Check if we should collect metrics
        should_collect = (
            force_collection or 
            (current_time - user_last_collection) >= self.collection_interval
        )
        
        if not should_collect:
            # Return cached metrics from Redis if available
            return await self._get_cached_metrics(user_id)
        
        # Update last collection time
        self._last_collection[user_id] = current_time
        
        try:
            # Collect system metrics
            user_id_str = user_id  # Use string directly instead of UUID
            org_id_str = self.organization_id  # Use string directly instead of UUID
            
            metrics_data = await system_metrics_service.collector.collect_system_metrics(
                user_id=user_id_str,
                organization_id=org_id_str
            )
            
            # Store in database if session is available
            if self.db_session:
                await self._store_metrics_db(metrics_data)
            
            # Store in Redis for real-time access
            if self.redis_client and self.enable_realtime_redis:
                await self._store_metrics_redis(user_id, metrics_data)
            
            return metrics_data
            
        except Exception as e:
            print(f"Error tracking system performance for user {user_id}: {e}")
            return None
    
    async def start_continuous_monitoring(
        self,
        user_id: str,
        custom_interval: Optional[int] = None
    ):
        """DEPRECATED: Start continuous system monitoring for a specific user.
        
        System monitoring should be organization-level, not user-level.
        Use organization-based monitoring instead.
        """
        interval = custom_interval or self.collection_interval
        
        # Stop existing task if running
        await self.stop_continuous_monitoring(user_id)
        
        # Start new monitoring task
        task = asyncio.create_task(
            self._continuous_monitoring_loop(user_id, interval)
        )
        self._collection_tasks[user_id] = task
        
        print(f"Started continuous monitoring for user {user_id} with {interval}s interval")
    
    async def stop_continuous_monitoring(self, user_id: str):
        """Stop continuous monitoring for a specific user."""
        if user_id in self._collection_tasks:
            task = self._collection_tasks[user_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._collection_tasks[user_id]
            print(f"Stopped continuous monitoring for user {user_id}")
    
    async def stop_all_monitoring(self):
        """Stop all continuous monitoring tasks."""
        tasks_to_stop = list(self._collection_tasks.keys())
        for user_id in tasks_to_stop:
            await self.stop_continuous_monitoring(user_id)
    
    async def get_user_system_metrics(
        self,
        user_id: str,
        source: str = "redis"  # "redis" or "database"
    ) -> Dict[str, Any]:
        """Get current system metrics for a user."""
        if source == "redis" and self.redis_client:
            return await self._get_redis_metrics(user_id)
        elif source == "database" and self.db_session:
            return await self._get_database_metrics(user_id)
        else:
            return {}
    
    async def get_organization_system_summary(self) -> Dict[str, Any]:
        """Get organization-wide system performance summary."""
        if not self.redis_client:
            return {}
        
        try:
            org_key = f"org:system:{self.organization_id}:summary"
            summary = await self.redis_client.hgetall(org_key)
            
            return {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v
                for k, v in summary.items()
            }
        except Exception as e:
            print(f"Error getting organization summary: {e}")
            return {}
    
    async def record_version_update(
        self,
        component_name: str,
        new_version: str,
        update_type: str = "upgrade",
        deployment_notes: str = None
    ) -> bool:
        """Record an orchestrator version update."""
        try:
            if not self.db_session:
                return False
                
            org_id_str = self.organization_id  # Use string directly instead of UUID
            
            await system_metrics_service.version_tracker.record_version_update(
                organization_id=org_id_str,
                component_name=component_name,
                new_version=new_version,
                update_type=update_type,
                db=self.db_session
            )
            
            # Update Redis with version info
            if self.redis_client:
                await self._update_version_redis(component_name, new_version)
            
            return True
            
        except Exception as e:
            print(f"Error recording version update: {e}")
            return False
    
    async def _continuous_monitoring_loop(self, user_id: str, interval: int):
        """Continuous monitoring loop for a specific user."""
        while True:
            try:
                await self.track_system_performance(user_id, force_collection=True)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in monitoring loop for user {user_id}: {e}")
                await asyncio.sleep(min(interval, 60))  # Wait before retrying
    
    async def _store_metrics_db(self, metrics_data: Dict[str, Any]):
        """Store metrics in database."""
        try:
            performance_record = UserSystemPerformance(**metrics_data)
            self.db_session.add(performance_record)
            await self.db_session.commit()
        except Exception as e:
            print(f"Error storing metrics in database: {e}")
            await self.db_session.rollback()
    
    async def _store_metrics_redis(self, user_id: str, metrics_data: Dict[str, Any]):
        """Store metrics in Redis for real-time access."""
        try:
            # User-specific metrics
            user_key = f"user:system:{user_id}:latest"
            
            # Convert datetime to ISO string for Redis storage
            redis_data = {}
            for key, value in metrics_data.items():
                if isinstance(value, datetime):
                    redis_data[key] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    redis_data[key] = str(value)
                elif value is not None:
                    redis_data[key] = str(value)
            
            await self.redis_client.hset(user_key, mapping=redis_data)
            await self.redis_client.expire(user_key, 300)  # 5 minutes TTL
            
            # Update organization summary
            await self._update_organization_summary(metrics_data)
            
            # Publish real-time update
            await self._publish_system_update(user_id, metrics_data)
            
        except Exception as e:
            print(f"Error storing metrics in Redis: {e}")
    
    async def _store_organization_metrics_redis(self, metrics_data: Dict[str, Any]):
        """Store organization-wide metrics in Redis for real-time access."""
        try:
            # Organization-specific metrics
            org_key = f"org:system:{self.organization_id}:latest"
            
            # Convert datetime to ISO string for Redis storage
            redis_data = {}
            for key, value in metrics_data.items():
                if isinstance(value, datetime):
                    redis_data[key] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    redis_data[key] = str(value)
                elif value is not None:
                    redis_data[key] = str(value)
            
            await self.redis_client.hset(org_key, mapping=redis_data)
            await self.redis_client.expire(org_key, 300)  # 5 minutes TTL
            
            # Update organization summary
            await self._update_organization_summary(metrics_data)
            
            # Publish real-time update
            await self._publish_organization_system_update(metrics_data)
            
        except Exception as e:
            print(f"Error storing organization metrics in Redis: {e}")
    
    async def _update_organization_summary(self, metrics_data: Dict[str, Any]):
        """Update organization-wide system metrics summary."""
        try:
            org_key = f"org:system:{self.organization_id}:summary"
            
            pipe = self.redis_client.pipeline()
            
            # Update counters and averages
            if metrics_data.get('cpu_usage_percent'):
                pipe.lpush(f"{org_key}:cpu_samples", metrics_data['cpu_usage_percent'])
                pipe.ltrim(f"{org_key}:cpu_samples", 0, 99)  # Keep last 100 samples
            
            if metrics_data.get('memory_percent'):
                pipe.lpush(f"{org_key}:memory_samples", metrics_data['memory_percent'])
                pipe.ltrim(f"{org_key}:memory_samples", 0, 99)
            
            if metrics_data.get('storage_percent'):
                pipe.lpush(f"{org_key}:storage_samples", metrics_data['storage_percent'])
                pipe.ltrim(f"{org_key}:storage_samples", 0, 99)
            
            # Update summary metadata
            pipe.hset(org_key, mapping={
                "last_updated": datetime.utcnow().isoformat(),
                "active_monitoring": "true"
            })
            pipe.expire(org_key, 3600)  # 1 hour TTL
            
            await pipe.execute()
            
        except Exception as e:
            print(f"Error updating organization summary: {e}")
    
    async def _publish_system_update(self, user_id: str, metrics_data: Dict[str, Any]):
        """Publish real-time system metrics update."""
        try:
            update_data = {
                "type": "system_metrics_update",
                "user_id": user_id,
                "organization_id": self.organization_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu_percent": metrics_data.get('cpu_usage_percent'),
                    "memory_percent": metrics_data.get('memory_percent'),
                    "storage_percent": metrics_data.get('storage_percent'),
                    "latency_ms": metrics_data.get('api_latency_ms')
                }
            }
            
            await self.redis_client.publish(
                f"org:{self.organization_id}:system_metrics",
                json.dumps(update_data)
            )
            
        except Exception as e:
            print(f"Error publishing system update: {e}")
    
    async def _update_version_redis(self, component_name: str, new_version: str):
        """Update version information in Redis."""
        try:
            version_key = f"org:system:{self.organization_id}:versions"
            await self.redis_client.hset(version_key, component_name, new_version)
            await self.redis_client.expire(version_key, 86400)  # 24 hours TTL
        except Exception as e:
            print(f"Error updating version in Redis: {e}")
    
    async def _get_cached_metrics(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics from Redis."""
        if not self.redis_client:
            return None
            
        return await self._get_redis_metrics(user_id)
    
    async def _get_cached_organization_metrics(self) -> Optional[Dict[str, Any]]:
        """Get cached organization metrics from Redis."""
        if not self.redis_client:
            return None
            
        return await self._get_redis_organization_metrics()
    
    async def _get_redis_organization_metrics(self) -> Dict[str, Any]:
        """Get organization metrics from Redis."""
        try:
            org_key = f"org:system:{self.organization_id}:latest"
            metrics = await self.redis_client.hgetall(org_key)
            
            return {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v
                for k, v in metrics.items()
            }
        except Exception as e:
            print(f"Error getting Redis organization metrics: {e}")
            return {}
    
    async def _publish_organization_system_update(self, metrics_data: Dict[str, Any]):
        """Publish real-time organization system metrics update."""
        try:
            update_data = {
                "type": "organization_system_metrics",
                "organization_id": self.organization_id,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "cpu_percent": metrics_data.get('cpu_usage_percent'),
                    "memory_percent": metrics_data.get('memory_percent'),
                    "storage_percent": metrics_data.get('storage_percent'),
                    "latency_ms": metrics_data.get('api_latency_ms')
                }
            }
            
            await self.redis_client.publish(
                f"org:{self.organization_id}:organization_system_metrics",
                json.dumps(update_data)
            )
            
        except Exception as e:
            print(f"Error publishing organization system update: {e}")
    
    async def _get_redis_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get metrics from Redis."""
        try:
            user_key = f"user:system:{user_id}:latest"
            metrics = await self.redis_client.hgetall(user_key)
            
            return {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v
                for k, v in metrics.items()
            }
        except Exception as e:
            print(f"Error getting Redis metrics: {e}")
            return {}
    
    async def _get_database_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get latest metrics from database."""
        try:
            user_id_str = user_id  # Use string directly instead of UUID
            metrics = await system_metrics_service.get_user_system_metrics(
                user_id=user_id_str,
                hours_back=1,
                db=self.db_session
            )
            
            if metrics:
                latest = metrics[0]
                return {
                    "cpu_percent": latest.cpu_usage_percent,
                    "memory_percent": latest.memory_percent,
                    "storage_percent": latest.storage_percent,
                    "timestamp": latest.timestamp.isoformat(),
                    "source": "database"
                }
            
            return {}
        except Exception as e:
            print(f"Error getting database metrics: {e}")
            return {}


@asynccontextmanager
async def system_monitoring_context(
    redis_client: Optional[redis.Redis] = None,
    db_session: Optional[AsyncSession] = None,
    organization_id: str = None,
    collection_interval: int = 60
):
    """Async context manager for system performance monitoring."""
    middleware = SystemPerformanceMiddleware(
        redis_client=redis_client,
        db_session=db_session,
        organization_id=organization_id,
        collection_interval=collection_interval
    )
    
    try:
        yield middleware
    finally:
        # Clean up all monitoring tasks
        await middleware.stop_all_monitoring()


class SystemMetricsScheduler:
    """Global scheduler for system metrics collection across all users."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        collection_interval: int = 300  # 5 minutes default
    ):
        self.redis_client = redis_client
        self.collection_interval = collection_interval
        self._running = False
        self._task = None
        self.middlewares = {}  # organization_id -> middleware
    
    def register_middleware(self, organization_id: str, middleware: SystemPerformanceMiddleware):
        """Register a middleware for scheduled collection."""
        self.middlewares[organization_id] = middleware
    
    def unregister_middleware(self, organization_id: str):
        """Unregister a middleware."""
        if organization_id in self.middlewares:
            del self.middlewares[organization_id]
    
    async def start_global_collection(self):
        """Start global system metrics collection."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._global_collection_loop())
        print("Started global system metrics collection")
    
    async def stop_global_collection(self):
        """Stop global system metrics collection."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        print("Stopped global system metrics collection")
    
    async def _global_collection_loop(self):
        """Global collection loop."""
        while self._running:
            try:
                await self._collect_for_all_organizations()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in global collection loop: {e}")
                await asyncio.sleep(60)
    
    async def _collect_for_all_organizations(self):
        """Collect metrics for all registered organizations."""
        for org_id, middleware in self.middlewares.items():
            try:
                # Get active users for this organization from Redis
                if self.redis_client:
                    pattern = f"user:system:*:latest"
                    keys = await self.redis_client.keys(pattern)
                    
                    for key in keys:
                        # Extract user ID and trigger collection
                        key_str = key.decode() if isinstance(key, bytes) else key
                        parts = key_str.split(":")
                        if len(parts) >= 3:
                            user_id = parts[2]
                            asyncio.create_task(
                                middleware.track_system_performance(user_id)
                            )
            except Exception as e:
                print(f"Error collecting for organization {org_id}: {e}")


# Global scheduler instance with 2-minute intervals
global_system_scheduler = SystemMetricsScheduler(collection_interval=120)  # 2 minutes