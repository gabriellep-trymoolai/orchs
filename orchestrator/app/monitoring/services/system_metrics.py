"""System performance metrics collection service."""

import asyncio
import json
import logging
import platform
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import uuid

import psutil
import docker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc

from ..config.database import get_db
from ..models.system_metrics import (
    UserSystemPerformance,
    OrchestratorVersionHistory,
    SystemPerformanceAggregated,
    SystemAlert
)

logger = logging.getLogger(__name__)


class SystemMetricsCollector:
    """Collects system performance metrics from the orchestrator."""
    
    def __init__(self):
        self.docker_client = None
        self._init_docker_client()
    
    def _init_docker_client(self):
        """Initialize Docker client for container metrics."""
        try:
            # Try different Docker client configurations
            docker_configs = [
                # Standard Docker socket configurations
                {"base_url": "unix://var/run/docker.sock"},
                {"base_url": "tcp://localhost:2376"},
                {"base_url": "tcp://localhost:2375"},
                # Default from environment
                {}
            ]
            
            for config in docker_configs:
                try:
                    if config:
                        self.docker_client = docker.DockerClient(**config)
                    else:
                        self.docker_client = docker.from_env()
                    
                    # Test the connection
                    self.docker_client.ping()
                    print(f"Docker client initialized successfully with config: {config or 'from_env'}")
                    return
                    
                except Exception as config_error:
                    continue
            
            # If all configurations fail
            raise Exception("All Docker client configurations failed")
            
        except Exception as e:
            print(f"Docker client initialization failed: {e}")
            print("System metrics will work without Docker container monitoring")
            self.docker_client = None
    
    async def collect_system_metrics(self, user_id: str, organization_id: str) -> Dict[str, Any]:
        """Collect comprehensive system performance metrics."""
        collection_start = time.time()
        
        try:
            # Collect all metrics
            cpu_metrics = self._collect_cpu_metrics()
            memory_metrics = self._collect_memory_metrics()
            storage_metrics = self._collect_storage_metrics()
            network_metrics = self._collect_network_metrics()
            container_metrics = self._collect_container_metrics()
            process_metrics = self._collect_process_metrics()
            latency_metrics = await self._collect_latency_metrics()
            
            collection_duration = int((time.time() - collection_start) * 1000)
            
            # Generate metric ID - ensure user_id is string
            user_id_str = str(user_id) if not isinstance(user_id, str) else user_id
            org_id_str = str(organization_id) if not isinstance(organization_id, str) else organization_id
            metric_id = f"metric_{int(time.time())}_{user_id_str.replace('_', '')[:8]}"
            
            # Debug prints
            print(f"DEBUG: user_id type: {type(user_id)}, value: {user_id}")
            print(f"DEBUG: organization_id type: {type(organization_id)}, value: {organization_id}")
            print(f"DEBUG: metric_id type: {type(metric_id)}, value: {metric_id}")
            
            # Combine all metrics
            metrics = {
                'metric_id': metric_id,
                'user_id': user_id_str,
                'organization_id': org_id_str,
                'timestamp': datetime.utcnow(),
                'collection_duration_ms': collection_duration,
                'collected_by': 'psutil+docker',
                **cpu_metrics,
                **memory_metrics,
                **storage_metrics,
                **network_metrics,
                **container_metrics,
                **process_metrics,
                **latency_metrics
            }
            
            logger.info(f"System metrics collected in {collection_duration}ms")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            raise
    
    def _collect_cpu_metrics(self) -> Dict[str, Any]:
        """Collect CPU performance metrics."""
        try:
            # Get CPU usage percentage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get load averages
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                load_1min, load_5min, load_15min = load_avg
            else:
                # Fallback for Windows
                load_1min = load_5min = load_15min = None
            
            # Get CPU count
            cpu_count = psutil.cpu_count()
            cores_used = (cpu_percent / 100) * cpu_count if cpu_count else None
            
            return {
                'cpu_usage_percent': cpu_percent,
                'cpu_load_1min': load_1min,
                'cpu_load_5min': load_5min,
                'cpu_load_15min': load_15min,
                'cpu_cores_used': cores_used
            }
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            return {}
    
    def _collect_memory_metrics(self) -> Dict[str, Any]:
        """Collect memory performance metrics."""
        try:
            # Virtual memory
            mem = psutil.virtual_memory()
            
            # Swap memory
            swap = psutil.swap_memory()
            
            return {
                'memory_usage_mb': int(mem.used / 1024 / 1024),
                'memory_percent': mem.percent,
                'memory_available_mb': int(mem.available / 1024 / 1024),
                'memory_total_mb': int(mem.total / 1024 / 1024),
                'swap_usage_mb': int(swap.used / 1024 / 1024),
                'swap_percent': swap.percent
            }
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            return {}
    
    def _collect_storage_metrics(self) -> Dict[str, Any]:
        """Collect storage performance metrics."""
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O statistics
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    # Convert to MB/s (approximate)
                    disk_read_mb_s = disk_io.read_bytes / 1024 / 1024 / 60  # Rough estimate
                    disk_write_mb_s = disk_io.write_bytes / 1024 / 1024 / 60
                    iops_read = disk_io.read_count
                    iops_write = disk_io.write_count
                else:
                    disk_read_mb_s = disk_write_mb_s = iops_read = iops_write = None
            except:
                disk_read_mb_s = disk_write_mb_s = iops_read = iops_write = None
            
            return {
                'storage_usage_gb': round(disk_usage.used / 1024 / 1024 / 1024, 2),
                'storage_percent': round((disk_usage.used / disk_usage.total) * 100, 2),
                'storage_available_gb': round(disk_usage.free / 1024 / 1024 / 1024, 2),
                'storage_total_gb': round(disk_usage.total / 1024 / 1024 / 1024, 2),
                'disk_read_mb_s': disk_read_mb_s,
                'disk_write_mb_s': disk_write_mb_s,
                'iops_read': iops_read,
                'iops_write': iops_write
            }
        except Exception as e:
            logger.error(f"Error collecting storage metrics: {e}")
            return {}
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network performance metrics."""
        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters()
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
            except:
                connections = None
            
            return {
                'network_in_mb_s': round(net_io.bytes_recv / 1024 / 1024 / 60, 2),  # Rough estimate
                'network_out_mb_s': round(net_io.bytes_sent / 1024 / 1024 / 60, 2),
                'network_connections': connections
            }
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return {}
    
    def _collect_container_metrics(self) -> Dict[str, Any]:
        """Collect container/service metrics."""
        try:
            if not self.docker_client:
                return {}
            
            containers = self.docker_client.containers.list()
            running_containers = len(containers)
            
            # Count container restarts
            total_restarts = 0
            for container in containers:
                try:
                    restart_count = container.attrs.get('RestartCount', 0)
                    total_restarts += restart_count
                except:
                    pass
            
            # Service count (approximated by running containers)
            service_count = running_containers
            
            return {
                'container_count': running_containers,
                'service_count': service_count,
                'container_restarts': total_restarts
            }
        except Exception as e:
            logger.error(f"Error collecting container metrics: {e}")
            return {}
    
    def _collect_process_metrics(self) -> Dict[str, Any]:
        """Collect process-level metrics."""
        try:
            # Process and thread counts
            process_count = len(psutil.pids())
            
            # Thread count (sum of threads from all processes)
            total_threads = 0
            total_fds = 0
            
            for proc in psutil.process_iter(['num_threads']):
                try:
                    total_threads += proc.info.get('num_threads', 0)
                except:
                    pass
            
            # File descriptors (Unix-like systems)
            try:
                if hasattr(psutil.Process(), 'num_fds'):
                    for proc in psutil.process_iter():
                        try:
                            total_fds += proc.num_fds()
                        except:
                            pass
            except:
                total_fds = None
            
            return {
                'process_count': process_count,
                'thread_count': total_threads,
                'file_descriptors': total_fds
            }
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return {}
    
    async def _collect_latency_metrics(self) -> Dict[str, Any]:
        """Collect latency metrics."""
        try:
            # These would be collected from actual request monitoring
            # For now, return placeholder values that would be filled by middleware
            return {
                'api_latency_ms': None,  # Will be filled by middleware
                'db_latency_ms': None,   # Will be filled by database monitoring
                'redis_latency_ms': None,  # Will be filled by Redis monitoring
                'system_latency_ms': None  # Will be calculated from overall response time
            }
        except Exception as e:
            logger.error(f"Error collecting latency metrics: {e}")
            return {}


class SystemMetricsService:
    """Service for managing system performance metrics."""
    
    def __init__(self):
        self.collector = SystemMetricsCollector()
        self.version_tracker = OrchestratorVersionTracker()
    
    async def record_system_metrics(self, user_id: str, organization_id: str, 
                                   db: AsyncSession) -> UserSystemPerformance:
        """Record system performance metrics for a user."""
        try:
            # Ensure IDs are strings
            user_id_str = str(user_id) if not isinstance(user_id, str) else user_id
            org_id_str = str(organization_id) if not isinstance(organization_id, str) else organization_id
            
            # Collect metrics
            metrics_data = await self.collector.collect_system_metrics(user_id_str, org_id_str)
            
            # Create database record
            performance_record = UserSystemPerformance(**metrics_data)
            
            # Save to database
            db.add(performance_record)
            await db.commit()
            await db.refresh(performance_record)
            
            logger.info(f"System metrics recorded for user {user_id} in organization {organization_id}")
            return performance_record
            
        except Exception as e:
            logger.error(f"Error recording system metrics: {e}")
            await db.rollback()
            raise
    
    async def record_organization_system_metrics(self, organization_id: str, 
                                               db: AsyncSession) -> UserSystemPerformance:
        """Record system performance metrics for an entire organization."""
        try:
            # Use a system user ID for organization-wide metrics
            system_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"system-{organization_id}"))
            
            # Ensure organization_id is string
            org_id_str = str(organization_id) if not isinstance(organization_id, str) else organization_id
            
            # Collect metrics
            metrics_data = await self.collector.collect_system_metrics(system_user_id, org_id_str)
            
            # Create database record
            performance_record = UserSystemPerformance(**metrics_data)
            
            # Save to database
            db.add(performance_record)
            await db.commit()
            await db.refresh(performance_record)
            
            logger.info(f"Organization system metrics recorded for {organization_id}")
            return performance_record
            
        except Exception as e:
            logger.error(f"Error recording organization system metrics: {e}")
            await db.rollback()
            raise
    
    async def get_user_system_metrics(self, user_id: str, db: AsyncSession, 
                                    hours_back: int = 24) -> List[UserSystemPerformance]:
        """Get recent system metrics for a user."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            query = select(UserSystemPerformance).where(
                UserSystemPerformance.user_id == user_id,
                UserSystemPerformance.timestamp >= cutoff_time
            ).order_by(desc(UserSystemPerformance.timestamp))
            
            result = await db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting user system metrics: {e}")
            raise
    
    async def get_organization_system_metrics(self, organization_id: str, db: AsyncSession, 
                                            hours_back: int = 24) -> List[UserSystemPerformance]:
        """Get recent system metrics for an organization."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            query = select(UserSystemPerformance).where(
                UserSystemPerformance.organization_id == organization_id,
                UserSystemPerformance.timestamp >= cutoff_time
            ).order_by(desc(UserSystemPerformance.timestamp))
            
            result = await db.execute(query)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting organization system metrics: {e}")
            raise
    
    async def get_latest_organization_system_metrics(self, organization_id: str, 
                                                   db: AsyncSession) -> Optional[UserSystemPerformance]:
        """Get the most recent system metrics for an organization."""
        try:
            # Ensure organization_id is string
            org_id_str = str(organization_id) if not isinstance(organization_id, str) else organization_id
            
            # Get the system user for this organization
            system_user_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"system-{org_id_str}"))
            
            query = select(UserSystemPerformance).where(
                UserSystemPerformance.organization_id == org_id_str,
                UserSystemPerformance.user_id == system_user_id
            ).order_by(desc(UserSystemPerformance.timestamp)).limit(1)
            
            result = await db.execute(query)
            return result.scalar_one_or_none()
            
        except Exception as e:
            logger.error(f"Error getting latest organization system metrics: {e}")
            raise
    
    async def aggregate_system_metrics(self, db: AsyncSession = None, time_bucket: str = 'hour'):
        """Aggregate system metrics for analytics."""
        if db is None:
            async with get_db() as db_session:
                return await self._perform_aggregation(time_bucket, db_session)
        else:
            return await self._perform_aggregation(time_bucket, db)
    
    async def _perform_aggregation(self, time_bucket: str, db: AsyncSession):
        """Perform the actual aggregation work."""
        try:
            # This would implement proper time-bucket aggregation
            # For now, we'll create a placeholder implementation
            logger.info(f"Aggregating system metrics for time bucket: {time_bucket}")
            
            # Implementation would aggregate metrics by user/org/time_bucket
            # and calculate avg, max, min, percentiles for each metric
            
        except Exception as e:
            logger.error(f"Error aggregating system metrics: {e}")
            raise


class OrchestratorVersionTracker:
    """Tracks orchestrator versions and update history."""
    
    def __init__(self):
        self.docker_client = None
        self._init_docker_client()
    
    def _init_docker_client(self):
        """Initialize Docker client."""
        try:
            # Try different Docker client configurations
            docker_configs = [
                # Standard Docker socket configurations
                {"base_url": "unix://var/run/docker.sock"},
                {"base_url": "tcp://localhost:2376"},
                {"base_url": "tcp://localhost:2375"},
                # Default from environment
                {}
            ]
            
            for config in docker_configs:
                try:
                    if config:
                        self.docker_client = docker.DockerClient(**config)
                    else:
                        self.docker_client = docker.from_env()
                    
                    # Test the connection
                    self.docker_client.ping()
                    print(f"Version tracker Docker client initialized successfully")
                    return
                    
                except Exception as config_error:
                    continue
            
            # If all configurations fail
            raise Exception("All Docker client configurations failed")
            
        except Exception as e:
            print(f"Version tracker Docker client initialization failed: {e}")
            self.docker_client = None
    
    async def record_version_update(self, organization_id: str, component_name: str, 
                                   new_version: str, db: AsyncSession = None, 
                                   update_type: str = 'upgrade') -> OrchestratorVersionHistory:
        """Record a version update."""
        try:
            # Get previous version
            previous_version = await self._get_current_version(organization_id, component_name, db)
            
            # Create version record
            version_record = OrchestratorVersionHistory(
                organization_id=organization_id,
                component_name=component_name,
                orchestrator_version=new_version,
                previous_version=previous_version,
                update_type=update_type,
                update_timestamp=datetime.utcnow(),
                update_status='success',
                deployment_method='docker'
            )
            
            if db:
                db.add(version_record)
                await db.commit()
                await db.refresh(version_record)
            
            logger.info(f"Version update recorded: {component_name} -> {new_version}")
            return version_record
            
        except Exception as e:
            logger.error(f"Error recording version update: {e}")
            raise
    
    async def _get_current_version(self, organization_id: str, component_name: str, 
                                 db: AsyncSession) -> Optional[str]:
        """Get current version of a component."""
        try:
            if not db:
                return None
                
            query = select(OrchestratorVersionHistory).where(
                OrchestratorVersionHistory.organization_id == organization_id,
                OrchestratorVersionHistory.component_name == component_name
            ).order_by(desc(OrchestratorVersionHistory.update_timestamp)).limit(1)
            
            result = await db.execute(query)
            latest_record = result.scalar_one_or_none()
            
            return latest_record.orchestrator_version if latest_record else None
            
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return None
    
    def get_docker_container_versions(self) -> Dict[str, str]:
        """Get versions of all running Docker containers."""
        if not self.docker_client:
            return {}
        
        try:
            versions = {}
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    name = container.name
                    image = container.image.tags[0] if container.image.tags else 'unknown'
                    versions[name] = image
                except Exception as e:
                    logger.warning(f"Error getting version for container {container.name}: {e}")
            
            return versions
            
        except Exception as e:
            logger.error(f"Error getting Docker container versions: {e}")
            return {}


# Singleton instances
system_metrics_service = SystemMetricsService()