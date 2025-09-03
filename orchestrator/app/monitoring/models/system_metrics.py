"""System performance metrics database models."""

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean,
    DECIMAL, JSON, Index, text, BigInteger
)
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from ...db.database import MonitoringBase


class UserSystemPerformance(MonitoringBase):
    """Per-user system performance metrics."""
    __tablename__ = "user_system_performance"
    
    # Primary identifiers
    metric_id = Column(String(255), primary_key=True)  # Format: "metric_001_user_001"
    user_id = Column(String(255), nullable=False)  # Format: "user_001_org_001"
    organization_id = Column(String(255), nullable=False)  # Format: "org_001"
    timestamp = Column(DateTime, nullable=False)
    
    # CPU Metrics
    cpu_usage_percent = Column(Float)  # Overall CPU usage percentage
    cpu_load_1min = Column(Float)  # 1-minute load average
    cpu_load_5min = Column(Float)  # 5-minute load average
    cpu_load_15min = Column(Float)  # 15-minute load average
    cpu_cores_used = Column(Float)  # Number of CPU cores utilized
    
    # Memory Metrics
    memory_usage_mb = Column(BigInteger)  # Memory usage in MB
    memory_percent = Column(Float)  # Memory usage percentage
    memory_available_mb = Column(BigInteger)  # Available memory in MB
    memory_total_mb = Column(BigInteger)  # Total memory in MB
    swap_usage_mb = Column(BigInteger)  # Swap usage in MB
    swap_percent = Column(Float)  # Swap usage percentage
    
    # Storage Metrics
    storage_usage_gb = Column(Float)  # Storage usage in GB
    storage_percent = Column(Float)  # Storage usage percentage
    storage_available_gb = Column(Float)  # Available storage in GB
    storage_total_gb = Column(Float)  # Total storage in GB
    disk_read_mb_s = Column(Float)  # Disk read speed MB/s
    disk_write_mb_s = Column(Float)  # Disk write speed MB/s
    iops_read = Column(Integer)  # Read I/O operations per second
    iops_write = Column(Integer)  # Write I/O operations per second
    
    # Network Metrics
    network_in_mb_s = Column(Float)  # Network input MB/s
    network_out_mb_s = Column(Float)  # Network output MB/s
    network_connections = Column(Integer)  # Active network connections
    
    # Latency Metrics
    api_latency_ms = Column(Integer)  # API response latency
    db_latency_ms = Column(Integer)  # Database query latency
    redis_latency_ms = Column(Integer)  # Redis operation latency
    system_latency_ms = Column(Integer)  # Overall system latency
    
    # Container/Service Metrics
    container_count = Column(Integer)  # Number of active containers
    service_count = Column(Integer)  # Number of running services
    container_restarts = Column(Integer)  # Container restart count
    
    # Process Metrics
    process_count = Column(Integer)  # Total process count
    thread_count = Column(Integer)  # Total thread count
    file_descriptors = Column(Integer)  # Open file descriptors
    
    # Metadata
    collected_by = Column(String(50))  # Collection method (prometheus, psutil, docker)
    collection_duration_ms = Column(Integer)  # Time taken to collect metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_sys_perf_user_time', 'user_id', 'timestamp'),
        Index('idx_sys_perf_org_time', 'organization_id', 'timestamp'),
        Index('idx_sys_perf_timestamp', 'timestamp'),
        Index('idx_sys_perf_cpu', 'cpu_usage_percent', 'timestamp'),
        Index('idx_sys_perf_memory', 'memory_percent', 'timestamp'),
    )


class OrchestratorVersionHistory(MonitoringBase):
    """Orchestrator version and update history tracking."""
    __tablename__ = "orchestrator_version_history"
    
    version_id = Column(String(255), primary_key=True)  # Format: "ver_001_org_001"
    organization_id = Column(String(255), nullable=False)  # Format: "org_001"
    
    # Version Information
    orchestrator_version = Column(String(50), nullable=False)
    previous_version = Column(String(50))
    component_name = Column(String(100), nullable=False)  # Service/container name
    component_type = Column(String(50))  # 'container', 'service', 'library', 'database'
    
    # Update Information
    update_type = Column(String(50))  # 'upgrade', 'downgrade', 'patch', 'rollback'
    update_timestamp = Column(DateTime, nullable=False)
    update_duration_seconds = Column(Integer)
    update_status = Column(String(20))  # 'success', 'failed', 'partial'
    
    # Deployment Details
    deployment_method = Column(String(50))  # 'docker', 'kubernetes', 'manual'
    deployed_by = Column(String(100))  # User or system that triggered update
    deployment_notes = Column(String)
    
    # Git/Source Information
    git_commit = Column(String(40))
    git_branch = Column(String(100))
    git_tag = Column(String(100))
    build_number = Column(String(50))
    
    # Configuration Changes
    config_changes = Column(JSONB)  # JSON of configuration changes
    environment_variables = Column(JSONB)  # Environment variable changes
    
    # Dependencies
    dependencies_updated = Column(JSONB)  # List of updated dependencies
    
    # Rollback Information
    is_rollback = Column(Boolean, default=False)
    rollback_from_version = Column(String(50))
    rollback_reason = Column(String)
    
    # Performance Impact
    performance_impact = Column(JSONB)  # Metrics before/after update
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_version_hist_org', 'organization_id', 'update_timestamp'),
        Index('idx_version_hist_component', 'component_name', 'update_timestamp'),
        Index('idx_version_hist_version', 'orchestrator_version'),
        Index('idx_version_hist_timestamp', 'update_timestamp'),
    )


class SystemPerformanceAggregated(MonitoringBase):
    """Aggregated system performance metrics for trending and analytics."""
    __tablename__ = "system_performance_aggregated"
    
    # Primary identifiers
    user_id = Column(String(255), primary_key=True)  # Format: "user_001_org_001"
    organization_id = Column(String(255), primary_key=True)  # Format: "org_001"
    timestamp = Column(DateTime, primary_key=True)
    time_bucket = Column(String(20), primary_key=True)  # 'minute', 'hour', 'day', 'month'
    
    # Aggregated CPU Metrics
    avg_cpu_percent = Column(Float)
    max_cpu_percent = Column(Float)
    min_cpu_percent = Column(Float)
    p95_cpu_percent = Column(Float)
    
    # Aggregated Memory Metrics
    avg_memory_percent = Column(Float)
    max_memory_percent = Column(Float)
    min_memory_percent = Column(Float)
    p95_memory_percent = Column(Float)
    avg_memory_mb = Column(BigInteger)
    
    # Aggregated Storage Metrics
    avg_storage_percent = Column(Float)
    max_storage_percent = Column(Float)
    storage_growth_gb = Column(Float)  # Storage growth in time period
    
    # Aggregated Latency Metrics
    avg_api_latency_ms = Column(Integer)
    p50_api_latency_ms = Column(Integer)
    p95_api_latency_ms = Column(Integer)
    p99_api_latency_ms = Column(Integer)
    avg_db_latency_ms = Column(Integer)
    avg_system_latency_ms = Column(Integer)
    
    # Network Aggregations
    total_network_in_gb = Column(Float)
    total_network_out_gb = Column(Float)
    avg_network_connections = Column(Integer)
    
    # Resource Utilization Score (0-100)
    resource_efficiency_score = Column(Float)
    
    # Alert Counts
    cpu_alerts_count = Column(Integer, default=0)
    memory_alerts_count = Column(Integer, default=0)
    storage_alerts_count = Column(Integer, default=0)
    
    # Sample Count
    sample_count = Column(Integer)  # Number of samples in aggregation
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_sys_perf_agg_user', 'user_id', 'timestamp'),
        Index('idx_sys_perf_agg_org', 'organization_id', 'timestamp'),
        Index('idx_sys_perf_agg_bucket', 'time_bucket', 'timestamp'),
    )


class SystemAlert(MonitoringBase):
    """System performance alerts and thresholds."""
    __tablename__ = "system_alerts"
    
    alert_id = Column(String(255), primary_key=True)  # Format: "alert_001_org_001"
    user_id = Column(String(255))  # Format: "user_001_org_001"
    organization_id = Column(String(255), nullable=False)  # Format: "org_001"
    
    # Alert Details
    alert_type = Column(String(50), nullable=False)  # 'cpu', 'memory', 'storage', 'latency'
    alert_severity = Column(String(20), nullable=False)  # 'critical', 'warning', 'info'
    alert_name = Column(String(200), nullable=False)
    alert_description = Column(String)
    
    # Threshold Information
    metric_name = Column(String(100), nullable=False)
    threshold_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=False)
    threshold_operator = Column(String(10))  # '>', '<', '>=', '<=', '=='
    
    # Alert State
    alert_status = Column(String(20), default='active')  # 'active', 'acknowledged', 'resolved'
    triggered_at = Column(DateTime, nullable=False)
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Resolution Details
    resolved_by = Column(String(100))
    resolution_notes = Column(String)
    auto_resolved = Column(Boolean, default=False)
    
    # Impact Assessment
    affected_services = Column(JSONB)
    estimated_impact = Column(String(20))  # 'high', 'medium', 'low'
    affected_users_count = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_alerts_org_status', 'organization_id', 'alert_status'),
        Index('idx_alerts_user', 'user_id', 'triggered_at'),
        Index('idx_alerts_type_severity', 'alert_type', 'alert_severity'),
        Index('idx_alerts_triggered', 'triggered_at'),
    )