"""API endpoints for system performance metrics."""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, func, and_
from pydantic import BaseModel

from ...config.database import get_db
from ...models.system_metrics import (
    UserSystemPerformance,
    OrchestratorVersionHistory,
    SystemPerformanceAggregated,
    SystemAlert
)
from ...services.system_metrics import system_metrics_service

router = APIRouter(prefix="/api/v1/system", tags=["system-metrics"])


# Response models
class SystemMetricsResponse(BaseModel):
    """Response model for system metrics."""
    metric_id: str
    user_id: str
    organization_id: str
    timestamp: datetime
    
    # CPU metrics
    cpu_usage_percent: Optional[float]
    cpu_load_1min: Optional[float]
    cpu_load_5min: Optional[float]
    cpu_load_15min: Optional[float]
    cpu_cores_used: Optional[float]
    
    # Memory metrics
    memory_usage_mb: Optional[int]
    memory_percent: Optional[float]
    memory_available_mb: Optional[int]
    memory_total_mb: Optional[int]
    swap_usage_mb: Optional[int]
    swap_percent: Optional[float]
    
    # Storage metrics
    storage_usage_gb: Optional[float]
    storage_percent: Optional[float]
    storage_available_gb: Optional[float]
    storage_total_gb: Optional[float]
    disk_read_mb_s: Optional[float]
    disk_write_mb_s: Optional[float]
    iops_read: Optional[int]
    iops_write: Optional[int]
    
    # Network metrics
    network_in_mb_s: Optional[float]
    network_out_mb_s: Optional[float]
    network_connections: Optional[int]
    
    # Latency metrics
    api_latency_ms: Optional[int]
    db_latency_ms: Optional[int]
    redis_latency_ms: Optional[int]
    system_latency_ms: Optional[int]
    
    # Container metrics
    container_count: Optional[int]
    service_count: Optional[int]
    container_restarts: Optional[int]
    
    # Process metrics
    process_count: Optional[int]
    thread_count: Optional[int]
    file_descriptors: Optional[int]
    
    # Metadata
    collected_by: Optional[str]
    collection_duration_ms: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


class VersionHistoryResponse(BaseModel):
    """Response model for version history."""
    version_id: str
    organization_id: str
    orchestrator_version: str
    previous_version: Optional[str]
    component_name: str
    component_type: Optional[str]
    update_type: Optional[str]
    update_timestamp: datetime
    update_duration_seconds: Optional[int]
    update_status: Optional[str]
    deployment_method: Optional[str]
    deployed_by: Optional[str]
    deployment_notes: Optional[str]
    git_commit: Optional[str]
    git_branch: Optional[str]
    git_tag: Optional[str]
    build_number: Optional[str]
    is_rollback: Optional[bool]
    rollback_from_version: Optional[str]
    rollback_reason: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class SystemAlertResponse(BaseModel):
    """Response model for system alerts."""
    alert_id: str
    user_id: Optional[str]
    organization_id: str
    alert_type: str
    alert_severity: str
    alert_name: str
    alert_description: Optional[str]
    metric_name: str
    threshold_value: float
    actual_value: float
    threshold_operator: Optional[str]
    alert_status: Optional[str]
    triggered_at: datetime
    acknowledged_at: Optional[datetime]
    resolved_at: Optional[datetime]
    resolved_by: Optional[str]
    resolution_notes: Optional[str]
    auto_resolved: Optional[bool]
    estimated_impact: Optional[str]
    affected_users_count: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# API ENDPOINTS - SYSTEM METRICS COLLECTION
# =============================================================================

# -----------------------------------------------------------------------------
# PRIMARY ORGANIZATION-BASED ENDPOINTS (Used by test interface)
# -----------------------------------------------------------------------------

@router.post("/collect/organization/{organization_id}")
async def collect_organization_system_metrics(
    organization_id: str,
    db: AsyncSession = Depends(get_db)
) -> SystemMetricsResponse:
    """
    Collect and store current system performance metrics for an organization.
    
    This is the primary endpoint used for organization-wide system monitoring.
    It tracks CPU, memory, storage, and other system resources at the 
    organizational level rather than per-user.
    """
    try:
        # Convert organization ID to UUID
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            # For non-UUID strings like "test-org-12345", generate a deterministic UUID
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        metrics = await system_metrics_service.record_organization_system_metrics(organization_id, db)
        return SystemMetricsResponse.model_validate(metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error collecting system metrics: {str(e)}")


# -----------------------------------------------------------------------------
# LEGACY USER-BASED ENDPOINTS (Kept for backward compatibility)
# -----------------------------------------------------------------------------

@router.post("/collect/{user_id}")
async def collect_system_metrics(
    user_id: str,
    organization_id: str = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db)
) -> SystemMetricsResponse:
    """
    LEGACY: Collect and store current system performance metrics for a user.
    
    Note: This endpoint is kept for backward compatibility. New implementations
    should use the organization-based endpoint above.
    """
    try:
        # Convert string IDs to UUIDs - generate UUID if not valid UUID format
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            # For non-UUID strings like "alice-doe-123", generate a deterministic UUID
            user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
        
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            # For non-UUID strings like "test-org-12345", generate a deterministic UUID
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        metrics = await system_metrics_service.record_system_metrics(user_uuid, org_uuid, db)
        return SystemMetricsResponse.model_validate(metrics)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error collecting system metrics: {str(e)}")


# -----------------------------------------------------------------------------
# DATA RETRIEVAL ENDPOINTS
# -----------------------------------------------------------------------------

@router.get("/metrics/organization/{organization_id}")
async def get_organization_system_metrics(
    organization_id: str,
    hours_back: int = Query(24, description="Hours of historical data to retrieve"),
    limit: int = Query(500, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db)
) -> List[SystemMetricsResponse]:
    """PRIMARY: Get system performance metrics for an entire organization."""
    try:
        # Convert organization ID to UUID
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = select(UserSystemPerformance).where(
            UserSystemPerformance.organization_id == organization_id,
            UserSystemPerformance.timestamp >= cutoff_time
        ).order_by(desc(UserSystemPerformance.timestamp)).limit(limit)
        
        result = await db.execute(query)
        metrics = result.scalars().all()
        
        return [SystemMetricsResponse.model_validate(metric) for metric in metrics]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving organization metrics: {str(e)}")


@router.get("/metrics/organization/{organization_id}/latest")
async def get_latest_organization_metrics(
    organization_id: str,
    db: AsyncSession = Depends(get_db)
) -> Optional[SystemMetricsResponse]:
    """Get the most recent system metrics for an organization."""
    try:
        # Convert organization ID to UUID
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        # Get the system user for this organization
        system_user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, f"system-{organization_id}")
            
        query = select(UserSystemPerformance).where(
            UserSystemPerformance.organization_id == organization_id,
            UserSystemPerformance.user_id == str(system_user_uuid)
        ).order_by(desc(UserSystemPerformance.timestamp)).limit(1)
        
        result = await db.execute(query)
        metric = result.scalar_one_or_none()
        
        if metric:
            return SystemMetricsResponse.model_validate(metric)
        return None
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving latest organization metrics: {str(e)}")


# -----------------------------------------------------------------------------
# LEGACY USER-BASED DATA RETRIEVAL (Kept for backward compatibility)
# -----------------------------------------------------------------------------

@router.get("/metrics/users/{user_id}")
async def get_user_system_metrics(
    user_id: str,
    hours_back: int = Query(24, description="Hours of historical data to retrieve"),
    limit: int = Query(100, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db)
) -> List[SystemMetricsResponse]:
    """LEGACY: Get system performance metrics for a specific user."""
    try:
        # Convert string ID to UUID
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
            
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = select(UserSystemPerformance).where(
            UserSystemPerformance.user_id == user_uuid,
            UserSystemPerformance.timestamp >= cutoff_time
        ).order_by(desc(UserSystemPerformance.timestamp)).limit(limit)
        
        result = await db.execute(query)
        metrics = result.scalars().all()
        
        return [SystemMetricsResponse.model_validate(metric) for metric in metrics]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system metrics: {str(e)}")


@router.get("/metrics/users/{user_id}/latest")
async def get_latest_user_metrics(
    user_id: str,
    db: AsyncSession = Depends(get_db)
) -> Optional[SystemMetricsResponse]:
    """Get the most recent system metrics for a user."""
    try:
        # Convert string ID to UUID
        try:
            user_uuid = uuid.UUID(user_id)
        except ValueError:
            user_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, user_id)
            
        query = select(UserSystemPerformance).where(
            UserSystemPerformance.user_id == user_uuid
        ).order_by(desc(UserSystemPerformance.timestamp)).limit(1)
        
        result = await db.execute(query)
        metric = result.scalar_one_or_none()
        
        if metric:
            return SystemMetricsResponse.model_validate(metric)
        return None
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving latest metrics: {str(e)}")


@router.get("/metrics/summary/cpu")
async def get_cpu_utilization_summary(
    organization_id: str = Query(..., description="Organization ID"),
    hours_back: int = Query(24, description="Hours to analyze"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get CPU utilization summary for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = select(
            func.avg(UserSystemPerformance.cpu_usage_percent).label('avg_cpu'),
            func.max(UserSystemPerformance.cpu_usage_percent).label('max_cpu'),
            func.min(UserSystemPerformance.cpu_usage_percent).label('min_cpu'),
            func.count().label('sample_count')
        ).where(
            UserSystemPerformance.organization_id == organization_id,
            UserSystemPerformance.timestamp >= cutoff_time,
            UserSystemPerformance.cpu_usage_percent.is_not(None)
        )
        
        result = await db.execute(query)
        summary = result.first()
        
        return {
            "organization_id": organization_id,
            "time_period_hours": hours_back,
            "avg_cpu_percent": round(summary.avg_cpu, 2) if summary.avg_cpu else None,
            "max_cpu_percent": summary.max_cpu,
            "min_cpu_percent": summary.min_cpu,
            "sample_count": summary.sample_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving CPU summary: {str(e)}")


@router.get("/metrics/summary/memory")
async def get_memory_utilization_summary(
    organization_id: str = Query(..., description="Organization ID"),
    hours_back: int = Query(24, description="Hours to analyze"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get memory utilization summary for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = select(
            func.avg(UserSystemPerformance.memory_percent).label('avg_memory'),
            func.max(UserSystemPerformance.memory_percent).label('max_memory'),
            func.min(UserSystemPerformance.memory_percent).label('min_memory'),
            func.avg(UserSystemPerformance.memory_usage_mb).label('avg_memory_mb'),
            func.count().label('sample_count')
        ).where(
            UserSystemPerformance.organization_id == organization_id,
            UserSystemPerformance.timestamp >= cutoff_time,
            UserSystemPerformance.memory_percent.is_not(None)
        )
        
        result = await db.execute(query)
        summary = result.first()
        
        return {
            "organization_id": organization_id,
            "time_period_hours": hours_back,
            "avg_memory_percent": round(summary.avg_memory, 2) if summary.avg_memory else None,
            "max_memory_percent": summary.max_memory,
            "min_memory_percent": summary.min_memory,
            "avg_memory_usage_mb": int(summary.avg_memory_mb) if summary.avg_memory_mb else None,
            "sample_count": summary.sample_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving memory summary: {str(e)}")


@router.get("/metrics/summary/storage")
async def get_storage_utilization_summary(
    organization_id: str = Query(..., description="Organization ID"),
    hours_back: int = Query(24, description="Hours to analyze"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get storage utilization summary for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        query = select(
            func.avg(UserSystemPerformance.storage_percent).label('avg_storage'),
            func.max(UserSystemPerformance.storage_percent).label('max_storage'),
            func.min(UserSystemPerformance.storage_percent).label('min_storage'),
            func.avg(UserSystemPerformance.storage_usage_gb).label('avg_storage_gb'),
            func.count().label('sample_count')
        ).where(
            UserSystemPerformance.organization_id == organization_id,
            UserSystemPerformance.timestamp >= cutoff_time,
            UserSystemPerformance.storage_percent.is_not(None)
        )
        
        result = await db.execute(query)
        summary = result.first()
        
        return {
            "organization_id": organization_id,
            "time_period_hours": hours_back,
            "avg_storage_percent": round(summary.avg_storage, 2) if summary.avg_storage else None,
            "max_storage_percent": summary.max_storage,
            "min_storage_percent": summary.min_storage,
            "avg_storage_usage_gb": round(summary.avg_storage_gb, 2) if summary.avg_storage_gb else None,
            "sample_count": summary.sample_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving storage summary: {str(e)}")


@router.get("/versions/organization/{organization_id}")
async def get_orchestrator_versions(
    organization_id: str,
    limit: int = Query(50, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db)
) -> List[VersionHistoryResponse]:
    """Get orchestrator version history for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        query = select(OrchestratorVersionHistory).where(
            OrchestratorVersionHistory.organization_id == organization_id
        ).order_by(desc(OrchestratorVersionHistory.update_timestamp)).limit(limit)
        
        result = await db.execute(query)
        versions = result.scalars().all()
        
        return [VersionHistoryResponse.model_validate(version) for version in versions]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving version history: {str(e)}")


@router.get("/versions/component/{component_name}")
async def get_component_versions(
    component_name: str,
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(20, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db)
) -> List[VersionHistoryResponse]:
    """Get version history for a specific component."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        query = select(OrchestratorVersionHistory).where(
            OrchestratorVersionHistory.organization_id == organization_id,
            OrchestratorVersionHistory.component_name == component_name
        ).order_by(desc(OrchestratorVersionHistory.update_timestamp)).limit(limit)
        
        result = await db.execute(query)
        versions = result.scalars().all()
        
        return [VersionHistoryResponse.model_validate(version) for version in versions]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving component versions: {str(e)}")


@router.post("/versions/record")
async def record_version_update(
    organization_id: str,
    component_name: str,
    new_version: str,
    update_type: str = "upgrade",
    deployment_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
) -> VersionHistoryResponse:
    """Record a new version update."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        version_record = await system_metrics_service.version_tracker.record_version_update(
            organization_id=organization_id,
            component_name=component_name,
            new_version=new_version,
            update_type=update_type,
            db=db
        )
        
        if deployment_notes:
            version_record.deployment_notes = deployment_notes
            await db.commit()
        
        return VersionHistoryResponse.model_validate(version_record)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording version update: {str(e)}")


@router.get("/alerts/organization/{organization_id}")
async def get_system_alerts(
    organization_id: str,
    status: Optional[str] = Query(None, description="Filter by alert status"),
    severity: Optional[str] = Query(None, description="Filter by alert severity"),
    alert_type: Optional[str] = Query(None, description="Filter by alert type"),
    limit: int = Query(100, description="Maximum number of alerts to return"),
    db: AsyncSession = Depends(get_db)
) -> List[SystemAlertResponse]:
    """Get system alerts for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        query = select(SystemAlert).where(
            SystemAlert.organization_id == organization_id
        )
        
        if status:
            query = query.where(SystemAlert.alert_status == status)
        if severity:
            query = query.where(SystemAlert.alert_severity == severity)
        if alert_type:
            query = query.where(SystemAlert.alert_type == alert_type)
        
        query = query.order_by(desc(SystemAlert.triggered_at)).limit(limit)
        
        result = await db.execute(query)
        alerts = result.scalars().all()
        
        return [SystemAlertResponse.model_validate(alert) for alert in alerts]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving system alerts: {str(e)}")


# =============================================================================
# MANUAL COLLECTION ENDPOINTS
# =============================================================================

@router.post("/collect/immediate")
async def collect_system_metrics_immediate(
    organization_id: Optional[str] = Query(None, description="Organization ID (defaults to org_001)"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Force immediate collection of system metrics for an organization."""
    try:
        # Use default organization if none provided
        if organization_id is None:
            organization_id = "org_001"
        
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            # For non-UUID strings like "org_001", generate a deterministic UUID
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        # Record metrics directly using the service
        metrics_record = await system_metrics_service.record_organization_system_metrics(
            organization_id=organization_id,  # Use original string, not UUID
            db=db
        )
        
        return {
            "status": "success",
            "message": "System metrics collected successfully",
            "metric_id": str(metrics_record.metric_id),
            "organization_id": str(metrics_record.organization_id),
            "timestamp": metrics_record.timestamp.isoformat(),
            "cpu_usage_percent": metrics_record.cpu_usage_percent,
            "memory_percent": metrics_record.memory_percent,
            "storage_percent": metrics_record.storage_percent,
            "collection_duration_ms": metrics_record.collection_duration_ms
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error collecting system metrics: {str(e)}")


@router.get("/status/collection")
async def get_collection_status(
    organization_id: str = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get the status of system metrics collection for an organization."""
    try:
        # Convert organization ID to UUID if needed
        try:
            org_uuid = uuid.UUID(organization_id)
        except ValueError:
            # For non-UUID strings like "org_001", generate a deterministic UUID
            org_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, organization_id)
        
        # Get latest metrics to check collection status
        latest_metrics = await system_metrics_service.get_latest_organization_system_metrics(
            organization_id=organization_id,  # Use original string, not UUID
            db=db
        )
        
        # Count total metrics for this organization
        count_query = select(func.count(UserSystemPerformance.metric_id)).where(
            UserSystemPerformance.organization_id == organization_id
        )
        result = await db.execute(count_query)
        total_records = result.scalar()
        
        # Calculate time since last collection
        time_since_last = None
        if latest_metrics:
            time_diff = datetime.utcnow() - latest_metrics.timestamp
            time_since_last = int(time_diff.total_seconds())
        
        return {
            "organization_id": str(organization_id),
            "collection_active": latest_metrics is not None,
            "total_metrics_records": total_records,
            "last_collection_timestamp": latest_metrics.timestamp.isoformat() if latest_metrics else None,
            "seconds_since_last_collection": time_since_last,
            "latest_metrics": {
                "cpu_percent": latest_metrics.cpu_usage_percent,
                "memory_percent": latest_metrics.memory_percent,
                "storage_percent": latest_metrics.storage_percent
            } if latest_metrics else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting collection status: {str(e)}")


@router.get("/status/background")
async def get_background_collection_status() -> Dict[str, Any]:
    """Get the status of background system metrics collection."""
    try:
        from ...config.settings import config
        
        # Return actual configuration values
        return {
            "background_collection_enabled": True,
            "global_scheduler_active": True,
            "collection_interval_seconds": config.system_metrics_interval,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Background collection should be active if services started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting background status: {str(e)}")


@router.get("/health")
async def system_metrics_health() -> Dict[str, str]:
    """Health check endpoint for system metrics service."""
    return {
        "status": "healthy",
        "service": "system-metrics",
        "timestamp": datetime.utcnow().isoformat()
    }