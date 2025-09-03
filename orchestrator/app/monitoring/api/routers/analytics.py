"""Analytics API endpoints for dashboard metrics - Direct PostgreSQL Phoenix queries."""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select
import asyncio
import os
import logging

# Import orchestrator database session since Phoenix data is in orchestrator DB
from ....db.database import db_manager

async def get_orchestrator_db():
    """Get orchestrator database session where Phoenix data resides."""
    async for session in db_manager.get_session():
        yield session

logger = logging.getLogger(__name__)

# We'll query Phoenix data directly from PostgreSQL
PHOENIX_SCHEMA = "phoenix"

router = APIRouter()


class PhoenixAnalyticsService:
    """Service to query Phoenix data directly from PostgreSQL database."""
    
    def __init__(self):
        """Initialize Phoenix Analytics Service to query PostgreSQL directly."""
        # We'll use the orchestrator database connection to query Phoenix schema
        self.phoenix_schema = PHOENIX_SCHEMA
        logger.info(f"Phoenix Analytics Service initialized to query PostgreSQL schema: {self.phoenix_schema}")
    
    def _empty_response(self, start_date: datetime, end_date: datetime, error: str = None) -> Dict[str, Any]:
        """Return empty analytics response with error message."""
        response = {
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "overview": {
                "total_api_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "avg_response_time_ms": 0,
                "cache_hit_rate": 0.0,
                "firewall_blocks": 0
            },
            "provider_breakdown": [],
            "data_source": "phoenix_native",
            "phoenix_available": True,  # Always true since we're using PostgreSQL
            "phoenix_connected": True
        }
        if error:
            response["error"] = error
        return response
    
    def _generate_fallback_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Return empty data structure when Phoenix is unavailable."""
        # Only used when Phoenix is truly unavailable
        # Real data should come from Phoenix
        return {
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "overview": {
                "total_api_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "avg_response_time_ms": 0,
                "cache_hit_rate": 0.0,
                "firewall_blocks": 0
            },
            "provider_breakdown": [],
            "data_source": "no_data",
            "phoenix_available": PHOENIX_AVAILABLE,
            "phoenix_connected": self.connection_validated,
            "message": "No data available. Make some API calls to see analytics."
        }
    
    async def get_analytics_overview_from_phoenix(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        organization_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get analytics overview data from Phoenix native schema."""
        if not db:
            return self._empty_response(start_date, end_date, "Database connection required")
        
        try:
            # Query Phoenix native schema - look for spans with LLM data
            query = text("""
                WITH llm_spans AS (
                    SELECT 
                        s.*,
                        -- Extract from Phoenix's native attributes structure
                        (s.attributes->'gen_ai'->'usage'->>'prompt_tokens')::INTEGER as prompt_tokens,
                        (s.attributes->'gen_ai'->'usage'->>'completion_tokens')::INTEGER as completion_tokens,
                        (s.attributes->'gen_ai'->'usage'->>'prompt_tokens')::INTEGER + (s.attributes->'gen_ai'->'usage'->>'completion_tokens')::INTEGER as total_tokens,
                        s.attributes->'gen_ai'->'request'->>'model' as model_name,
                        s.attributes->'gen_ai'->>'system' as provider,
                        EXTRACT(EPOCH FROM (s.end_time - s.start_time)) * 1000 as duration_ms,
                        COALESCE(sc.total_cost, 
                            -- Try MoolAI cost attribute first
                            COALESCE((s.attributes->'moolai'->>'cost')::FLOAT, 
                                -- Try nested MoolAI llm cost
                                COALESCE((s.attributes->'moolai'->'llm'->>'cost')::FLOAT,
                                    -- Try direct cost attribute
                                    (s.attributes->>'cost')::FLOAT, 0)
                            )
                        ) as cost
                    FROM phoenix.spans s
                    LEFT JOIN phoenix.span_costs sc ON s.id = sc.span_rowid
                    WHERE (
                        -- Only include actual LLM provider API calls (not internal operations)
                        (s.name ILIKE 'openai.%' AND s.attributes ? 'gen_ai') OR
                        (s.name ILIKE 'anthropic.%' AND s.attributes ? 'gen_ai') OR
                        (s.name ILIKE 'cohere.%' AND s.attributes ? 'gen_ai') OR
                        -- Include spans with proper LLM provider system classification
                        (s.attributes->'gen_ai'->>'system' IN ('openai', 'anthropic', 'cohere', 'azure')) OR
                        -- Include spans with OpenAI-specific attributes
                        (s.attributes ? 'openai' AND s.attributes ? 'gen_ai')
                        -- Exclude internal MoolAI operations: moolai.firewall.*, moolai.cache.*, moolai.request.*, etc.
                    )
                        AND s.start_time >= :start_time
                        AND s.start_time <= :end_time
                ),
                analytics_summary AS (
                    SELECT 
                        COUNT(*) as total_api_calls,
                        SUM(cost) as total_cost,
                        SUM(COALESCE(total_tokens, 0)) as total_tokens,
                        AVG(duration_ms)::INTEGER as avg_response_time_ms
                    FROM llm_spans
                    WHERE 1=1  -- Include all LLM spans regardless of token counts
                ),
                cache_summary AS (
                    -- Separate query for cache hits from MoolAI cache spans
                    -- Use only moolai.cache.lookup spans to avoid double-counting
                    -- (both cache.lookup and request.process spans have same cache data)
                    SELECT 
                        COUNT(*) as total_cache_requests,
                        COUNT(*) FILTER (WHERE 
                            (s.attributes->'moolai'->'cache'->>'hit')::boolean = true
                        ) as cache_hits,
                        CASE 
                            WHEN COUNT(*) > 0 THEN 
                                (COUNT(*) FILTER (WHERE 
                                    (s.attributes->'moolai'->'cache'->>'hit')::boolean = true
                                )) * 100.0 / COUNT(*)
                            ELSE 0 
                        END as cache_hit_rate
                    FROM phoenix.spans s
                    WHERE s.name = 'moolai.cache.lookup'
                    AND s.start_time >= :start_time
                    AND s.start_time <= :end_time
                ),
                firewall_summary AS (
                    -- Separate query for firewall blocks from MoolAI firewall spans
                    -- Use only moolai.firewall.scan spans to avoid double-counting
                    -- (both firewall.scan and request.process spans have same firewall data)
                    SELECT 
                        COUNT(*) FILTER (WHERE 
                            (s.attributes->'moolai'->'firewall'->>'blocked')::boolean = true
                        ) as firewall_blocks
                    FROM phoenix.spans s
                    WHERE s.name = 'moolai.firewall.scan'
                    AND s.start_time >= :start_time
                    AND s.start_time <= :end_time
                ),
                provider_stats AS (
                    SELECT 
                        COALESCE(provider, 'openai') as provider,
                        COALESCE(model_name, 'gpt-3.5-turbo') as model,
                        COUNT(*) as calls,
                        SUM(COALESCE(total_tokens, 0)) as tokens,
                        SUM(cost) as cost
                    FROM llm_spans
                    WHERE 1=1  -- Include all LLM spans regardless of token counts
                    GROUP BY COALESCE(provider, 'openai'), COALESCE(model_name, 'gpt-3.5-turbo')
                )
                SELECT 
                    s.total_api_calls,
                    s.total_cost,
                    s.total_tokens,
                    s.avg_response_time_ms,
                    c.cache_hit_rate,
                    f.firewall_blocks,
                    jsonb_agg(
                        jsonb_build_object(
                            'provider', p.provider,
                            'model', p.model,
                            'calls', p.calls,
                            'tokens', p.tokens,
                            'cost', p.cost
                        )
                    ) as provider_breakdown
                FROM analytics_summary s
                CROSS JOIN cache_summary c
                CROSS JOIN firewall_summary f
                CROSS JOIN provider_stats p
                GROUP BY s.total_api_calls, s.total_cost, s.total_tokens, 
                         s.avg_response_time_ms, c.cache_hit_rate, f.firewall_blocks;
            """)
            
            result = await db.execute(query, {
                'start_time': start_date,
                'end_time': end_date
            })
            
            row = result.fetchone()
            
            if row and row.total_api_calls > 0:
                return {
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "overview": {
                        "total_api_calls": int(row.total_api_calls or 0),
                        "total_cost": float(row.total_cost or 0),
                        "total_tokens": int(row.total_tokens or 0),
                        "avg_response_time_ms": int(row.avg_response_time_ms or 0),
                        "cache_hit_rate": float(row.cache_hit_rate or 0),
                        "firewall_blocks": int(row.firewall_blocks or 0)
                    },
                    "provider_breakdown": row.provider_breakdown or [],
                    "data_source": "phoenix_native",
                    "phoenix_available": True,
                    "phoenix_connected": True
                }
            else:
                logger.info(f"No LLM data found in Phoenix between {start_date} and {end_date}")
                return self._empty_response(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Phoenix native query error: {e}")
            return self._empty_response(start_date, end_date, str(e))
    
    async def get_provider_breakdown_from_phoenix(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        organization_id: Optional[str] = None,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """Get provider breakdown from Phoenix PostgreSQL database."""
        try:
            project_name = f"moolai-org_001"  # Default org for now
            if organization_id:
                project_name = f"moolai-{organization_id}"

            # Query provider breakdown from Phoenix native schema
            query = text("""
                WITH llm_spans AS (
                    SELECT 
                        -- Extract from Phoenix's native attributes structure
                        (s.attributes->'gen_ai'->'usage'->>'prompt_tokens')::INTEGER as prompt_tokens,
                        (s.attributes->'gen_ai'->'usage'->>'completion_tokens')::INTEGER as completion_tokens,
                        (s.attributes->'gen_ai'->'usage'->>'prompt_tokens')::INTEGER + (s.attributes->'gen_ai'->'usage'->>'completion_tokens')::INTEGER as total_tokens,
                        s.attributes->'gen_ai'->'request'->>'model' as model_name,
                        s.attributes->'gen_ai'->>'system' as provider,
                        EXTRACT(EPOCH FROM (s.end_time - s.start_time)) * 1000 as duration_ms,
                        COALESCE(sc.total_cost, 
                            -- Try MoolAI cost attribute first
                            COALESCE((s.attributes->'moolai'->>'cost')::FLOAT, 
                                -- Try nested MoolAI llm cost
                                COALESCE((s.attributes->'moolai'->'llm'->>'cost')::FLOAT,
                                    -- Try direct cost attribute
                                    (s.attributes->>'cost')::FLOAT, 0)
                            )
                        ) as total_cost
                    FROM phoenix.spans s
                    LEFT JOIN phoenix.span_costs sc ON s.id = sc.span_rowid
                    WHERE (
                        -- Only include actual LLM provider API calls (not internal operations)
                        (s.name ILIKE 'openai.%' AND s.attributes ? 'gen_ai') OR
                        (s.name ILIKE 'anthropic.%' AND s.attributes ? 'gen_ai') OR
                        (s.name ILIKE 'cohere.%' AND s.attributes ? 'gen_ai') OR
                        -- Include spans with proper LLM provider system classification
                        (s.attributes->'gen_ai'->>'system' IN ('openai', 'anthropic', 'cohere', 'azure')) OR
                        -- Include spans with OpenAI-specific attributes
                        (s.attributes ? 'openai' AND s.attributes ? 'gen_ai')
                        -- Exclude internal MoolAI operations: moolai.firewall.*, moolai.cache.*, moolai.request.*, etc.
                    )
                        AND s.start_time >= :start_time
                        AND s.start_time <= :end_time
                        AND (COALESCE((s.attributes->'gen_ai'->'usage'->>'prompt_tokens')::INTEGER, 0) > 0 
                             OR COALESCE((s.attributes->'gen_ai'->'usage'->>'completion_tokens')::INTEGER, 0) > 0 
                             OR s.attributes ? 'moolai.session_id')
                ),
                provider_stats AS (
                    SELECT 
                        COALESCE(provider, 'openai') as provider,
                        COALESCE(model_name, 'gpt-3.5-turbo') as model_name,
                        COUNT(*) as call_count,
                        SUM(COALESCE(total_tokens, 0)) as total_tokens,
                        SUM(COALESCE(prompt_tokens, 0)) as prompt_tokens,
                        SUM(COALESCE(completion_tokens, 0)) as completion_tokens,
                        SUM(total_cost) as total_cost,
                        AVG(duration_ms)::INTEGER as avg_latency
                    FROM llm_spans
                    GROUP BY provider, model_name
                )
                SELECT 
                    provider,
                    model_name,
                    call_count,
                    total_tokens,
                    prompt_tokens,
                    completion_tokens,
                    total_cost,
                    avg_latency
                FROM provider_stats
                ORDER BY call_count DESC;
            """)
            
            if db:
                result = await db.execute(query, {
                    'start_time': start_date,
                    'end_time': end_date
                })
                rows = result.fetchall()
                
                provider_breakdown = []
                for row in rows:
                    provider_breakdown.append({
                        "provider": row.provider,
                        "model": row.model_name,
                        "calls": int(row.call_count),
                        "tokens": int(row.total_tokens or 0),
                        "prompt_tokens": int(row.prompt_tokens or 0),
                        "completion_tokens": int(row.completion_tokens or 0),
                        "cost": float(row.total_cost or 0),
                        "avg_latency_ms": int(row.avg_latency or 0)
                    })
                
                return {
                    "time_range": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    },
                    "provider_breakdown": provider_breakdown,
                    "data_source": "phoenix_postgresql"
                }
            else:
                return {
                    "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                    "provider_breakdown": [],
                    "data_source": "phoenix_postgresql",
                    "error": "Database session not available"
                }
                
        except Exception as e:
            logger.error(f"Phoenix provider breakdown query error: {e}")
            return {
                "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "provider_breakdown": [],
                "data_source": "phoenix_postgresql",
                "error": str(e)
            }
    
    async def get_time_series_from_phoenix(
        self,
        metric: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get time series data from Phoenix."""
        if not self.client:
            return {
                "metric": metric,
                "interval": interval,
                "time_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "data": [],
                "data_source": "phoenix_native",
                "error": "Phoenix client not available"
            }
        
        # For now, return empty time series data
        # This could be enhanced to query actual time series data from Phoenix
        return {
            "metric": metric,
            "interval": interval,
            "time_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "data": [],
            "data_source": "phoenix_native"
        }
    
    async def get_provider_breakdown_from_langfuse(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get provider breakdown from Langfuse and format for existing dashboard."""
        if not self.langfuse_client:
            raise HTTPException(status_code=503, detail="Langfuse client not available")
        
        try:
            traces = self.langfuse_client.fetch_traces(
                from_timestamp=start_date,
                to_timestamp=end_date,
                user_id=organization_id if organization_id else None
            )
            
            # Group by provider and model
            provider_stats = {}
            for trace in traces:
                model = trace.metadata.get('model', 'unknown')
                provider = 'openai' if 'gpt' in model.lower() else 'anthropic' if 'claude' in model.lower() else 'unknown'
                
                key = f"{provider}_{model}"
                if key not in provider_stats:
                    provider_stats[key] = {
                        "provider": provider,
                        "model": model,
                        "calls": 0,
                        "cost": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "latencies": []
                    }
                
                provider_stats[key]["calls"] += 1
                provider_stats[key]["cost"] += trace.cost or 0
                if hasattr(trace, 'usage') and trace.usage:
                    provider_stats[key]["input_tokens"] += trace.usage.input or 0
                    provider_stats[key]["output_tokens"] += trace.usage.output or 0
                if hasattr(trace, 'latency') and trace.latency:
                    provider_stats[key]["latencies"].append(trace.latency)
            
            # Format breakdown
            breakdown = []
            for stats in provider_stats.values():
                avg_latency = sum(stats["latencies"]) / len(stats["latencies"]) if stats["latencies"] else 0
                avg_cost_per_query = stats["cost"] / stats["calls"] if stats["calls"] > 0 else 0
                
                breakdown.append({
                    "provider": stats["provider"],
                    "model": stats["model"],
                    "calls": stats["calls"],
                    "cost": float(stats["cost"]),
                    "input_tokens": stats["input_tokens"],
                    "output_tokens": stats["output_tokens"],
                    "total_tokens": stats["input_tokens"] + stats["output_tokens"],
                    "avg_latency_ms": int(avg_latency),
                    "avg_cost_per_query": avg_cost_per_query
                })
            
            # Sort by cost descending
            breakdown.sort(key=lambda x: x["cost"], reverse=True)
            
            return {
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "provider_breakdown": breakdown,
                "data_source": "langfuse"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query Langfuse provider breakdown: {str(e)}")
    
    async def get_time_series_from_langfuse(
        self,
        metric: str,
        interval: str,
        start_date: datetime,
        end_date: datetime,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get time series data from Langfuse."""
        if not self.langfuse_client:
            raise HTTPException(status_code=503, detail="Langfuse client not available")
        
        try:
            traces = self.langfuse_client.fetch_traces(
                from_timestamp=start_date,
                to_timestamp=end_date,
                user_id=organization_id if organization_id else None
            )
            
            # Group traces by time buckets
            from collections import defaultdict
            buckets = defaultdict(list)
            
            for trace in traces:
                timestamp = trace.timestamp
                if interval == "hour":
                    bucket = timestamp.replace(minute=0, second=0, microsecond=0)
                else:  # day
                    bucket = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
                
                buckets[bucket].append(trace)
            
            # Calculate metrics for each bucket
            time_series = []
            for bucket_time in sorted(buckets.keys()):
                bucket_traces = buckets[bucket_time]
                
                if metric == "cost":
                    value = sum(trace.cost or 0 for trace in bucket_traces)
                elif metric == "calls":
                    value = len(bucket_traces)
                elif metric == "tokens":
                    value = sum(
                        trace.usage.total_tokens if hasattr(trace, 'usage') and trace.usage else 0
                        for trace in bucket_traces
                    )
                elif metric == "latency":
                    latencies = [trace.latency for trace in bucket_traces if hasattr(trace, 'latency') and trace.latency]
                    value = sum(latencies) / len(latencies) if latencies else 0
                else:
                    value = 0
                
                time_series.append({
                    "timestamp": bucket_time.isoformat(),
                    "value": float(value) if metric != "calls" else int(value)
                })
            
            return {
                "metric": metric,
                "interval": interval,
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "data": time_series,
                "data_source": "langfuse"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query Langfuse time series: {str(e)}")

# Initialize Phoenix analytics service
phoenix_analytics = PhoenixAnalyticsService()


@router.get("/analytics/overview")
async def get_analytics_overview(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    organization_id: Optional[str] = Query(None),
    use_phoenix: bool = Query(True, description="Use Phoenix backend for analytics (legacy DB disabled)"),
    db: AsyncSession = Depends(get_orchestrator_db)
):
    """Get comprehensive analytics overview for the dashboard using Langfuse backend."""
    try:
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # If end_date is at midnight (from date-only input), set to end of day
        if end_date.time() == datetime.min.time():
            logger.info(f"Adjusting end_date from midnight to end-of-day: {end_date} -> {end_date.replace(hour=23, minute=59, second=59, microsecond=999999)}")
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # If start_date is at midnight (from date-only input), keep as beginning of day
        if start_date.time() == datetime.min.time():
            logger.info(f"Confirmed start_date at beginning-of-day: {start_date}")
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Always use Phoenix backend (legacy database removed)
        if use_phoenix:
            return await phoenix_analytics.get_analytics_overview_from_phoenix(
                start_date, end_date, organization_id, db
            )
        else:
            # Return message for legacy mode
            return {
                "message": "Legacy database monitoring has been removed. Using Phoenix backend.",
                "redirect": "Set use_phoenix=true to use Phoenix analytics",
                "data_source": "legacy_disabled",
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "overview": {
                    "total_api_calls": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "avg_response_time_ms": 0,
                    "cache_hit_rate": 0.0,
                    "firewall_blocks": 0
                },
                "provider_breakdown": []
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/provider-breakdown")
async def get_provider_breakdown(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    organization_id: Optional[str] = Query(None),
    use_phoenix: bool = Query(True, description="Use Phoenix backend for analytics (legacy DB disabled)"),
    db: AsyncSession = Depends(get_orchestrator_db)
):
    """Get detailed provider breakdown for API calls and costs using Langfuse backend."""
    try:
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # If end_date is at midnight (from date-only input), set to end of day
        if end_date.time() == datetime.min.time():
            logger.info(f"Adjusting end_date from midnight to end-of-day: {end_date} -> {end_date.replace(hour=23, minute=59, second=59, microsecond=999999)}")
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # If start_date is at midnight (from date-only input), keep as beginning of day
        if start_date.time() == datetime.min.time():
            logger.info(f"Confirmed start_date at beginning-of-day: {start_date}")
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Always use Phoenix backend (legacy database removed)
        if use_phoenix:
            return await phoenix_analytics.get_provider_breakdown_from_phoenix(
                start_date, end_date, organization_id, db
            )
        else:
            # Return message for legacy mode
            return {
                "message": "Legacy database monitoring has been removed. Using Phoenix backend.",
                "redirect": "Set use_phoenix=true to use Phoenix analytics",
                "data_source": "legacy_disabled",
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "provider_breakdown": []
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/time-series")
async def get_time_series_data(
    metric: str = Query("cost", regex="^(cost|calls|tokens|latency)$"),
    interval: str = Query("hour", regex="^(hour|day)$"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    organization_id: Optional[str] = Query(None),
    use_phoenix: bool = Query(True, description="Use Phoenix backend for analytics (legacy DB disabled)"),
    db: AsyncSession = Depends(get_orchestrator_db)
):
    """Get time series data for specified metric using Langfuse backend."""
    try:
        # Default to last 24 hours if no dates provided
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            if interval == "hour":
                start_date = end_date - timedelta(hours=24)
            else:
                start_date = end_date - timedelta(days=30)
        
        # If end_date is at midnight (from date-only input), set to end of day
        if end_date.time() == datetime.min.time():
            logger.info(f"Adjusting end_date from midnight to end-of-day: {end_date} -> {end_date.replace(hour=23, minute=59, second=59, microsecond=999999)}")
            end_date = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # If start_date is at midnight (from date-only input), keep as beginning of day
        if start_date.time() == datetime.min.time():
            logger.info(f"Confirmed start_date at beginning-of-day: {start_date}")
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Always use Phoenix backend (legacy database removed)
        if use_phoenix:
            return await phoenix_analytics.get_time_series_from_phoenix(
                metric, interval, start_date, end_date, organization_id
            )
        else:
            # Return message for legacy mode
            return {
                "message": "Legacy database monitoring has been removed. Using Phoenix backend.",
                "redirect": "Set use_phoenix=true to use Phoenix analytics",
                "data_source": "legacy_disabled",
                "metric": metric,
                "interval": interval,
                "time_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "data": []
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/cache-performance")
async def get_cache_performance(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    organization_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_orchestrator_db)
):
    """Legacy cache performance endpoint - data now available through Langfuse analytics."""
    return {
        "message": "Legacy cache monitoring has been removed. Cache performance is now tracked in Langfuse traces.",
        "redirect": "Use /analytics/overview?use_langfuse=true to see cache hit rates",
        "time_range": {
            "start": start_date.isoformat() if start_date else datetime.now(timezone.utc).isoformat(),
            "end": end_date.isoformat() if end_date else datetime.now(timezone.utc).isoformat()
        },
        "cache_performance": {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "cache_miss_rate": 0.0,
            "avg_similarity": 0.0,
            "avg_cache_latency_ms": 0,
            "avg_fresh_latency_ms": 0
        },
        "similarity_breakdown": []
    }


@router.get("/analytics/firewall-activity")
async def get_firewall_activity(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    organization_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_orchestrator_db)
):
    """Legacy firewall activity endpoint - data now available through Langfuse analytics."""
    return {
        "message": "Legacy firewall monitoring has been removed. Firewall activity is now tracked in Langfuse traces.",
        "redirect": "Use /analytics/overview?use_langfuse=true to see firewall block counts",
        "time_range": {
            "start": start_date.isoformat() if start_date else datetime.now(timezone.utc).isoformat(),
            "end": end_date.isoformat() if end_date else datetime.now(timezone.utc).isoformat()
        },
        "firewall_activity": {
            "total_requests": 0,
            "blocked_requests": 0,
            "allowed_requests": 0,
            "block_rate": 0.0,
            "allow_rate": 0.0
        },
        "block_reasons": {
            "pii_violations": 0,
            "secrets_detected": 0,
            "toxicity_detected": 0
        }
    }