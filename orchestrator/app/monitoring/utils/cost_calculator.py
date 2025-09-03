"""Cost calculation utilities using Phoenix pricing data."""

from decimal import Decimal
from typing import Union, Optional, Dict
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import logging

logger = logging.getLogger(__name__)

# Cache for model costs to avoid repeated DB queries
_model_cost_cache: Dict[str, Dict[str, float]] = {}


async def get_model_costs_from_phoenix(
    model_name: str,
    db_session: AsyncSession
) -> Dict[str, float]:
    """
    Get model pricing from Phoenix token_prices table.
    
    Args:
        model_name: The model name (e.g., "gpt-3.5-turbo")
        db_session: Database session for Phoenix data
    
    Returns:
        Dictionary with 'input' and 'output' costs per token
    """
    # Check cache first
    cache_key = model_name.lower()
    if cache_key in _model_cost_cache:
        return _model_cost_cache[cache_key]
    
    try:
        # Query Phoenix for model pricing
        query = text("""
            SELECT 
                tp.token_type,
                tp.base_rate
            FROM phoenix.token_prices tp
            JOIN phoenix.generative_models gm ON gm.id = tp.model_id
            WHERE (
                LOWER(gm.name) = LOWER(:model_name)
                OR LOWER(:model_name) LIKE '%' || LOWER(gm.name) || '%'
                OR (gm.name_pattern IS NOT NULL AND LOWER(:model_name) ~ gm.name_pattern)
            )
            AND tp.token_type IN ('input', 'output')
            ORDER BY gm.id
            LIMIT 2;
        """)
        
        result = await db_session.execute(query, {"model_name": model_name})
        rows = result.fetchall()
        
        costs = {}
        for row in rows:
            token_type = row.token_type
            base_rate = float(row.base_rate)
            
            if token_type == 'input':
                # Convert from per-token to per-1K tokens
                costs['input'] = base_rate * 1000
            elif token_type == 'output':
                costs['output'] = base_rate * 1000
        
        # If we found pricing, cache it and return
        if 'input' in costs and 'output' in costs:
            _model_cost_cache[cache_key] = costs
            logger.info(f"Found Phoenix pricing for {model_name}: {costs}")
            return costs
        
        # Fallback: Try to find default GPT-3.5-turbo pricing
        if model_name != "gpt-3.5-turbo":
            logger.warning(f"No Phoenix pricing for {model_name}, using GPT-3.5-turbo fallback")
            return await get_model_costs_from_phoenix("gpt-3.5-turbo", db_session)
        
    except Exception as e:
        logger.error(f"Error fetching Phoenix pricing for {model_name}: {e}")
    
    # Ultimate fallback - hardcoded GPT-3.5-turbo current pricing
    fallback = {
        "input": 0.0005,   # $0.50 per 1M tokens
        "output": 0.0015   # $1.50 per 1M tokens
    }
    logger.warning(f"Using hardcoded fallback pricing for {model_name}")
    return fallback


async def calculate_cost_async(
    model: str,
    input_tokens: Union[int, float],
    output_tokens: Union[int, float],
    db_session: Optional[AsyncSession] = None
) -> float:
    """
    Calculate the cost of an LLM API call using Phoenix pricing data.
    
    Args:
        model: The model name (e.g., "gpt-3.5-turbo")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        db_session: Database session for fetching Phoenix pricing
    
    Returns:
        Total cost in USD
    """
    if db_session:
        costs = await get_model_costs_from_phoenix(model, db_session)
    else:
        # Use cached or fallback pricing
        costs = _model_cost_cache.get(model.lower(), {
            "input": 0.0005,
            "output": 0.0015
        })
    
    # Calculate costs (prices are per 1K tokens)
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    
    total_cost = input_cost + output_cost
    
    # Round to 6 decimal places for precision
    return round(total_cost, 6)


def calculate_cost(
    model: str,
    input_tokens: Union[int, float],
    output_tokens: Union[int, float]
) -> float:
    """
    Synchronous wrapper for cost calculation.
    Uses cached pricing or hardcoded fallback.
    
    Args:
        model: The model name (e.g., "gpt-3.5-turbo")
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
    
    Returns:
        Total cost in USD
    """
    # Use cached pricing or fallback
    costs = _model_cost_cache.get(model.lower(), {
        "input": 0.0005,   # GPT-3.5-turbo fallback
        "output": 0.0015
    })
    
    # Calculate costs (prices are per 1K tokens)
    input_cost = (input_tokens / 1000) * costs["input"]
    output_cost = (output_tokens / 1000) * costs["output"]
    
    total_cost = input_cost + output_cost
    
    # Round to 6 decimal places for precision
    return round(total_cost, 6)


async def store_cost_in_phoenix(
    span_id: int,
    trace_id: int,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_cost: float,
    db_session: AsyncSession
) -> bool:
    """
    Store calculated cost in Phoenix span_costs table.
    
    Args:
        span_id: Phoenix span ID
        trace_id: Phoenix trace ID
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_cost: Total calculated cost
        db_session: Database session
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get model ID from Phoenix
        model_query = text("""
            SELECT id FROM phoenix.generative_models
            WHERE LOWER(name) = LOWER(:model_name)
            OR LOWER(:model_name) LIKE '%' || LOWER(name) || '%'
            LIMIT 1;
        """)
        
        result = await db_session.execute(model_query, {"model_name": model})
        model_row = result.fetchone()
        model_id = model_row.id if model_row else 37  # Default to GPT-3.5-turbo
        
        # Insert or update cost record
        insert_query = text("""
            INSERT INTO phoenix.span_costs (
                span_rowid,
                trace_rowid,
                model_id,
                span_start_time,
                total_tokens,
                prompt_tokens,
                completion_tokens,
                total_cost,
                prompt_cost,
                completion_cost
            )
            VALUES (
                :span_id,
                :trace_id,
                :model_id,
                NOW(),
                :total_tokens,
                :prompt_tokens,
                :completion_tokens,
                :total_cost,
                :prompt_cost,
                :completion_cost
            )
            ON CONFLICT (span_rowid) DO UPDATE SET
                total_cost = EXCLUDED.total_cost,
                prompt_cost = EXCLUDED.prompt_cost,
                completion_cost = EXCLUDED.completion_cost,
                total_tokens = EXCLUDED.total_tokens,
                prompt_tokens = EXCLUDED.prompt_tokens,
                completion_tokens = EXCLUDED.completion_tokens;
        """)
        
        # Calculate individual costs
        costs = await get_model_costs_from_phoenix(model, db_session)
        prompt_cost = (input_tokens / 1000) * costs["input"]
        completion_cost = (output_tokens / 1000) * costs["output"]
        
        await db_session.execute(insert_query, {
            "span_id": span_id,
            "trace_id": trace_id,
            "model_id": model_id,
            "total_tokens": float(input_tokens + output_tokens),
            "prompt_tokens": float(input_tokens),
            "completion_tokens": float(output_tokens),
            "total_cost": total_cost,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost
        })
        
        await db_session.commit()
        logger.info(f"Stored cost ${total_cost:.6f} for span {span_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store cost in Phoenix: {e}")
        await db_session.rollback()
        return False


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Rule of thumb: 1 token ≈ 4 characters or 0.75 words
    
    Args:
        text: The text to estimate tokens for
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Use character count method (more accurate for code/technical content)
    char_estimate = len(text) / 4
    
    # Use word count method
    word_estimate = len(text.split()) * 1.33
    
    # Return average of both methods
    return int((char_estimate + word_estimate) / 2)


async def refresh_cost_cache(db_session: AsyncSession) -> Dict[str, Dict[str, float]]:
    """
    Refresh the model cost cache from Phoenix database.
    
    Args:
        db_session: Database session
    
    Returns:
        Updated cache dictionary
    """
    global _model_cost_cache
    
    try:
        query = text("""
            SELECT 
                gm.name as model_name,
                tp.token_type,
                tp.base_rate
            FROM phoenix.token_prices tp
            JOIN phoenix.generative_models gm ON gm.id = tp.model_id
            WHERE tp.token_type IN ('input', 'output')
            ORDER BY gm.name, tp.token_type;
        """)
        
        result = await db_session.execute(query)
        rows = result.fetchall()
        
        new_cache = {}
        for row in rows:
            model_name = row.model_name.lower()
            token_type = row.token_type
            base_rate = float(row.base_rate)
            
            if model_name not in new_cache:
                new_cache[model_name] = {}
            
            # Convert from per-token to per-1K tokens
            new_cache[model_name][token_type] = base_rate * 1000
        
        _model_cost_cache = new_cache
        logger.info(f"Refreshed cost cache with {len(new_cache)} models")
        return new_cache
        
    except Exception as e:
        logger.error(f"Failed to refresh cost cache: {e}")
        return _model_cost_cache


def calculate_monthly_projection(
    daily_cost: float,
    growth_rate: float = 1.0
) -> float:
    """
    Project monthly costs based on daily usage.
    
    Args:
        daily_cost: Current daily cost
        growth_rate: Expected growth multiplier (1.0 = no growth)
    
    Returns:
        Projected monthly cost
    """
    return daily_cost * 30 * growth_rate