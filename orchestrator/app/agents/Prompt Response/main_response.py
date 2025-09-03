from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# Phoenix/OpenTelemetry observability - OpenAI client is auto-instrumented
from openai import AsyncOpenAI
try:
    from opentelemetry import trace
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
import os
from dotenv import load_dotenv
import asyncio
from typing import Optional
import httpx
import logging
import time
import json
import hashlib
import numpy as np

# Import firewall services for security checks
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../services'))
from Firewall.server import _pii_local, _secrets_local, _toxicity_local

# Import evaluation services
evaluation_path = os.path.join(os.path.dirname(__file__), '../../services/Evaluation')
if evaluation_path not in sys.path:
    sys.path.append(evaluation_path)
    
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from answer_correctness import evaluate_answer_correctness
    from answer_relevance import evaluate_answer_relevance  
    from goal_accuracy import evaluate_goal_accuracy
    from hallucination import evaluate_hallucination
    from summarization import evaluate_summarization
    logger.info("Evaluation services imported successfully")
except ImportError as e:
    logger.warning(f"Could not import evaluation services: {e}")
    # Create placeholder functions to avoid runtime errors
    async def evaluate_answer_correctness(query: str, answer: str) -> dict:
        return {"score": 0.0, "explanation": "Evaluation service unavailable"}
    async def evaluate_answer_relevance(query: str, answer: str) -> dict:
        return {"score": 0.0, "explanation": "Evaluation service unavailable"}
    async def evaluate_goal_accuracy(query: str, answer: str) -> dict:
        return {"score": 0.0, "explanation": "Evaluation service unavailable"}
    async def evaluate_hallucination(query: str, answer: str) -> dict:
        return {"score": 0.0, "explanation": "Evaluation service unavailable"}
    async def evaluate_summarization(query: str, answer: str) -> dict:
        return {"score": 0.0, "explanation": "Evaluation service unavailable"}

# Import monitoring middleware - conditional to avoid import issues
monitoring_middleware = None
LLMMonitoringMiddleware = None
try:
    # Use absolute path to find the monitoring middleware
    import sys
    import os
    
    # Get the absolute path to the app directory
    current_file = os.path.abspath(__file__)
    agents_dir = os.path.dirname(current_file)  # /app/app/agents/Prompt Response/
    app_dir = os.path.dirname(os.path.dirname(agents_dir))  # /app/app/
    root_dir = os.path.dirname(app_dir)  # /app/
    
    # Add both app directory and root directory to Python path
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Try multiple import approaches
    try:
        from monitoring.middleware.monitoring import LLMMonitoringMiddleware
        logger.info("Successfully imported LLMMonitoringMiddleware via relative import")
    except ImportError:
        from app.monitoring.middleware.monitoring import LLMMonitoringMiddleware
        logger.info("Successfully imported LLMMonitoringMiddleware via absolute import")
        
except ImportError as e:
    logger.warning(f"Could not import monitoring middleware: {e}")
    LLMMonitoringMiddleware = None

# Load environment variables
load_dotenv()

# Organization configuration
organization_id = os.getenv("ORGANIZATION_ID", "org_001")

# Initialize OpenAI client with Phoenix OpenTelemetry instrumentation
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client - Phoenix OpenTelemetry instrumentation handles tracing automatically  
client = AsyncOpenAI(api_key=openai_api_key)

# Cache service configuration
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
LLM_CACHE_REDIS_URL = os.getenv("REDIS_LLM_CACHE_URL", "redis://localhost:6379/1")

# Firewall service configuration
ENABLE_FIREWALL = os.getenv("ENABLE_FIREWALL", "true").lower() == "true"

# Initialize dedicated LLM cache (completely separate from monitoring cache)
llm_cache_client = None
if ENABLE_CACHING:
    # Create dedicated Redis client for LLM cache only
    import redis
    from urllib.parse import urlparse
    import json
    import hashlib
    from sentence_transformers import SentenceTransformer
    
    parsed_url = urlparse(LLM_CACHE_REDIS_URL)
    redis_host = parsed_url.hostname or "localhost"
    redis_port = parsed_url.port or 6379
    redis_db = int(parsed_url.path.lstrip('/')) if parsed_url.path else 1
    
    try:
        llm_cache_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
            socket_timeout=2,
            socket_connect_timeout=2,
            health_check_interval=30,
        )
        
        # Test connection
        llm_cache_client.ping()
        logger.info(f"LLM cache connected to Redis DB {redis_db} at {redis_host}:{redis_port}")
        
        # Initialize sentence transformer for semantic similarity
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    except Exception as e:
        logger.warning(f"Failed to initialize LLM cache: {e}")
        llm_cache_client = None

# Set cache_service to None - we'll use llm_cache_client directly
cache_service = None

# Initialize monitoring middleware
monitoring_middleware = None
MonitoringSessionLocal = None

if LLMMonitoringMiddleware is not None:
    try:
        # Import dependencies for monitoring
        import redis.asyncio as redis_async
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker
        
        # Get monitoring database URL
        monitoring_db_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://orchestrator_user:orchestrator_pass@localhost:5432/orchestrator_org_001")
        organization_id = os.getenv("ORGANIZATION_ID", "org_001")
        redis_monitoring_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Create async engine for monitoring database
        monitoring_engine = create_async_engine(monitoring_db_url, echo=False)
        MonitoringSessionLocal = sessionmaker(
            monitoring_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        # Create Redis client for monitoring
        parsed_redis_url = urlparse(redis_monitoring_url)
        redis_monitoring_client = redis_async.Redis(
            host=parsed_redis_url.hostname or "localhost",
            port=parsed_redis_url.port or 6379,
            db=int(parsed_redis_url.path.lstrip('/')) if parsed_redis_url.path else 0,
            decode_responses=False
        )
        
        # Initialize monitoring middleware
        monitoring_middleware = LLMMonitoringMiddleware(
            redis_client=redis_monitoring_client,
            db_session=None,  # Will be set per request
            organization_id=organization_id
        )
        
        logger.info("Monitoring middleware initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize monitoring middleware: {e}")
        monitoring_middleware = None
        MonitoringSessionLocal = None

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    model: Optional[str] = "gpt-3.5-turbo"

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    from_cache: bool = False
    similarity: Optional[float] = None

class CacheConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    ttl: Optional[int] = None
    similarity_threshold: Optional[float] = None

# Initialize FastAPI app
app = FastAPI(
    title="LLM Response API",
    description="A minimal FastAPI app that returns LLM responses for user queries",
    version="1.0.0"
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System instruction
SYSTEM_INSTRUCTION = "You are a helpful assistant. Provide clear, concise, and accurate responses to user questions."

# LLM Cache integration - completely separate from monitoring cache
async def get_cached_response(query: str, session_id: str = "default") -> Optional[dict]:
    """Try to get response from dedicated LLM cache"""
    if not ENABLE_CACHING or not llm_cache_client:
        return None
        
    try:
        # Create cache key from query
        cache_key = f"llm_cache:{session_id}:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check for exact match first
        cached_data = llm_cache_client.get(cache_key)
        if cached_data:
            result = json.loads(cached_data.decode('utf-8'))
            logger.info(f"LLM cache hit (exact): {cache_key}")
            return {
                "response": result["response"],
                "from_cache": True,
                "similarity": 1.0,
                "session_id": session_id,
                "cache_key": cache_key
            }
        
        # Semantic similarity search (simplified)
        if 'sentence_model' in globals():
            query_embedding = sentence_model.encode(query)
            
            # Search for similar queries in cache (simplified approach)
            # For production, you'd want a more sophisticated vector search
            pattern = f"llm_cache:{session_id}:*"
            cache_keys = llm_cache_client.keys(pattern)
            
            for key in cache_keys[:10]:  # Limit to 10 most recent for performance
                try:
                    cached_data = llm_cache_client.get(key)
                    if cached_data:
                        cached_result = json.loads(cached_data.decode('utf-8'))
                        if 'original_query' in cached_result:
                            cached_embedding = sentence_model.encode(cached_result['original_query'])
                            similarity = float(np.dot(query_embedding, cached_embedding) / 
                                             (np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)))
                            
                            if similarity >= 0.75:  # 75% similarity threshold
                                logger.info(f"LLM cache hit (semantic): similarity={similarity:.3f}")
                                return {
                                    "response": cached_result["response"],
                                    "from_cache": True,
                                    "similarity": similarity,
                                    "session_id": session_id,
                                    "cache_key": key.decode('utf-8') if isinstance(key, bytes) else key
                                }
                except Exception as e:
                    logger.debug(f"Error checking cached item {key}: {e}")
                    continue
        
        return None
    except Exception as e:
        logger.error(f"LLM cache error: {e}")
        return None

async def store_cached_response(query: str, response: str, session_id: str = "default", ttl: int = 3600):
    """Store response in dedicated LLM cache"""
    if not ENABLE_CACHING or not llm_cache_client:
        return
        
    try:
        cache_key = f"llm_cache:{session_id}:{hashlib.md5(query.encode()).hexdigest()}"
        cache_data = {
            "response": response,
            "original_query": query,
            "timestamp": time.time(),
            "session_id": session_id
        }
        
        llm_cache_client.setex(cache_key, ttl, json.dumps(cache_data))
        logger.info(f"Stored in LLM cache: {cache_key}")
    except Exception as e:
        logger.error(f"Error storing in LLM cache: {e}")

async def firewall_scan(text: str, request_span=None) -> dict:
    """
    Enhanced firewall scanning with Phoenix tracing.
    Fan out PII, secrets, and toxicity scans in parallel with comprehensive observability.
    Returns a dict with each scan's JSON response and tracing metadata.
    """
    if not ENABLE_FIREWALL:
        result = {"pii": {"contains_pii": False}, "secrets": {"contains_secrets": False}, "toxicity": {"contains_toxicity": False}}
        
        # Set firewall disabled attributes
        if TRACING_AVAILABLE and request_span:
            request_span.set_attribute("moolai.firewall.enabled", False)
            request_span.set_attribute("moolai.firewall.blocked", False)
        
        return result
    
    if TRACING_AVAILABLE:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("moolai.firewall.scan") as firewall_span:
            # Set firewall scan attributes
            firewall_span.set_attribute("moolai.firewall.enabled", True)
            firewall_span.set_attribute("moolai.firewall.text_length", len(text))
            firewall_span.set_attribute("moolai.firewall.text_hash", hashlib.md5(text.encode()).hexdigest()[:8])
            
            try:
                # Create child spans for individual scans
                with tracer.start_as_current_span("moolai.firewall.pii_scan") as pii_span:
                    pii_task = asyncio.create_task(asyncio.to_thread(_pii_local, text))
                
                with tracer.start_as_current_span("moolai.firewall.secrets_scan") as secrets_span:
                    secrets_task = asyncio.create_task(asyncio.to_thread(_secrets_local, text))
                    
                with tracer.start_as_current_span("moolai.firewall.toxicity_scan") as toxicity_span:
                    toxicity_task = asyncio.create_task(asyncio.to_thread(_toxicity_local, text))
                
                # Wait for all scans to complete
                pii_result, secrets_result, toxicity_result = await asyncio.gather(
                    pii_task, secrets_task, toxicity_task
                )
                
                # Set individual scan results
                pii_blocked = pii_result.get("contains_pii", False)
                secrets_blocked = secrets_result.get("contains_secrets", False) 
                toxicity_blocked = toxicity_result.get("contains_toxicity", False)
                
                firewall_span.set_attribute("moolai.firewall.pii.blocked", pii_blocked)
                firewall_span.set_attribute("moolai.firewall.secrets.blocked", secrets_blocked)
                firewall_span.set_attribute("moolai.firewall.toxicity.blocked", toxicity_blocked)
                
                # Overall firewall result
                blocked = pii_blocked or secrets_blocked or toxicity_blocked
                firewall_span.set_attribute("moolai.firewall.blocked", blocked)
                
                # Set attributes on request span for top-level visibility
                if request_span:
                    request_span.set_attribute("moolai.firewall.enabled", True)
                    request_span.set_attribute("moolai.firewall.blocked", blocked)
                    request_span.set_attribute("moolai.firewall.pii.blocked", pii_blocked)
                    request_span.set_attribute("moolai.firewall.secrets.blocked", secrets_blocked) 
                    request_span.set_attribute("moolai.firewall.toxicity.blocked", toxicity_blocked)
                
                return {
                    "pii": pii_result,
                    "secrets": secrets_result,
                    "toxicity": toxicity_result,
                }
                
            except Exception as e:
                firewall_span.set_attribute("moolai.firewall.error", str(e))
                firewall_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                if request_span:
                    request_span.set_attribute("moolai.firewall.error", str(e))
                
                logger.error(f"Firewall service error: {e}")
                raise HTTPException(status_code=500, detail=f"Firewall error: {str(e)}")
    else:
        # Fallback without tracing
        try:
            tasks = [
                asyncio.create_task(asyncio.to_thread(_pii_local, text)),
                asyncio.create_task(asyncio.to_thread(_secrets_local, text)),
                asyncio.create_task(asyncio.to_thread(_toxicity_local, text)),
            ]
            
            pii_result, secrets_result, toxicity_result = await asyncio.gather(*tasks)
            
            return {
                "pii": pii_result,
                "secrets": secrets_result,
                "toxicity": toxicity_result,
            }
        except Exception as e:
            logger.error(f"Firewall service error: {e}")
            raise HTTPException(status_code=500, detail=f"Firewall error: {str(e)}")

async def generate_llm_response(query: str, session_id: str = "default", user_id: str = "default_user", model: str = "gpt-3.5-turbo") -> dict:
    """Generate LLM response with enhanced Phoenix OpenTelemetry observability and comprehensive tracing"""
    
    # Create root span with vendor-prefixed attributes for comprehensive request tracing
    tracer = None
    if TRACING_AVAILABLE:
        try:
            tracer = trace.get_tracer(__name__)
        except Exception:
            tracer = None
    
    # Start comprehensive request span with vendor-prefixed attributes
    if tracer:
        with tracer.start_as_current_span("moolai.request.process") as request_span:
            # Set request-level attributes using vendor prefix
            request_span.set_attribute("moolai.session_id", session_id)
            request_span.set_attribute("moolai.user_id", user_id)
            request_span.set_attribute("moolai.query.length", len(query))
            request_span.set_attribute("moolai.query.hash", hashlib.md5(query.encode()).hexdigest()[:8])
            
            return await _generate_llm_response_internal(query, session_id, user_id, model, request_span)
    else:
        return await _generate_llm_response_internal(query, session_id, user_id, model, None)

async def _generate_llm_response_internal(query: str, session_id: str, user_id: str, model: str, request_span) -> dict:
    """Internal LLM response generation with Phoenix tracing context"""
    
    # Firewall scanning with enhanced tracing - MUST be first to protect the system
    logger.info(f"Firewall check starting - ENABLE_FIREWALL={ENABLE_FIREWALL}")
    if ENABLE_FIREWALL:
        logger.info(f"Running firewall scan on query: {query[:50]}...")
        if TRACING_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("moolai.firewall.scan") as firewall_span:
                scan_result = await firewall_scan(query.strip(), request_span)
        else:
            scan_result = await firewall_scan(query.strip(), request_span)
        
        logger.info(f"Firewall scan results: PII={scan_result['pii']['contains_pii']}, Secrets={scan_result['secrets']['contains_secrets']}, Toxicity={scan_result['toxicity']['contains_toxicity']}")
        
        # Check if content should be blocked
        if scan_result["pii"]["contains_pii"] or scan_result["secrets"]["contains_secrets"] or scan_result["toxicity"]["contains_toxicity"]:
            # Log the blocked request
            logger.warning(f"FIREWALL BLOCKING REQUEST - PII: {scan_result['pii']['contains_pii']}, Secrets: {scan_result['secrets']['contains_secrets']}, Toxicity: {scan_result['toxicity']['contains_toxicity']}")
            
            # Return blocked response
            return {
                "answer": "Request blocked by security firewall due to sensitive content detection.",
                "session_id": session_id,
                "from_cache": False,
                "similarity": None,
                "firewall_blocked": True,
                "firewall_reasons": scan_result
            }
    else:
        logger.info("Firewall is disabled, skipping scan")
    
    # Start monitoring if available (legacy - will be removed in Phase 4)
    request_context = None
    if monitoring_middleware:
        try:
            # Create database session for monitoring
            async with MonitoringSessionLocal() as db_session:
                monitoring_middleware.db_session = db_session
                request_context = await monitoring_middleware.track_request(
                    user_id=user_id,
                    agent_type="prompt_response",
                    prompt=query,
                    session_id=session_id
                )
        except Exception as e:
            logger.warning(f"Failed to start monitoring: {e}")
    
    # Cache lookup with enhanced tracing
    cache_hit = False
    cache_similarity = None
    if ENABLE_CACHING:
        if TRACING_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span("moolai.cache.lookup") as cache_span:
                # Set cache lookup attributes
                cache_span.set_attribute("moolai.cache.enabled", True)
                cache_span.set_attribute("moolai.cache.session_id", session_id)
                
                cache_result = await get_cached_response(query, session_id)
                if cache_result and cache_result.get("from_cache"):
                    cache_hit = True
                    cache_similarity = cache_result.get("similarity")
                    logger.info(f"LLM Cache HIT for session {session_id} (similarity: {cache_result.get('similarity', 'exact')})")
                    
                    # Set vendor-prefixed cache attributes for Phoenix tracking
                    cache_span.set_attribute("moolai.cache.hit", True)
                    cache_span.set_attribute("moolai.cache.similarity", cache_similarity or 1.0)
                    cache_span.set_attribute("moolai.cache.key", cache_result.get("cache_key", "unknown"))
                    
                    # Also set on request span for top-level visibility
                    if request_span:
                        request_span.set_attribute("moolai.cache.hit", True)
                        request_span.set_attribute("moolai.cache.similarity", cache_similarity or 1.0)
                else:
                    # Cache miss
                    cache_span.set_attribute("moolai.cache.hit", False)
                    cache_span.set_attribute("moolai.cache.similarity", 0.0)
                    
                    if request_span:
                        request_span.set_attribute("moolai.cache.hit", False)
                        request_span.set_attribute("moolai.cache.similarity", 0.0)
        else:
            cache_result = await get_cached_response(query, session_id)
            if cache_result and cache_result.get("from_cache"):
                cache_hit = True
                cache_similarity = cache_result.get("similarity")
                logger.info(f"LLM Cache HIT for session {session_id} (similarity: {cache_result.get('similarity', 'exact')})")
            
            # Track response with monitoring
            if monitoring_middleware and request_context:
                try:
                    async with MonitoringSessionLocal() as db_session:
                        monitoring_middleware.db_session = db_session
                        await monitoring_middleware.track_response(
                            request_context=request_context,
                            response=cache_result["response"],
                            model=model,
                            cache_hit=True,
                            cache_similarity=cache_similarity
                        )
                except Exception as e:
                    logger.warning(f"Failed to track cached response: {e}")
            
            return {
                "answer": cache_result["response"],
                "session_id": session_id,
                "from_cache": True,
                "similarity": cache_result.get("similarity"),
                "tokens_used": 0,  # No new tokens used from cache
                "cost": 0.0        # No cost for cached response
            }
    
    # Generate fresh response from OpenAI with enhanced tracing
    logger.info(f"LLM Cache MISS - generating fresh response for session {session_id}")
    
    if TRACING_AVAILABLE:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("moolai.llm.call") as llm_span:
            # Set LLM call attributes with vendor prefix
            llm_span.set_attribute("moolai.llm.model", model)
            llm_span.set_attribute("moolai.llm.temperature", 0.2)
            llm_span.set_attribute("moolai.llm.max_tokens", 1000)
            llm_span.set_attribute("moolai.llm.cache_miss", True)
            
            if request_span:
                request_span.set_attribute("moolai.llm.model", model)
                request_span.set_attribute("moolai.llm.fresh_call", True)
            
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": query.strip()}
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            # Calculate cost using our cost calculator
            cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                try:
                    # Import cost calculator dynamically
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '../../monitoring/utils'))
                    from cost_calculator import calculate_cost
                    
                    cost = calculate_cost(
                        model=model,
                        input_tokens=response.usage.prompt_tokens or 0,
                        output_tokens=response.usage.completion_tokens or 0
                    )
                    logger.info(f"Calculated cost: ${cost:.6f} for {response.usage.total_tokens} tokens")
                except Exception as e:
                    logger.warning(f"Could not calculate cost: {e}")
                    # Fallback calculation
                    cost = ((response.usage.prompt_tokens or 0) * 0.0000005 + 
                           (response.usage.completion_tokens or 0) * 0.0000015)
            
            # Set response attributes including cost
            if hasattr(response, 'usage') and response.usage:
                llm_span.set_attribute("moolai.llm.input_tokens", response.usage.prompt_tokens or 0)
                llm_span.set_attribute("moolai.llm.output_tokens", response.usage.completion_tokens or 0)
                llm_span.set_attribute("moolai.llm.total_tokens", response.usage.total_tokens or 0)
                llm_span.set_attribute("moolai.llm.cost", cost)
                
                if request_span:
                    request_span.set_attribute("moolai.tokens.input", response.usage.prompt_tokens or 0)
                    request_span.set_attribute("moolai.tokens.output", response.usage.completion_tokens or 0)
                    request_span.set_attribute("moolai.tokens.total", response.usage.total_tokens or 0)
                    request_span.set_attribute("moolai.cost", cost)
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": query.strip()}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        # Calculate cost even without tracing
        cost = 0.0
        if hasattr(response, 'usage') and response.usage:
            try:
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '../../monitoring/utils'))
                from cost_calculator import calculate_cost
                
                cost = calculate_cost(
                    model=model,
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0
                )
            except Exception:
                # Fallback calculation
                cost = ((response.usage.prompt_tokens or 0) * 0.0000005 + 
                       (response.usage.completion_tokens or 0) * 0.0000015)
    
    answer = response.choices[0].message.content
    
    # Track response with monitoring
    if monitoring_middleware and request_context:
        try:
            async with MonitoringSessionLocal() as db_session:
                monitoring_middleware.db_session = db_session
                await monitoring_middleware.track_response(
                    request_context=request_context,
                    response=response,
                    model=model,
                    cache_hit=False,
                    cache_similarity=None
                )
        except Exception as e:
            logger.warning(f"Failed to track fresh response: {e}")
    
    # Store in dedicated LLM cache (completely separate from monitoring)
    if ENABLE_CACHING:
        await store_cached_response(query, answer, session_id)
        logger.info(f"Stored fresh response in dedicated LLM cache for session {session_id}")
    
    # Include cost and token information in response
    result = {
        "answer": answer,
        "session_id": session_id,
        "from_cache": False,
        "similarity": None
    }
    
    # Add usage and cost information if available
    if hasattr(response, 'usage') and response.usage:
        result["tokens_used"] = response.usage.total_tokens or 0
        result["prompt_tokens"] = response.usage.prompt_tokens or 0
        result["completion_tokens"] = response.usage.completion_tokens or 0
        result["cost"] = cost if 'cost' in locals() else 0.0
    
    return result

@app.get("/respond")
# Phoenix/OpenTelemetry tracing handled automatically
async def get_response(
    query: str = Query(..., description="User query to get LLM response for"),
    session_id: str = Query("default", description="Session ID for caching"),
    user_id: str = Query("default_user", description="User ID for monitoring"),
    model: str = Query("gpt-3.5-turbo", description="LLM model to use")
):
    """
    Get LLM response for a user query with caching support and monitoring.
    
    Args:
        query: The user's question or prompt
        session_id: Session ID for cache isolation
        user_id: User ID for monitoring
        
    Returns:
        JSON response with the LLM's answer and cache metadata
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query parameter cannot be empty")
    
    # Enhanced firewall check with tracing
    firewall_blocked = False
    firewall_reasons = None
    if ENABLE_FIREWALL:
        # Pass request span context to firewall scan for comprehensive tracing
        current_span = None
        if TRACING_AVAILABLE:
            try:
                current_span = trace.get_current_span()
            except Exception:
                pass
        
        scan = await firewall_scan(query.strip(), current_span)
        if scan["pii"]["contains_pii"] or scan["secrets"]["contains_secrets"] or scan["toxicity"]["contains_toxicity"]:
            firewall_blocked = True
            firewall_reasons = scan
            
            # Track blocked request with monitoring
            if monitoring_middleware:
                try:
                    async with MonitoringSessionLocal() as db_session:
                        monitoring_middleware.db_session = db_session
                        request_context = await monitoring_middleware.track_request(
                            user_id=user_id,
                            agent_type="prompt_response",
                            prompt=query,
                            session_id=session_id
                        )
                        await monitoring_middleware.track_response(
                            request_context=request_context,
                            response="Request blocked by firewall",
                            model=model,
                            error=None,
                            cache_hit=False,
                            cache_similarity=None,
                            firewall_blocked=True,
                            firewall_reasons=scan
                        )
                except Exception as e:
                    logger.warning(f"Failed to track blocked request: {e}")
            
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Content blocked by firewall",
                    "scan_results": scan
                }
            )
    
    try:
        result = await asyncio.wait_for(
            generate_llm_response(query.strip(), session_id, user_id, model),
            timeout=35.0  # Slightly longer timeout to account for cache calls
        )
        return result
        
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "Request timeout - the service took too long to respond"}
        )
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.post("/respond", response_model=QueryResponse)
# Phoenix/OpenTelemetry tracing handled automatically
async def post_response(request: QueryRequest):
    """
    Get LLM response for a user query (POST version with JSON body).
    
    Args:
        request: JSON body containing the query and optional session_id
        
    Returns:
        JSON response with the LLM's answer and cache metadata
    """
    query = request.query
    session_id = request.session_id or "default"
    model = request.model or "gpt-3.5-turbo"
    
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Enhanced firewall check with tracing
    if ENABLE_FIREWALL:
        # Pass request span context to firewall scan for comprehensive tracing
        current_span = None
        if TRACING_AVAILABLE:
            try:
                current_span = trace.get_current_span()
            except Exception:
                pass
        
        scan = await firewall_scan(query.strip(), current_span)
        if scan["pii"]["contains_pii"] or scan["secrets"]["contains_secrets"] or scan["toxicity"]["contains_toxicity"]:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Content blocked by firewall",
                    "scan_results": scan
                }
            )
    
    try:
        result = await asyncio.wait_for(
            generate_llm_response(query.strip(), session_id, model=model),
            timeout=35.0  # Slightly longer timeout to account for cache calls
        )
        
        return QueryResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            from_cache=result["from_cache"],
            similarity=result["similarity"]
        )
        
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=408,
            content={"error": "Request timeout - the service took too long to respond"}
        )
    except Exception as e:
        logger.error(f"Error in post_response: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint with cache service status"""
    cache_status = "unknown"
    if ENABLE_CACHING and cache_service:
        try:
            cache_status = "connected" if cache_service.ping() else "disconnected"
        except:
            cache_status = "disconnected"
    else:
        cache_status = "disabled"
    
    return {
        "status": "healthy",
        "cache_service": cache_status,
        "caching_enabled": ENABLE_CACHING
    }

# Cache management endpoints
@app.get("/cache/health")
async def cache_health():
    """Check cache service health"""
    if not ENABLE_CACHING or not cache_service:
        return {"cache_available": False, "message": "Caching is disabled"}
    
    try:
        cache_available = cache_service.ping()
        if cache_available:
            return {
                "cache_available": True,
                "cache_service": "local",
                "cache_details": cache_service.stats()
            }
        else:
            return {
                "cache_available": False,
                "cache_service": "local",
                "error": "Cache service ping failed"
            }
    except Exception as e:
        return {
            "cache_available": False,
            "cache_service": "local",
            "error": str(e)
        }

@app.get("/cache/config")
async def get_cache_config():
    """Get cache service configuration"""
    if not ENABLE_CACHING or not cache_service:
        return {"error": "Caching is disabled"}
    
    try:
        # Return basic cache configuration
        return {
            "enabled": ENABLE_CACHING,
            "cache_type": "Redis",
            "status": "local_service"
        }
    except Exception as e:
        return {"error": f"Cache service unavailable: {str(e)}"}

@app.post("/cache/config")
async def update_cache_config(config: CacheConfigUpdate):
    """Update cache service configuration"""
    if not ENABLE_CACHING or not cache_service:
        return {"error": "Caching is disabled"}
    
    try:
        # Update cache configuration if supported
        updated_config = {"message": "Cache config update not supported in local mode"}
        if config.enabled is not None:
            updated_config["enabled"] = config.enabled
        if config.ttl is not None:
            updated_config["ttl"] = config.ttl
        if config.similarity_threshold is not None:
            updated_config["similarity_threshold"] = config.similarity_threshold
        return updated_config
    except Exception as e:
        return {"error": f"Cache service unavailable: {str(e)}"}

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache service statistics"""
    if not ENABLE_CACHING or not cache_service:
        return {"error": "Caching is disabled"}
    
    try:
        return cache_service.stats()
    except Exception as e:
        return {"error": f"Cache service unavailable: {str(e)}"}

@app.delete("/cache/keys")
async def clear_cache():
    """Clear all cache entries"""
    if not ENABLE_CACHING or not cache_service:
        return {"error": "Caching is disabled"}
    
    try:
        cache_service.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        return {"error": f"Cache service unavailable: {str(e)}"}

# Evaluation endpoints
@app.post("/evaluate/correctness")
async def evaluate_response_correctness(request: QueryRequest):
    """Evaluate answer correctness for a query-response pair"""
    try:
        # First get the response
        response_data = await generate_llm_response(request.query, request.session_id, model=request.model)
        answer = response_data["answer"]
        
        # Then evaluate it
        evaluation = await evaluate_answer_correctness(request.query, answer)
        
        return {
            "query": request.query,
            "answer": answer,
            "evaluation": evaluation,
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Error in correctness evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/relevance")
async def evaluate_response_relevance(request: QueryRequest):
    """Evaluate answer relevance for a query-response pair"""
    try:
        # First get the response
        response_data = await generate_llm_response(request.query, request.session_id, model=request.model)
        answer = response_data["answer"]
        
        # Then evaluate it
        evaluation = await evaluate_answer_relevance(request.query, answer)
        
        return {
            "query": request.query,
            "answer": answer,
            "evaluation": evaluation,
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Error in relevance evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/comprehensive")
async def evaluate_response_comprehensive(request: QueryRequest):
    """Run comprehensive evaluation on a query-response pair"""
    try:
        # First get the response
        response_data = await generate_llm_response(request.query, request.session_id, model=request.model)
        answer = response_data["answer"]
        
        # Run all evaluations in parallel
        evaluations = await asyncio.gather(
            evaluate_answer_correctness(request.query, answer),
            evaluate_answer_relevance(request.query, answer),
            evaluate_goal_accuracy(request.query, answer),
            evaluate_hallucination(request.query, answer),
            evaluate_summarization(request.query, answer)
        )
        
        return {
            "query": request.query,
            "answer": answer,
            "evaluations": {
                "correctness": evaluations[0],
                "relevance": evaluations[1],
                "goal_accuracy": evaluations[2],
                "hallucination": evaluations[3],
                "summarization": evaluations[4]
            },
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
