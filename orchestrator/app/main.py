"""Main FastAPI application for orchestrator service."""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Initialize logging FIRST - before any other imports
from .core.logging_config import setup_logging, get_logger

# Setup logging as early as possible
setup_logging()
logger = get_logger(__name__)

# Import database components
from .db.database import db_manager, init_db

# Import controller registration client
from .services.controller_client import ensure_controller_registration, cleanup_controller_registration

# Import API routers
from .api.v1.orchestrator import router as orchestrator_router
from .api.routes_websocket import router as websocket_router
from .api.routes_cache import router as cache_router
from .api.routes_llm import router as llm_router, agents_router
from .api.routes_firewall import router as firewall_router
from .api.routes_auth import router as auth_router

# Import monitoring API routers
from .monitoring.api.routers.system_metrics import router as monitoring_metrics_router
from .monitoring.api.routers.streaming import router as monitoring_streaming_router
# Monitoring WebSocket router disabled - analytics now handled by unified /ws/v1/session
# from .monitoring.api.routers.websocket import router as monitoring_websocket_router
from .monitoring.api.routers.analytics import router as analytics_router

# Import agents
from .agents import PromptResponseAgent

# Phoenix Arize AI observability client
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("Warning: Phoenix/OpenTelemetry not available - continuing without LLM observability")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    from .core.logging_config import log_exception
    
    # Startup
    logger.info("🚀 Starting MoolAI Orchestrator Service...")
    
    # Validate logging setup
    from .core.startup_validation import validate_logging_setup, log_startup_summary
    validation_results = validate_logging_setup()
    
    # Initialize database
    try:
        logger.info("Initializing database connection...")
        # Test database connection first
        connection_ok = await db_manager.test_connection()
        if not connection_ok:
            logger.warning("Database connection test failed")
        
        await init_db()
        logger.info("✅ Orchestrator database initialized successfully")
    except Exception as e:
        logger.error("❌ Orchestrator database initialization failed")
        log_exception(logger, e, {"component": "database_init"})
        raise  # Fail startup if database init fails
    
    # Register with controller
    environment = os.getenv("ENVIRONMENT", "production")
    try:
        if environment.lower() == "development":
            logger.info("🔧 DEVELOPMENT MODE: Attempting controller registration (optional)...")
        else:
            logger.info("🏭 PRODUCTION MODE: Registering with MoolAI Controller (required)...")
        
        await ensure_controller_registration()
        logger.info("✅ Controller registration process completed")
    except Exception as e:
        logger.error("❌ Controller registration failed")
        log_exception(logger, e, {"component": "controller_registration", "environment": environment})
        if environment.lower() != "development":
            logger.critical("Production mode: Orchestrator cannot start without controller registration")
            raise  # Only fail startup in production mode
        else:
            logger.warning("Development mode: Continuing without controller")  # Should not reach here with new logic
    
    # Initialize prompt-response agent
    try:
        organization_id = os.getenv("ORGANIZATION_ID", "default-org")
        app.state.prompt_agent = PromptResponseAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            organization_id=organization_id
        )
        print(f"Prompt-Response Agent initialized for organization: {organization_id}")
    except Exception as e:
        print(f"Prompt-Response Agent initialization failed: {e}")
        app.state.prompt_agent = None
    
    # Initialize development users (development mode)
    try:
        from .utils.dev_users import ensure_dev_users
        created_users = await ensure_dev_users()
        if created_users:
            print("Development users created successfully")
        else:
            print("Development users already exist or not in development mode")
    except Exception as e:
        print(f"Development user initialization warning: {e}")
        # Don't fail startup for dev user creation
    
    # Initialize session management system
    try:
        from .utils.session_config import session_config
        from .utils.buffer_manager import buffer_manager
        from .utils.session_dispatch import cleanup_expired_sessions
        import asyncio
        
        print("Initializing session management system...")
        
        # Initialize session configuration
        config = session_config.get_config()
        print(f"Session management enabled for organization: {config.get('orchestrator_id')}")
        
        # Set up periodic session cleanup
        async def periodic_session_cleanup():
            while True:
                try:
                    timeout = session_config.get_session_config().get('timeout_seconds', 1800)
                    interval = session_config.get_session_config().get('cleanup_interval', 300)
                    cleanup_expired_sessions(timeout)
                    await asyncio.sleep(interval)
                except Exception as e:
                    print(f"Session cleanup error: {e}")
                    await asyncio.sleep(60)  # Retry in 1 minute on error
        
        # Start background session cleanup
        cleanup_task = asyncio.create_task(periodic_session_cleanup())
        app.state.session_cleanup_task = cleanup_task
        app.state.session_config = session_config
        app.state.buffer_manager = buffer_manager
        
        print("Session management system initialized")
    except Exception as e:
        print(f"Session management initialization failed: {e}")
        app.state.session_cleanup_task = None
        app.state.session_config = None
        app.state.buffer_manager = None
    
    # Initialize embedded system monitoring
    try:
        from .monitoring.middleware.system_monitoring import SystemPerformanceMiddleware
        import redis.asyncio as redis
        
        print("Initializing embedded system monitoring...")
        
        # Setup Redis connection for system monitoring
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.from_url(redis_url)
        
        # Create system monitoring middleware
        system_monitoring_middleware = SystemPerformanceMiddleware(
            redis_client=redis_client,
            organization_id=organization_id,
            collection_interval=30,  # 30 seconds for testing
            enable_realtime_redis=True
        )
        
        # Start background system monitoring
        await system_monitoring_middleware.start_continuous_organization_monitoring()
        app.state.system_monitoring_middleware = system_monitoring_middleware
        
        print(f"System monitoring started for organization: {organization_id}")
    except Exception as e:
        print(f"System monitoring initialization failed: {e}")
        app.state.system_monitoring_middleware = None
    
    # Initialize Phoenix AI observability for LLM monitoring
    try:
        if PHOENIX_AVAILABLE:
            print("Initializing Phoenix AI observability...")
            
            # Get Phoenix configuration
            phoenix_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://phoenix:4317")
            project_name = os.getenv("PHOENIX_PROJECT_NAME", f"moolai-{organization_id}")
            
            # Create resource with service information
            resource = Resource.create({
                "service.name": "moolai-orchestrator",
                "service.version": "1.0.0",
                "service.instance.id": organization_id,
                "project.name": project_name
            })
            
            # Initialize tracing
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Custom span processor to properly classify LLM spans
            from .monitoring.phoenix_span_processor import PhoenixLLMSpanProcessor
            llm_span_processor = PhoenixLLMSpanProcessor()
            tracer_provider.add_span_processor(llm_span_processor)
            
            # Configure OTLP span exporter for Phoenix
            otlp_exporter = OTLPSpanExporter(
                endpoint=phoenix_endpoint,
                insecure=True  # Use insecure connection for internal Docker network
            )
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Initialize metrics (optional)
            metric_reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(
                    endpoint=phoenix_endpoint,
                    insecure=True
                ),
                export_interval_millis=30000
            )
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            metrics.set_meter_provider(meter_provider)
            
            # Auto-instrument OpenAI calls
            OpenAIInstrumentor().instrument()
            
            # Auto-instrument FastAPI
            FastAPIInstrumentor.instrument_app(app)
            
            # Store configuration in app state
            app.state.phoenix_tracer = trace.get_tracer("moolai-orchestrator")
            app.state.phoenix_meter = metrics.get_meter("moolai-orchestrator")
            app.state.phoenix_endpoint = phoenix_endpoint
            
            print(f"Phoenix observability initialized for organization: {organization_id}")
            print(f"Phoenix endpoint: {phoenix_endpoint}")
            print(f"Project name: {project_name}")
        else:
            print("Phoenix/OpenTelemetry packages not available - skipping initialization")
            app.state.phoenix_tracer = None
            app.state.phoenix_meter = None
    except Exception as e:
        logger.error("❌ Phoenix initialization failed")
        log_exception(logger, e, {"component": "phoenix_init"})
        app.state.phoenix_tracer = None
        app.state.phoenix_meter = None
    
    # Log startup completion summary
    log_startup_summary(validation_results)
    logger.info("🎉 MoolAI Orchestrator Service startup completed!")
    
    yield
    
    # Shutdown
    print("Shutting down MoolAI Orchestrator Service...")
    
    # Stop session management
    try:
        if hasattr(app.state, 'session_cleanup_task') and app.state.session_cleanup_task:
            print("Stopping session management...")
            app.state.session_cleanup_task.cancel()
            try:
                await app.state.session_cleanup_task
            except asyncio.CancelledError:
                pass
            print("Session management stopped")
    except Exception as e:
        print(f"Error stopping session management: {e}")
    
    # Stop system monitoring
    try:
        if hasattr(app.state, 'system_monitoring_middleware') and app.state.system_monitoring_middleware:
            print("Stopping system monitoring...")
            await app.state.system_monitoring_middleware.stop_continuous_organization_monitoring()
            print("System monitoring stopped")
    except Exception as e:
        print(f"Error stopping system monitoring: {e}")
    
    # Phoenix client cleanup
    try:
        if hasattr(app.state, 'phoenix_tracer') and app.state.phoenix_tracer:
            print("Stopping Phoenix observability...")
            # Flush any pending traces
            if trace.get_tracer_provider():
                trace.get_tracer_provider().force_flush(timeout_millis=5000)
            print("Phoenix observability stopped")
    except Exception as e:
        print(f"Error stopping Phoenix observability: {e}")
    
    # Deregister from controller
    try:
        print("Deregistering from controller...")
        await cleanup_controller_registration()
        print("Successfully deregistered from controller")
    except Exception as e:
        print(f"Error during controller deregistration: {e}")
    
    # Close database connections
    try:
        await db_manager.close()
        print("Orchestrator database connections closed")
    except Exception as e:
        print(f"Error closing orchestrator database connections: {e}")


# Create FastAPI app
app = FastAPI(
    title="MoolAI Orchestrator Service",
    description="AI Workflow Orchestration and LLM Management",
    version="1.0.0",
    lifespan=lifespan
)

# Add request logging middleware FIRST (outer layer)
from .middleware.request_logger import setup_request_logging
setup_request_logging(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(orchestrator_router, prefix="/api/v1")
app.include_router(auth_router)     # Authentication router (has its own /api/v1/auth prefix)
app.include_router(websocket_router)  # WebSocket router has its own /ws prefix
app.include_router(cache_router)    # Cache router already has prefix
app.include_router(llm_router)      # LLM router already has prefix
app.include_router(agents_router)   # Agents router already has prefix
app.include_router(firewall_router) # Firewall router already has prefix

# Include monitoring API routers (routers already have their own prefixes)
app.include_router(monitoring_metrics_router, tags=["monitoring"])
app.include_router(monitoring_streaming_router, tags=["streaming"])
# Monitoring WebSocket router disabled - analytics handled by unified /ws/v1/session
# app.include_router(monitoring_websocket_router, tags=["websocket"])
app.include_router(analytics_router, prefix="/api/v1", tags=["analytics"])

# Setup static file serving for frontend
frontend_dist_path = Path(__file__).parent / "gui" / "frontend" / "dist"
if frontend_dist_path.exists():
    # Mount static assets
    app.mount("/assets", StaticFiles(directory=str(frontend_dist_path / "assets")), name="assets")
    
    @app.get("/")
    def serve_frontend():
        """Serve the frontend SPA."""
        return FileResponse(str(frontend_dist_path / "index.html"))
    
    @app.get("/{path:path}")
    def catch_all(request: Request, path: str):
        """Catch all routes and serve the frontend SPA for client-side routing."""
        # Exclude API routes
        if path.startswith("api/") or path.startswith("ws/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        # Serve static files if they exist
        file_path = frontend_dist_path / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        
        # Fallback to index.html for SPA routes
        return FileResponse(str(frontend_dist_path / "index.html"))
else:
    print("Warning: Frontend dist directory not found, serving API only")
    
    @app.get("/")
    def root():
        return {"message": "MoolAI Orchestrator Service", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "service": "orchestrator",
        "version": "1.0.0"
    }
    
    # Check database
    try:
        db_connected = await db_manager.test_connection()
        if db_connected:
            health_status["database"] = "connected"
        else:
            health_status["database"] = "disconnected"
            health_status["status"] = "degraded"
    except Exception:
        health_status["database"] = "error"
        health_status["status"] = "degraded"
    
    return health_status
