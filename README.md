# MoolAI Orchestrator Package

The MoolAI Orchestrator is a worker service that handles AI workflows, LLM calls, and user interactions. It registers with a central Controller service for management and coordination.

## Package Contents

```
mooli_orchestrator_package/
├── orchestrator/           # Orchestrator service code
│   ├── app/               # FastAPI application
│   │   ├── api/           # REST API and WebSocket endpoints
│   │   ├── agents/        # AI agents
│   │   ├── models/        # Database models
│   │   ├── services/      # Business logic (LLM, cache, auth, etc.)
│   │   ├── monitoring/    # Embedded monitoring system
│   │   ├── utils/         # Utilities and helpers
│   │   └── main.py        # Application entry point
│   └── requirements.txt   # Python dependencies
├── common/                # Shared modules
├── Dockerfile            # Docker image configuration
├── docker-compose.yml    # Multi-container setup
├── build.sh             # Image build script
├── .env.example         # Environment configuration template
└── README.md            # This file
```

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Access to a MoolAI Controller instance
- OpenAI API key (required)
- Anthropic API key (optional)

### 1. Build the Docker Image

```bash
./build.sh
```

This creates a `moolai/orchestrator:latest` Docker image.

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

**Required Environment Variables:**
- `CONTROLLER_URL`: Address of your controller service
- `ORCHESTRATOR_DB_PASSWORD`: Orchestrator database password
- `MONITORING_DB_PASSWORD`: Monitoring database password
- `JWT_SECRET`: JWT secret (must match controller)
- `OPENAI_API_KEY`: Your OpenAI API key

### 3. Deploy with Docker Compose

```bash
# Start all services (without Phoenix)
docker-compose up -d

# Or start with Phoenix AI observability
COMPOSE_PROFILES=with-phoenix docker-compose up -d

# View logs
docker-compose logs -f moolai-orchestrator

# Check health
curl http://localhost:8000/health
```

## Services Included

The orchestrator package includes:

1. **Orchestrator Service** (Port 8000)
   - FastAPI application
   - WebSocket chat interface
   - LLM API endpoints
   - Authentication and user management

2. **PostgreSQL Databases**
   - Orchestrator DB (Port 5434): Main application data
   - Monitoring DB (Port 5432): Performance metrics

3. **Redis Cache** (Port 6379)
   - LLM response caching
   - Session management

4. **Phoenix AI Observability** (Optional, Ports 6006, 4317)
   - LLM call tracing and monitoring
   - Performance analytics

## API Endpoints

Once deployed, the orchestrator provides:

### Core APIs
- **Health Check**: `GET /health`
- **WebSocket Chat**: `WS /ws/chat`
- **LLM API**: `POST /api/v1/llm/prompt`

### Monitoring APIs
- **System Metrics**: `GET /api/v1/system/metrics`
- **Cache Stats**: `GET /api/v1/cache/stats`
- **Analytics**: `GET /api/v1/analytics/overview`

### Authentication APIs
- **Login**: `POST /api/v1/auth/login`
- **User Info**: `GET /api/v1/auth/me`
- **Development Users**: `GET /api/v1/auth/users/dev`

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORG_ID` | `org_001` | Organization identifier |
| `ORCHESTRATOR_ID` | `orchestrator_001` | Unique orchestrator ID |
| `ORCHESTRATOR_PORT` | `8000` | Service port |
| `CONTROLLER_URL` | **Required** | Controller service URL |
| `ORCHESTRATOR_DB_PASSWORD` | **Required** | Orchestrator DB password |
| `MONITORING_DB_PASSWORD` | **Required** | Monitoring DB password |
| `JWT_SECRET` | **Required** | Must match controller |
| `OPENAI_API_KEY` | **Required** | OpenAI API key |
| `ANTHROPIC_API_KEY` | Optional | Anthropic API key |
| `ENVIRONMENT` | `production` | Environment mode |
| `MULTI_USER_MODE` | `true` | Enable multi-user support |
| `LOG_LEVEL` | `INFO` | Logging level |

### Phoenix Configuration
When using `COMPOSE_PROFILES=with-phoenix`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PHOENIX_UI_PORT` | `6006` | Phoenix web UI port |
| `PHOENIX_GRPC_PORT` | `4317` | Phoenix gRPC collector port |

## Multi-User Authentication

The orchestrator supports multi-user authentication with development and production modes:

### Development Mode (Default)
- Password-free login for development users
- Auto-creates users: `dinakar_developer`, `amogh_analyst`, `gabriella_admin`
- JWT tokens still generated for API consistency

### Production Mode
Set `ENVIRONMENT=production` for:
- Full JWT token validation
- Integration with Azure Entra ID (future)
- Stricter authentication requirements

## Phoenix AI Observability

Phoenix provides comprehensive LLM observability:

### Features
- **LLM Call Tracing**: Track all OpenAI/Anthropic API calls
- **Performance Metrics**: Response times, token usage, costs
- **Real-time Analytics**: Live dashboard with insights
- **Error Tracking**: Failed requests and debugging info

### Access
- **Web UI**: `http://localhost:6006`
- **gRPC Endpoint**: `http://localhost:4317`
- **Data Storage**: PostgreSQL in `phoenix` schema

## Monitoring

### Health Checks
```bash
# Orchestrator health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres-orchestrator pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

### Logs
```bash
# Orchestrator logs
docker-compose logs -f moolai-orchestrator

# All services
docker-compose logs -f

# Specific service
docker-compose logs -f postgres-orchestrator
```

### System Metrics
```bash
# System metrics API
curl http://localhost:8000/api/v1/system/metrics

# Cache statistics
curl http://localhost:8000/api/v1/cache/stats

# Analytics overview
curl http://localhost:8000/api/v1/analytics/overview
```

## Controller Registration

The orchestrator automatically registers with the controller on startup. Manual registration:

```bash
curl -X POST ${CONTROLLER_URL}/api/v1/internal/orchestrators/register \
  -H "Content-Type: application/json" \
  -d '{
    "orchestrator_id": "'${ORCHESTRATOR_ID}'",
    "organization_id": "'${ORG_ID}'",
    "name": "Orchestrator '${ORG_ID}'",
    "url": "http://orchestrator-ip:8000"
  }'
```

## Troubleshooting

### Container Won't Start
1. **Check logs**: `docker-compose logs moolai-orchestrator`
2. **Verify environment**: Check all required variables in `.env`
3. **Database issues**: `docker-compose logs postgres-orchestrator`

### Controller Registration Fails
1. **Network connectivity**: `ping controller-ip`
2. **Controller accessibility**: `curl http://controller-ip:9000/health`
3. **Environment variables**: Verify `CONTROLLER_URL` is correct

### Authentication Issues
1. **JWT secret mismatch**: Ensure `JWT_SECRET` matches controller
2. **Development users**: Check `curl http://localhost:8000/api/v1/auth/users/dev`
3. **Login test**: `curl -X POST http://localhost:8000/api/v1/auth/login -d '{"username":"dinakar_developer"}'`

### LLM API Issues
1. **API key validity**: Test OpenAI API key separately
2. **Network access**: Ensure orchestrator can reach external APIs
3. **Cache issues**: Check Redis: `docker-compose exec redis redis-cli monitor`

### Phoenix Not Working
1. **Profile enabled**: Use `COMPOSE_PROFILES=with-phoenix`
2. **Database connection**: Check Phoenix logs
3. **Port conflicts**: Verify ports 6006 and 4317 are available

## Development

### Building from Source
Modify code in `orchestrator/app/` and rebuild:

```bash
./build.sh
docker-compose up -d
```

### Adding New Features
1. **Backend**: Modify `orchestrator/app/`
2. **Frontend**: Update `orchestrator/app/gui/frontend/`
3. **Dependencies**: Update `requirements.txt`
4. **Rebuild**: `./build.sh`

### Testing
```bash
# Test WebSocket connection
# Open http://localhost:8000/test-websocket

# Test LLM API
curl -X POST http://localhost:8000/api/v1/llm/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'

# Test authentication
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "dinakar_developer"}'
```

## Production Deployment

### Recommended Setup
1. **Dedicated VM per Organization**: Each org gets its own orchestrator VM
2. **Persistent Storage**: Volume mounts for databases and Redis
3. **Backup Strategy**: Regular database backups
4. **Network Security**: Firewall rules, VPN access
5. **Monitoring**: Log aggregation, alerting
6. **SSL/TLS**: HTTPS termination with reverse proxy

### Scaling Considerations
- **Horizontal**: Deploy multiple orchestrators per organization
- **Vertical**: Increase VM resources based on load
- **Database**: Monitor PostgreSQL performance
- **Cache**: Monitor Redis memory usage

### Backup and Recovery
```bash
# Backup orchestrator database
docker-compose exec postgres-orchestrator pg_dump -U orchestrator_user orchestrator_org_001 > orch_backup.sql

# Backup monitoring database
docker-compose exec postgres-monitoring pg_dump -U monitoring_user monitoring_org_001 > mon_backup.sql

# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
```

## Network Architecture

```
┌─────────────────────────────────────────┐
│            Orchestrator VM              │
│                                         │
│ ┌─────────────┐  ┌──────────────────┐   │
│ │Orchestrator │  │  PostgreSQL      │   │
│ │Port: 8000   │──┤  Orch: 5434      │   │
│ │             │  │  Mon:  5432      │   │
│ └─────────────┘  └──────────────────┘   │
│        │                                │
│ ┌──────▼──────┐  ┌──────────────────┐   │
│ │   Redis     │  │     Phoenix      │   │
│ │ Port: 6379  │  │  UI:   6006      │   │
│ │             │  │  gRPC: 4317      │   │
│ └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────┘
                    │
                    │ Registration
                    ▼
┌─────────────────────────────────────────┐
│           Controller VM                 │
│ ┌─────────────┐  ┌──────────────────┐   │
│ │ Controller  │  │   PostgreSQL     │   │
│ │ Port: 9000  │──┤   Port: 5436     │   │
│ └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────┘
```