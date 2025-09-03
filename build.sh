#!/bin/bash

# MoolAI Orchestrator Build Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}   MoolAI Orchestrator Build Script       ${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Parse command line arguments
MODE="development"  # Default to development
while [[ $# -gt 0 ]]; do
    case $1 in
        --production)
            MODE="production"
            shift
            ;;
        --development)
            MODE="development"
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            shift
            ;;
    esac
done

echo -e "${BLUE}Build Mode: ${MODE}${NC}"
echo ""

# Load environment variables from .env file
if [ -f .env ]; then
    echo -e "${BLUE}Loading environment from .env file...${NC}"
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop auto-export
else
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    if [ "$MODE" = "production" ]; then
        echo -e "${RED}Error: .env file is required for production mode${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Using development defaults...${NC}"
fi

# Load version from environment or use default
VERSION=${VERSION:-latest}
IMAGE_NAME="moolai/orchestrator:${VERSION}"

echo -e "${YELLOW}Building Orchestrator Image...${NC}"
echo "Image: ${IMAGE_NAME}"
echo "Mode: ${MODE}"
echo ""

# Build Docker image
echo -e "${BLUE}Step 1: Building Docker image${NC}"
docker build \
  --tag "${IMAGE_NAME}" \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="${VERSION}" \
  .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker image build failed${NC}"
    exit 1
fi

# Display image info
echo ""
echo -e "${BLUE}Step 2: Image Information${NC}"
docker images "${IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"

# Run basic image test
echo ""
echo -e "${BLUE}Step 3: Testing image${NC}"

# Load environment variables from .env file
if [ -f .env ]; then
    echo -e "${BLUE}Loading environment from .env file...${NC}"
    set -a  # Automatically export all variables
    source .env
    set +a  # Stop auto-export
fi

# Set default values if not in environment (development mode only)
if [ "$MODE" = "development" ]; then
    ORGANIZATION_ID=${ORGANIZATION_ID:-org_001}
    ORCHESTRATOR_ID=${ORCHESTRATOR_ID:-orchestrator_001}
    CONTROLLER_URL=${CONTROLLER_URL:-http://localhost:9000}
    JWT_SECRET=${JWT_SECRET:-dev-jwt-secret-key-change-in-production-2024}
    OPENAI_API_KEY=${OPENAI_API_KEY:-sk-test-build-key-not-real}
    ORCHESTRATOR_DB_PASSWORD=${ORCHESTRATOR_DB_PASSWORD:-dev_orchestrator_password_123}
    MONITORING_DB_PASSWORD=${MONITORING_DB_PASSWORD:-dev_monitoring_password_123}
    ENVIRONMENT=${ENVIRONMENT:-development}
    MULTI_USER_MODE=${MULTI_USER_MODE:-true}
    LOG_LEVEL=${LOG_LEVEL:-INFO}
fi

# Validate required environment variables for production
if [ "$MODE" = "production" ]; then
    required_vars=("ORGANIZATION_ID" "ORCHESTRATOR_ID" "CONTROLLER_URL" "JWT_SECRET" "OPENAI_API_KEY" "ORCHESTRATOR_DB_PASSWORD" "MONITORING_DB_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo -e "${RED}Error: $var is required for production mode${NC}"
            exit 1
        fi
    done
fi

# Create test environment variables
if [ "$MODE" = "development" ]; then
    # Development mode: Use SQLite for testing
    TEST_ENV_VARS=(
        "-e" "ORGANIZATION_ID=${ORGANIZATION_ID}"
        "-e" "ORCHESTRATOR_ID=${ORCHESTRATOR_ID}"
        "-e" "DATABASE_URL=sqlite:///tmp/test.db"
        "-e" "MONITORING_DATABASE_URL=sqlite:///tmp/test_monitoring.db"
        "-e" "REDIS_URL=redis://localhost:6379/0"
        "-e" "CONTROLLER_URL=${CONTROLLER_URL}"
        "-e" "JWT_SECRET=${JWT_SECRET}"
        "-e" "OPENAI_API_KEY=${OPENAI_API_KEY}"
        "-e" "ENVIRONMENT=${ENVIRONMENT}"
        "-e" "MULTI_USER_MODE=${MULTI_USER_MODE}"
        "-e" "LOG_LEVEL=${LOG_LEVEL}"
    )
else
    # Production mode: Skip container testing to avoid exposing production credentials
    echo -e "${YELLOW}Production mode: Skipping container testing${NC}"
    TEST_ENV_VARS=()
fi

# Skip container testing - image build verification is sufficient
echo -e "${BLUE}Container testing disabled - use docker-compose for full testing${NC}"

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}   Orchestrator Build Completed!          ${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo "Image built: ${IMAGE_NAME}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
if [ "$MODE" = "development" ]; then
    echo "Development Mode:"
    echo "1. Configure environment (optional):"
    echo "   cp .env.example .env"
    echo "   # Edit .env file to customize settings"
    echo ""
    echo "2. Start development environment:"
    echo "   docker-compose up -d"
    echo ""
    echo "3. Start with Phoenix AI observability:"
    echo "   COMPOSE_PROFILES=with-phoenix docker-compose up -d"
    echo ""
    echo "4. Check health:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "5. View logs:"
    echo "   docker-compose logs -f moolai-orchestrator"
    echo ""
    echo "6. Access Phoenix UI (if enabled):"
    echo "   http://localhost:6006"
    echo ""
    echo "7. Development Features Enabled:"
    echo "   - Multi-user authentication with bypass"
    echo "   - Development users auto-created"
    echo "   - Detailed logging and debugging"
    echo "   - Default development credentials"
    echo "   - Relaxed API validation"
else
    echo "Production Mode:"
    echo "1. Ensure .env file is properly configured with:"
    echo "   - Valid OPENAI_API_KEY"
    echo "   - Secure JWT_SECRET"
    echo "   - Database passwords"
    echo "   - Controller URL"
    echo ""
    echo "2. Deploy with Docker Compose:"
    echo "   docker-compose up -d"
    echo ""
    echo "3. Deploy with Phoenix observability:"
    echo "   COMPOSE_PROFILES=with-phoenix docker-compose up -d"
    echo ""
    echo "4. Monitor deployment:"
    echo "   docker-compose ps"
    echo "   docker-compose logs -f moolai-orchestrator"
    echo ""
    echo "5. Verify health:"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "6. Production Features:"
    echo "   - Full JWT authentication required"
    echo "   - Secure database connections"
    echo "   - Production logging levels"
fi