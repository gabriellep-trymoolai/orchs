#!/usr/bin/env python3
"""
Quick test script to validate logging configuration.
Run this to see if logs are working correctly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent / "orchestrator" / "app"
sys.path.insert(0, str(app_dir))

# Set environment variables for testing
os.environ["ENVIRONMENT"] = "development"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["ORGANIZATION_ID"] = "test_org_001"
os.environ["ORCHESTRATOR_ID"] = "test_orchestrator_001"

async def test_logging():
    """Test various logging levels and features."""
    print("🧪 Testing Comprehensive Logging Infrastructure...")
    print("=" * 50)
    
    try:
        # Import and setup logging
        from core.logging_config import setup_logging, get_logger, audit_logger, log_exception
        
        # Initialize logging
        setup_logging()
        logger = get_logger(__name__)
        
        print("\n1. Testing Basic Logging Levels:")
        logger.debug("🔍 DEBUG: This is a debug message with context")
        logger.info("ℹ️  INFO: Application started successfully")
        logger.warning("⚠️  WARNING: This is a warning message")
        logger.error("❌ ERROR: This is an error message")
        logger.critical("🚨 CRITICAL: This is a critical message")
        
        print("\n2. Testing Audit Logger:")
        # Test audit logging
        audit_logger.log_action(
            action="test_action",
            user_id="test_user_123",
            session_id="test_session_456",
            resource="test_resource",
            details={"test_key": "test_value", "test_number": 42}
        )
        
        audit_logger.log_login(
            user_id="test_user_123",
            username="test_user",
            success=True,
            ip_address="192.168.1.100"
        )
        
        audit_logger.log_database_operation(
            operation="CREATE",
            table="test_table",
            record_id=123,
            user_id="test_user_123",
            changes={"name": "test", "status": "active"}
        )
        
        print("\n3. Testing Exception Logging:")
        try:
            # Cause an intentional exception
            result = 1 / 0
        except Exception as e:
            log_exception(logger, e, {
                "context": "testing_exception_logging",
                "operation": "division_by_zero",
                "user_id": "test_user_123"
            })
        
        print("\n4. Testing Structured Data:")
        logger.info(
            "User performed action",
            extra={
                "user_id": "test_user_123",
                "action": "login",
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {
                    "ip_address": "192.168.1.100",
                    "user_agent": "Test Browser 1.0",
                    "success": True
                }
            }
        )
        
        print("\n✅ Logging tests completed!")
        print("=" * 50)
        print("📋 What to check:")
        print("1. All log levels should appear above")
        print("2. Logs should be colored in development mode")
        print("3. Each log should include org_id context")
        print("4. Exception should show full traceback")
        print("5. Audit logs should have structured data")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed the new dependencies:")
        print("pip install python-json-logger colorlog")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_logging())
    if success:
        print("\n🎉 Logging infrastructure is ready!")
        print("Next steps:")
        print("1. Build Docker image: docker-compose build moolai-orchestrator")
        print("2. Start services: docker-compose up -d")
        print("3. Check logs: docker-compose logs -f moolai-orchestrator")
    else:
        print("\n❌ Fix the issues above before deploying")
    
    sys.exit(0 if success else 1)