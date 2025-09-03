"""
Centralized Logging Configuration
==================================
Provides comprehensive logging for all orchestrator operations.
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

# Determine environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")

class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self):
        super().__init__()
        self.organization_id = os.getenv("ORGANIZATION_ID", "unknown")
        self.orchestrator_id = os.getenv("ORCHESTRATOR_ID", "unknown")
        
    def filter(self, record):
        record.organization_id = self.organization_id
        record.orchestrator_id = self.orchestrator_id
        record.environment = ENVIRONMENT
        record.timestamp = datetime.utcnow().isoformat()
        return True


class ColoredFormatter(logging.Formatter):
    """Colored console output for development."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color for console output in development
        if ENVIRONMENT == "development" and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                record.name = f"\033[34m{record.name}\033[0m"  # Blue for logger name
        return super().format(record)


def setup_logging():
    """
    Configure comprehensive logging for the orchestrator.
    
    Features:
    - Structured JSON logging in production
    - Colored console output in development
    - Contextual information (org_id, orchestrator_id)
    - Request correlation IDs
    - SQL query logging in debug mode
    """
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # Add context filter
    context_filter = ContextFilter()
    console_handler.addFilter(context_filter)
    
    if ENVIRONMENT == "development":
        # Development: Human-readable colored output
        formatter = ColoredFormatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s | %(message)s | org:%(organization_id)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Production: Structured JSON logging
        formatter = jsonlogger.JsonFormatter(
            fmt='%(timestamp)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d %(funcName)s %(organization_id)s %(orchestrator_id)s %(environment)s',
            rename_fields={'levelname': 'level', 'name': 'logger'}
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    configure_module_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured | level={LOG_LEVEL} | env={ENVIRONMENT} | format={'json' if ENVIRONMENT != 'development' else 'colored'}"
    )


def configure_module_loggers():
    """Configure logging levels for specific modules."""
    
    # SQL logging (only in DEBUG mode)
    if LOG_LEVEL.upper() == "DEBUG":
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)
    else:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    
    # Ensure our modules log at appropriate levels
    logging.getLogger("app").setLevel(getattr(logging, LOG_LEVEL.upper()))
    logging.getLogger("orchestrator").setLevel(getattr(logging, LOG_LEVEL.upper()))
    logging.getLogger("monitoring").setLevel(getattr(logging, LOG_LEVEL.upper()))
    logging.getLogger("controller_client").setLevel(getattr(logging, LOG_LEVEL.upper()))


class AuditLogger:
    """
    Specialized logger for audit trail.
    Logs important user actions and system events.
    """
    
    def __init__(self, name="audit"):
        self.logger = logging.getLogger(f"audit.{name}")
        
    def log_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True
    ):
        """Log an auditable action."""
        log_data = {
            "audit_action": action,
            "user_id": user_id or "system",
            "session_id": session_id,
            "resource": resource,
            "success": success,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if success:
            self.logger.info(f"AUDIT: {action}", extra=log_data)
        else:
            self.logger.warning(f"AUDIT FAILURE: {action}", extra=log_data)
    
    def log_login(self, user_id: str, username: str, success: bool, ip_address: Optional[str] = None):
        """Log login attempt."""
        self.log_action(
            action="user_login",
            user_id=user_id if success else username,
            success=success,
            details={"username": username, "ip_address": ip_address}
        )
    
    def log_database_operation(
        self,
        operation: str,  # CREATE, UPDATE, DELETE
        table: str,
        record_id: Any,
        user_id: Optional[str] = None,
        changes: Optional[Dict[str, Any]] = None
    ):
        """Log database operation."""
        self.log_action(
            action=f"db_{operation.lower()}",
            user_id=user_id,
            resource=f"{table}:{record_id}",
            details={"changes": changes} if changes else None
        )


# Global audit logger instance
audit_logger = AuditLogger()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exc: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context information
    """
    import traceback
    
    exc_info = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "context": context or {}
    }
    
    logger.error(f"Exception occurred: {type(exc).__name__}: {str(exc)}", extra=exc_info, exc_info=True)