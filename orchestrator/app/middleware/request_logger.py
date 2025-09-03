"""
Request Logging Middleware
===========================
Logs all HTTP requests and responses with timing and context.
"""

import time
import uuid
import json
import logging
from typing import Callable, Optional, Dict, Any
from fastapi import Request, Response
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message

from ..core.logging_config import get_logger, log_exception

logger = get_logger(__name__)

# Headers to exclude from logging (sensitive data)
EXCLUDED_HEADERS = {
    'authorization',
    'cookie',
    'x-api-key',
    'x-auth-token',
    'x-access-token'
}

# Paths to exclude from body logging
EXCLUDED_BODY_PATHS = {
    '/api/v1/auth/login',  # Don't log passwords
    '/api/v1/auth/register',
    '/api/v1/auth/change-password'
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all HTTP requests and responses.
    
    Features:
    - Request/response timing
    - Correlation IDs for request tracking
    - Body capture (with sensitive data filtering)
    - Error logging with context
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate correlation ID for this request
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Start timing
        start_time = time.time()
        
        # Extract request info
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)
        client_host = request.client.host if request.client else "unknown"
        
        # Get user info if available
        user_id = getattr(request.state, "user_id", None)
        session_id = getattr(request.state, "session_id", None)
        
        # Log request
        request_log = {
            "correlation_id": correlation_id,
            "method": method,
            "path": path,
            "query_params": query_params if query_params else None,
            "client_host": client_host,
            "user_id": user_id,
            "session_id": session_id,
            "headers": self._get_safe_headers(request.headers)
        }
        
        # Log request body for non-excluded paths
        if method in ["POST", "PUT", "PATCH"] and path not in EXCLUDED_BODY_PATHS:
            try:
                # Store body for later use
                body_bytes = await request.body()
                request._body = body_bytes  # Store for handler use
                
                if body_bytes:
                    try:
                        body_json = json.loads(body_bytes)
                        # Sanitize sensitive fields
                        body_json = self._sanitize_body(body_json)
                        request_log["body"] = body_json
                    except json.JSONDecodeError:
                        request_log["body"] = f"<binary data: {len(body_bytes)} bytes>"
            except Exception as e:
                logger.warning(f"Could not read request body: {e}")
        
        logger.info(f"Request received: {method} {path}", extra=request_log)
        
        # Process request
        response = None
        error_occurred = False
        error_detail = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error_occurred = True
            error_detail = str(e)
            log_exception(logger, e, {"correlation_id": correlation_id, "request": request_log})
            raise
        finally:
            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Log response
            response_log = {
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "status_code": response.status_code if response else 500,
                "duration_ms": duration_ms,
                "user_id": user_id,
                "session_id": session_id
            }
            
            if error_occurred:
                response_log["error"] = error_detail
                logger.error(f"Request failed: {method} {path}", extra=response_log)
            else:
                # Determine log level based on status code
                status_code = response.status_code if response else 500
                if status_code >= 500:
                    logger.error(f"Request completed with error: {method} {path}", extra=response_log)
                elif status_code >= 400:
                    logger.warning(f"Request completed with client error: {method} {path}", extra=response_log)
                else:
                    logger.info(f"Request completed: {method} {path}", extra=response_log)
        
        return response
    
    def _get_safe_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Get headers with sensitive values masked."""
        safe_headers = {}
        for key, value in headers.items():
            if key.lower() in EXCLUDED_HEADERS:
                safe_headers[key] = "<REDACTED>"
            else:
                safe_headers[key] = value
        return safe_headers
    
    def _sanitize_body(self, body: Any) -> Any:
        """Recursively sanitize sensitive fields in request body."""
        if isinstance(body, dict):
            sanitized = {}
            for key, value in body.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'secret', 'key', 'auth']):
                    sanitized[key] = "<REDACTED>"
                elif isinstance(value, (dict, list)):
                    sanitized[key] = self._sanitize_body(value)
                else:
                    sanitized[key] = value
            return sanitized
        elif isinstance(body, list):
            return [self._sanitize_body(item) for item in body]
        else:
            return body


class LoggingRoute(APIRoute):
    """
    Custom APIRoute class that adds logging context.
    Can be used for route-specific logging enhancement.
    """
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Add route-specific context
            request.state.route_name = self.name
            request.state.route_path = self.path
            
            # Call original handler
            response = await original_route_handler(request)
            return response
        
        return custom_route_handler


def setup_request_logging(app):
    """
    Setup request logging for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Request logging middleware initialized")