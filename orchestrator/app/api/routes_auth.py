"""
Authentication API Routes
=========================

Provides login/logout endpoints with development bypass support.
"""

import os
import logging
from typing import Optional
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from ..services.auth_service import (
    authenticate_user, 
    create_jwt_token, 
    validate_jwt_token,
    list_dev_users
)
from ..core.logging_config import get_logger, audit_logger, log_exception

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
security = HTTPBearer(auto_error=False)

# Request/Response Models
class LoginRequest(BaseModel):
    username: str
    password: Optional[str] = None  # Optional in development mode
    org_id: Optional[str] = "org_001"

class LoginResponse(BaseModel):
    success: bool
    token: str
    user: dict
    expires_at: str
    message: str

class UserResponse(BaseModel):
    user_id: str
    username: str
    full_name: str
    department: Optional[str]
    is_admin: bool
    is_connected: bool = False

class DevUsersResponse(BaseModel):
    users: list[UserResponse]
    total: int
    development_mode: bool


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Login endpoint with development bypass.
    
    Development mode: Password optional, creates JWT for any valid username
    Production mode: Validates password and creates secure JWT
    """
    client_ip = http_request.client.host if http_request.client else "unknown"
    environment = os.getenv("ENVIRONMENT", "production").lower()
    
    logger.info(f"Login attempt | username={request.username} | org={request.org_id} | ip={client_ip} | env={environment}")
    
    try:
        # Authenticate user
        user_data = await authenticate_user(
            username=request.username,
            password=request.password or "",  # Empty password in dev mode
            org_id=request.org_id
        )
        
        if not user_data:
            logger.warning(f"❌ Authentication failed | username={request.username} | org={request.org_id} | ip={client_ip}")
            audit_logger.log_login(
                user_id=request.username,  # Use username since auth failed
                username=request.username,
                success=False,
                ip_address=client_ip
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Create JWT token
        try:
            jwt_token = create_jwt_token(user_data)
            
            # Calculate expiry time
            from datetime import timedelta
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            
            logger.info(f"✅ Login successful | user={user_data['username']} | user_id={user_data['user_id']} | org={request.org_id} | ip={client_ip}")
            
            # Log successful authentication
            audit_logger.log_login(
                user_id=user_data['user_id'],
                username=user_data['username'],
                success=True,
                ip_address=client_ip
            )
            
            return LoginResponse(
                success=True,
                token=jwt_token,
                user=user_data,
                expires_at=expires_at.isoformat(),
                message=f"Welcome back, {user_data['full_name'] or user_data['username']}!"
            )
            
        except Exception as e:
            logger.error(f"❌ JWT token creation failed for user {user_data['username']}")
            log_exception(logger, e, {"username": user_data['username'], "ip_address": client_ip})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication token creation failed"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/logout")
async def logout():
    """
    Logout endpoint.
    
    In JWT-based auth, logout is typically handled client-side by discarding the token.
    This endpoint can be used for logging/audit purposes.
    """
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@router.get("/me")
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Get current authenticated user information.
    
    Development mode: Accepts any token or no token
    Production mode: Validates JWT token
    """
    is_development = os.getenv("ENVIRONMENT", "development") == "development"
    
    if is_development and not credentials:
        # Development mode without token - return default user
        dev_users = await list_dev_users()
        if dev_users:
            default_user = dev_users[0]  # Return first dev user
            return {
                "user_id": default_user["user_id"],
                "username": default_user["username"],
                "full_name": default_user["full_name"],
                "department": default_user.get("department"),
                "is_admin": default_user.get("is_admin", False),
                "authenticated": True,
                "development_mode": True
            }
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Validate token
    token_data = validate_jwt_token(credentials.credentials)
    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    return {
        "user_id": token_data["user_id"],
        "username": token_data["username"],
        "org_id": token_data.get("org_id"),
        "is_admin": token_data.get("is_admin", False),
        "authenticated": True,
        "development_mode": is_development
    }


@router.get("/users/dev", response_model=DevUsersResponse)
async def get_development_users():
    """
    Get available development users for login.
    Only available in development mode.
    """
    is_development = os.getenv("ENVIRONMENT", "development") == "development"
    
    if not is_development:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Development users only available in development mode"
        )
    
    try:
        dev_users = await list_dev_users()
        
        users = [
            UserResponse(
                user_id=user["user_id"],
                username=user["username"],
                full_name=user["full_name"],
                department=user.get("department"),
                is_admin=user.get("is_admin", False),
                is_connected=False  # TODO: Check WebSocket connections
            )
            for user in dev_users
        ]
        
        return DevUsersResponse(
            users=users,
            total=len(users),
            development_mode=True
        )
        
    except Exception as e:
        logger.error(f"Failed to get development users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve development users"
        )


@router.get("/validate")
async def validate_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Validate JWT token.
    
    Returns token validation status and user info if valid.
    """
    if not credentials:
        return {
            "valid": False,
            "message": "No token provided"
        }
    
    token_data = validate_jwt_token(credentials.credentials)
    if not token_data:
        return {
            "valid": False,
            "message": "Invalid or expired token"
        }
    
    return {
        "valid": True,
        "user": token_data,
        "message": "Token is valid"
    }


@router.get("/status")
async def get_auth_status():
    """Get authentication system status."""
    is_development = os.getenv("ENVIRONMENT", "development") == "development"
    multi_user_enabled = os.getenv("MULTI_USER_MODE", "true").lower() == "true"
    
    return {
        "development_mode": is_development,
        "multi_user_enabled": multi_user_enabled,
        "jwt_enabled": True,
        "development_bypass": is_development,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }