"""
Authentication Service
======================

Provides JWT authentication with development bypass support.
Development mode allows login with username/password without requiring JWT tokens.
"""

import os
import jwt
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.user import User
from ..db.database import db_manager
from ..utils.dev_users import get_or_create_user

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

# Development Configuration
IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "development") == "development"
MULTI_USER_ENABLED = os.getenv("MULTI_USER_MODE", "true").lower() == "true"

class AuthService:
    """Authentication service with development bypass support."""
    
    def __init__(self):
        self.is_development = IS_DEVELOPMENT
        self.multi_user_enabled = MULTI_USER_ENABLED
        
    async def authenticate_user(self, username: str, password: str, org_id: str = "org_001") -> Optional[Dict[str, Any]]:
        """
        Authenticate user with development bypass.
        
        Development mode: Password is optional, returns user if exists
        Production mode: Validates actual password (to be implemented)
        """
        try:
            async for db in db_manager.get_session():
                try:
                    # Find user by username
                    result = await db.execute(
                        select(User).where(User.username == username)
                    )
                    user = result.scalar_one_or_none()
                    
                    if not user:
                        if self.is_development and self.multi_user_enabled:
                            logger.info(f"User {username} not found, but development mode enabled")
                            return None
                        else:
                            logger.warning(f"Authentication failed: User {username} not found")
                            return None
                    
                    # Check if user is active
                    if not user.is_active:
                        logger.warning(f"Authentication failed: User {username} is inactive")
                        return None
                    
                    # Development mode bypass
                    if self.is_development:
                        logger.info(f"Development mode: Bypassing password check for {username}")
                        return self._create_user_response(user)
                    
                    # Production mode password validation (to be implemented)
                    # For now, accept any password in development
                    logger.warning("Production password validation not yet implemented")
                    return self._create_user_response(user)
                    
                except Exception as e:
                    await db.rollback()
                    logger.error(f"Database error during authentication: {e}")
                    return None
                finally:
                    await db.close()
                    
        except Exception as e:
            logger.error(f"Authentication service error: {e}")
            return None
    
    def _create_user_response(self, user: User) -> Dict[str, Any]:
        """Create standardized user response."""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin,
            "preferred_model": user.preferred_model,
            "department": user.department,
            "job_title": user.job_title,
            "is_active": user.is_active
        }
    
    def create_jwt_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT token for user."""
        try:
            # Extract org_id from user_id (e.g., "alice_dev_001_org_001" -> "org_001")
            user_id = user_data["user_id"]
            org_id = "org_001"  # Default
            if "_org_" in user_id:
                org_id = user_id.split("_org_")[-1]
            
            payload = {
                "user_id": user_data["user_id"],
                "username": user_data["username"],
                "org_id": org_id,
                "is_admin": user_data.get("is_admin", False),
                "iat": datetime.now(timezone.utc),
                "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
            }
            
            token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
            logger.info(f"JWT token created for user: {user_data['username']}")
            return token
            
        except Exception as e:
            logger.error(f"JWT token creation failed: {e}")
            raise
    
    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return user data."""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            
            # Check expiry
            exp_timestamp = payload.get("exp")
            if exp_timestamp and datetime.fromtimestamp(exp_timestamp, timezone.utc) < datetime.now(timezone.utc):
                logger.warning("JWT token expired")
                return None
            
            return {
                "user_id": payload.get("user_id"),
                "username": payload.get("username"),
                "org_id": payload.get("org_id"),
                "is_admin": payload.get("is_admin", False)
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT token validation error: {e}")
            return None
    
    async def authenticate_websocket(self, user_id: Optional[str], token: Optional[str], org_id: str = "org_001") -> Optional[User]:
        """
        Authenticate WebSocket connection with development bypass.
        
        Development mode: Creates/gets user without token validation
        Production mode: Requires valid JWT token
        """
        try:
            # Development mode bypass
            if self.is_development:
                logger.debug(f"Development mode WebSocket auth for user_id: {user_id}")
                
                # Try to get/create user via development system
                user = await get_or_create_user(user_id, org_id)
                if user:
                    logger.info(f"Development WebSocket auth successful: {user.username}")
                    return user
                
                # Fallback: allow anonymous development connection
                if not user_id:
                    logger.info("Anonymous development WebSocket connection allowed")
                    # Return a mock user for anonymous connections
                    return await get_or_create_user("alice_dev_001_org_001", org_id)
                
                return None
            
            # Production mode - require JWT token
            if not token:
                logger.warning("Production mode requires JWT token for WebSocket")
                return None
            
            # Validate JWT token
            token_data = self.validate_jwt_token(token)
            if not token_data:
                logger.warning("Invalid JWT token for WebSocket")
                return None
            
            # Ensure user_id matches token
            if user_id and user_id != token_data["user_id"]:
                logger.warning(f"User ID mismatch: {user_id} != {token_data['user_id']}")
                return None
            
            # Get user from database
            async for db in db_manager.get_session():
                try:
                    result = await db.execute(
                        select(User).where(User.user_id == token_data["user_id"])
                    )
                    user = result.scalar_one_or_none()
                    
                    if user and user.is_active:
                        logger.info(f"Production WebSocket auth successful: {user.username}")
                        return user
                    else:
                        logger.warning(f"User not found or inactive: {token_data['user_id']}")
                        return None
                        
                except Exception as e:
                    logger.error(f"Database error during WebSocket auth: {e}")
                    return None
                finally:
                    await db.close()
                    
        except Exception as e:
            logger.error(f"WebSocket authentication error: {e}")
            return None
    
    async def list_development_users(self) -> list[Dict[str, Any]]:
        """List available users for development mode login."""
        if not self.is_development:
            return []
        
        try:
            from ..utils.dev_users import list_available_dev_users
            return await list_available_dev_users()
        except Exception as e:
            logger.error(f"Failed to list development users: {e}")
            return []

# Global auth service instance
auth_service = AuthService()

# Convenience functions
async def authenticate_user(username: str, password: str, org_id: str = "org_001") -> Optional[Dict[str, Any]]:
    """Authenticate user (development bypass enabled)."""
    return await auth_service.authenticate_user(username, password, org_id)

def create_jwt_token(user_data: Dict[str, Any]) -> str:
    """Create JWT token for authenticated user."""
    return auth_service.create_jwt_token(user_data)

def validate_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """Validate JWT token."""
    return auth_service.validate_jwt_token(token)

async def authenticate_websocket_user(user_id: Optional[str], token: Optional[str], org_id: str = "org_001") -> Optional[User]:
    """Authenticate WebSocket connection."""
    return await auth_service.authenticate_websocket(user_id, token, org_id)

async def list_dev_users() -> list[Dict[str, Any]]:
    """List development users for login UI."""
    return await auth_service.list_development_users()