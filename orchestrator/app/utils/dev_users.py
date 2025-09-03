"""
Development Users Management
============================

Provides development-friendly user creation and management for multi-user WebSocket support.
Works with development bypass mode - no authentication required.
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from datetime import datetime

from ..models.user import User
from ..db.database import db_manager

logger = logging.getLogger(__name__)

# Development users configuration
DEV_USERS = [
    {
        "user_id": "dinakar_dev_001_org_001",
        "username": "dinakar_developer",
        "email": "dinakar@dev.moolai.com",
        "full_name": "Dinakar Developer",
        "is_active": True,
        "is_admin": False,
        "preferred_model": "gpt-4",
        "department": "Engineering",
        "job_title": "Senior Developer"
    },
    {
        "user_id": "amogh_dev_002_org_001", 
        "username": "amogh_analyst",
        "email": "amogh@dev.moolai.com",
        "full_name": "Amogh Analyst",
        "is_active": True,
        "is_admin": False,
        "preferred_model": "gpt-3.5-turbo",
        "department": "Analytics",
        "job_title": "Data Analyst"
    },
    {
        "user_id": "gabby_dev_003_org_002",
        "username": "gabby_admin",
        "email": "gabby@dev.moolai.com", 
        "full_name": "Gabriella Admin",
        "is_active": True,
        "is_admin": True,
        "preferred_model": "gpt-4",
        "department": "Operations",
        "job_title": "System Administrator"
    }
]

class DevUserManager:
    """Manages development users with bypass authentication."""
    
    def __init__(self):
        self.is_development = os.getenv("ENVIRONMENT", "development") == "development"
        self.multi_user_enabled = os.getenv("MULTI_USER_MODE", "true").lower() == "true"
        
    async def ensure_dev_users_exist(self) -> bool:
        """Ensure development users exist in database. Returns True if users were created."""
        if not self.is_development or not self.multi_user_enabled:
            return False
            
        try:
            async for db in db_manager.get_session():
                try:
                    created_count = 0
                    
                    for user_data in DEV_USERS:
                        # Check if user already exists
                        result = await db.execute(
                            select(User).where(User.user_id == user_data["user_id"])
                        )
                        existing_user = result.scalar_one_or_none()
                        
                        if not existing_user:
                            # Create new development user
                            new_user = User(**user_data)
                            db.add(new_user)
                            created_count += 1
                            logger.info(f"Created development user: {user_data['username']}")
                    
                    if created_count > 0:
                        await db.commit()
                        logger.info(f"Created {created_count} development users")
                        return True
                    else:
                        logger.debug("All development users already exist")
                        return False
                        
                except Exception as e:
                    await db.rollback()
                    logger.error(f"Failed to create development users: {e}")
                    return False
                finally:
                    await db.close()
                    
        except Exception as e:
            logger.error(f"Database connection failed for dev user creation: {e}")
            return False
    
    async def get_or_create_dev_user(self, user_id: Optional[str], org_id: str = "org_001") -> Optional[User]:
        """Get existing user or create development user on-the-fly."""
        if not self.is_development:
            # Production mode - only return existing users
            return await self.get_existing_user(user_id, org_id) if user_id else None
            
        # Development mode with bypass
        if not user_id:
            # Return default dev user for anonymous connections
            user_id = "alice_dev_001_org_001"
            
        try:
            async for db in db_manager.get_session():
                try:
                    # Try to get existing user first
                    result = await db.execute(
                        select(User).where(User.user_id == user_id)
                    )
                    existing_user = result.scalar_one_or_none()
                    
                    if existing_user:
                        return existing_user
                    
                    # Create ad-hoc development user
                    if user_id.startswith("dev_user_"):
                        new_user = User(
                            user_id=user_id,
                            username=user_id.replace("_", "-"),
                            email=f"{user_id}@dev.moolai.com",
                            full_name=f"Dev User {user_id.split('_')[-1][:8]}",
                            is_active=True,
                            is_admin=False,
                            preferred_model="gpt-3.5-turbo",
                            department="Development",
                            job_title="Developer"
                        )
                        
                        db.add(new_user)
                        await db.commit()
                        
                        logger.info(f"Created ad-hoc development user: {user_id}")
                        return new_user
                    
                    # If not a dev user pattern, return None
                    return None
                    
                except Exception as e:
                    await db.rollback()
                    logger.error(f"Failed to get/create development user {user_id}: {e}")
                    return None
                finally:
                    await db.close()
                    
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    async def get_existing_user(self, user_id: str, org_id: str) -> Optional[User]:
        """Get existing user from database."""
        try:
            async for db in db_manager.get_session():
                try:
                    result = await db.execute(
                        select(User).where(User.user_id == user_id)
                    )
                    return result.scalar_one_or_none()
                except Exception as e:
                    logger.error(f"Failed to get user {user_id}: {e}")
                    return None
                finally:
                    await db.close()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None
    
    async def list_dev_users(self) -> list[Dict[str, Any]]:
        """List all development users for UI selection."""
        if not self.is_development or not self.multi_user_enabled:
            return []
            
        try:
            async for db in db_manager.get_session():
                try:
                    result = await db.execute(
                        select(User).where(User.user_id.like('%_dev_%')).order_by(User.username)
                    )
                    users = result.scalars().all()
                    
                    return [
                        {
                            "user_id": user.user_id,
                            "username": user.username,
                            "full_name": user.full_name,
                            "department": user.department,
                            "preferred_model": user.preferred_model,
                            "is_admin": user.is_admin
                        }
                        for user in users
                    ]
                except Exception as e:
                    logger.error(f"Failed to list development users: {e}")
                    return []
                finally:
                    await db.close()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return []

# Global instance
dev_user_manager = DevUserManager()

# Convenience functions
async def ensure_dev_users() -> bool:
    """Ensure development users exist. Call during startup."""
    return await dev_user_manager.ensure_dev_users_exist()

async def get_or_create_user(user_id: Optional[str], org_id: str = "org_001") -> Optional[User]:
    """Get or create user based on development mode."""
    return await dev_user_manager.get_or_create_dev_user(user_id, org_id)

async def list_available_dev_users() -> list[Dict[str, Any]]:
    """List available development users for frontend."""
    return await dev_user_manager.list_dev_users()