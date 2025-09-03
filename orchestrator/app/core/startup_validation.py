"""
Startup Validation for Logging Infrastructure
==============================================
Validates logging configuration during application startup.
"""

import os
from typing import Dict, Any

from .logging_config import get_logger, audit_logger

logger = get_logger(__name__)

def validate_logging_setup() -> Dict[str, Any]:
    """
    Validate that logging is properly configured.
    
    Returns:
        Dict with validation results
    """
    validation_results = {
        "logging_configured": False,
        "environment": None,
        "log_level": None,
        "organization_id": None,
        "orchestrator_id": None,
        "issues": []
    }
    
    try:
        # Check environment configuration
        environment = os.getenv("ENVIRONMENT", "production")
        log_level = os.getenv("LOG_LEVEL", "INFO")
        org_id = os.getenv("ORGANIZATION_ID")
        orch_id = os.getenv("ORCHESTRATOR_ID")
        
        validation_results.update({
            "environment": environment,
            "log_level": log_level,
            "organization_id": org_id,
            "orchestrator_id": orch_id
        })
        
        # Validate required environment variables
        if not org_id:
            validation_results["issues"].append("ORGANIZATION_ID not set")
        
        if not orch_id:
            validation_results["issues"].append("ORCHESTRATOR_ID not set")
        
        # Test basic logging
        logger.info("🔍 Logging validation starting...")
        logger.debug(f"Environment: {environment} | Log Level: {log_level}")
        
        # Test audit logging
        audit_logger.log_action(
            action="startup_validation",
            user_id="system",
            resource="logging_infrastructure",
            details={
                "environment": environment,
                "log_level": log_level,
                "org_id": org_id,
                "orchestrator_id": orch_id
            }
        )
        
        # If we get here, logging is working
        validation_results["logging_configured"] = True
        
        if validation_results["issues"]:
            logger.warning(f"⚠️ Logging validation issues: {validation_results['issues']}")
        else:
            logger.info("✅ Logging validation completed successfully")
            
    except Exception as e:
        logger.error(f"❌ Logging validation failed: {e}")
        validation_results["issues"].append(f"Validation error: {str(e)}")
    
    return validation_results

def log_startup_summary(validation_results: Dict[str, Any]):
    """Log a summary of the startup validation."""
    logger.info("=" * 60)
    logger.info("🚀 MOOLAI ORCHESTRATOR STARTUP SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Environment: {validation_results.get('environment', 'unknown')}")
    logger.info(f"Log Level: {validation_results.get('log_level', 'unknown')}")
    logger.info(f"Organization: {validation_results.get('organization_id', 'unknown')}")
    logger.info(f"Orchestrator: {validation_results.get('orchestrator_id', 'unknown')}")
    logger.info(f"Logging Status: {'✅ OK' if validation_results.get('logging_configured') else '❌ FAILED'}")
    
    if validation_results.get("issues"):
        logger.warning(f"Issues: {len(validation_results['issues'])}")
        for issue in validation_results["issues"]:
            logger.warning(f"  - {issue}")
    
    logger.info("=" * 60)