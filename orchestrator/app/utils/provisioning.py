# Session-aware provisioning utilities for dynamic configuration updates
from __future__ import annotations
from typing import Dict, Any
from copy import deepcopy
import logging

logger = logging.getLogger("session_provisioning")

def apply_session_provisioning(data: Dict[str, Any], config_manager) -> bool:
    """
    Apply incoming provisioning data to session-aware configuration.
    
    Returns True if the config actually changed.
    Supports session-aware features and dynamic updates.
    """
    if not isinstance(data, dict):
        return False

    org_id = data.get("orchestrator_id")  # optional; don't hard-fail if absent
    incoming_features = data.get("features") or {}
    incoming_meta = data.get("metadata") or {}
    incoming_session = data.get("session") or {}
    name = data.get("name")
    location = data.get("location")
    replace_features = bool(data.get("replace_features", False))  # optional behavior knob

    # current cfg snapshot
    curr = config_manager.get_config()
    new = deepcopy(curr)

    # --- features merge ---
    new.setdefault("features", {})
    if replace_features:
        new["features"].clear()
    for k, v in incoming_features.items():
        new["features"][k] = v

    # --- session configuration merge ---
    new.setdefault("session", {})
    for k, v in incoming_session.items():
        new["session"][k] = v

    # --- metadata merge ---
    new.setdefault("metadata", {})
    new["metadata"].update(incoming_meta)

    # --- org name/location (if provided) ---
    org = new.setdefault("organization", {})
    if name:
        org["name"] = name
    if location:
        org["location"] = location

    # --- orchestrator ID update (if provided) ---
    if org_id:
        new["orchestrator_id"] = org_id

    # change detection (structural)
    if new == curr:
        logger.debug("[SESSION-PROVISIONING] no changes detected")
        return False  # no-op

    # persist (change-aware + throttled)
    try:
        if hasattr(config_manager, "update_config_if_changed"):
            # returns True only when it wrote (debounce may skip the immediate write)
            wrote = config_manager.update_config_if_changed(new, reason="session_provisioning")
            logger.info(f"[SESSION-PROVISIONING] configuration updated (wrote: {wrote})")
            # even if debounced, we still changed in-memory; signal True to caller
            return True
        else:
            config_manager.update_config(new)
            logger.info("[SESSION-PROVISIONING] configuration updated (legacy method)")
            return True
    except Exception as e:
        logger.error(f"[SESSION-PROVISIONING] failed to update config: {e}")
        return False

def apply_runtime_feature_toggle(feature_name: str, enabled: bool, config_manager) -> bool:
    """
    Toggle a feature flag at runtime without full provisioning.
    
    Returns True if the toggle was applied successfully.
    """
    try:
        patch = {"features": {feature_name: enabled}}
        updated_config = config_manager.patch_config(patch)
        
        logger.info(f"[SESSION-PROVISIONING] feature '{feature_name}' {'enabled' if enabled else 'disabled'}")
        return True
        
    except Exception as e:
        logger.error(f"[SESSION-PROVISIONING] failed to toggle feature '{feature_name}': {e}")
        return False

def update_session_limits(max_sessions: int = None, timeout_seconds: int = None, config_manager = None) -> bool:
    """
    Update session limits dynamically.
    
    Returns True if limits were updated successfully.
    """
    if not config_manager:
        return False
        
    try:
        session_patch = {}
        if max_sessions is not None:
            session_patch["max_concurrent_sessions"] = max_sessions
        if timeout_seconds is not None:
            session_patch["timeout_seconds"] = timeout_seconds
            
        if not session_patch:
            return False
            
        patch = {"session": session_patch}
        config_manager.patch_config(patch)
        
        logger.info(f"[SESSION-PROVISIONING] session limits updated: {session_patch}")
        return True
        
    except Exception as e:
        logger.error(f"[SESSION-PROVISIONING] failed to update session limits: {e}")
        return False

def get_provisioning_status(config_manager) -> Dict[str, Any]:
    """
    Get current provisioning status and configuration summary.
    """
    try:
        config = config_manager.get_config()
        
        return {
            "orchestrator_id": config.get("orchestrator_id"),
            "organization": config.get("organization", {}),
            "features": config.get("features", {}),
            "session_config": config.get("session", {}),
            "status": config.get("status"),
            "last_seen": config.get("last_seen"),
            "metadata": {
                "version": config.get("metadata", {}).get("version"),
                "last_updated": config.get("metadata", {}).get("last_updated"),
                "capabilities": config.get("metadata", {}).get("capabilities", [])
            }
        }
        
    except Exception as e:
        logger.error(f"[SESSION-PROVISIONING] failed to get status: {e}")
        return {"error": str(e)}

__all__ = [
    "apply_session_provisioning", 
    "apply_runtime_feature_toggle",
    "update_session_limits",
    "get_provisioning_status"
]