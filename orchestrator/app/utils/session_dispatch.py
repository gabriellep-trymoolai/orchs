# Enhanced Session-aware message dispatch with state management
from __future__ import annotations
import os, time, uuid, logging
from typing import Any, Dict, Optional
from datetime import datetime, timezone

# Import new hybrid system components
from .session_state import SessionState, get_session_manager
from .session_actions import get_action_processor, ActionResult
from .session_responses import get_response_generator

try:
    from .buffer_manager import buffer_manager
except ImportError:
    buffer_manager = None

logger = logging.getLogger("session_dispatch")

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# --- throttle presence writes ---
_TOUCH_MIN_SECS = float(os.getenv("SESSION_CONFIG_TOUCH_MIN_SECS", "30"))
_last_touch_write_at: float = 0.0

def _persist_session_presence(cfg_mgr, session_count: int = 0) -> None:
    """Persist lightweight presence with session context at most every _TOUCH_MIN_SECS."""
    global _last_touch_write_at, _TOUCH_MIN_SECS
    now = time.time()
    if now - _last_touch_write_at < _TOUCH_MIN_SECS:
        return
    try:
        # Update presence and session activity
        cfg_mgr.touch_presence(status="active")
        cfg_mgr.update_session_activity(session_count)
        _last_touch_write_at = now
    except Exception as e:
        logger.warning(f"[SESSION-DISPATCH] persist presence failed: {e}")

def _update_active_user_session(user_id: str, session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """Update active user with session context."""
    try:
        if buffer_manager:
            metadata = meta or {}
            if session_id:
                metadata["session_id"] = session_id
            buffer_manager.update_active_user(
                user_id=user_id, 
                orch_id=session_id or user_id, 
                metadata=metadata
            )
    except Exception as e:
        logger.warning(f"[SESSION-DISPATCH] active_user update failed: {e}")

def _add_session_prompt(user_id: str, prompt_id: str, prompt_text: str, session_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
    """Add prompt with session context."""
    try:
        if buffer_manager:
            metadata = meta or {}
            if session_id:
                metadata["session_id"] = session_id
            metadata["source"] = "websocket_session"
            buffer_manager.add_prompt(
                prompt_id=prompt_id,
                user_id=user_id,
                prompt=prompt_text,
                response=None,
                metadata=metadata,
            )
    except Exception as e:
        logger.warning(f"[SESSION-DISPATCH] add_prompt failed: {e}")

def _update_session_prompt_response(prompt_id: Optional[str], response_data: Any, session_id: Optional[str] = None):
    """Update prompt response with session context."""
    if not prompt_id:
        logger.warning("[SESSION-DISPATCH] update_prompt called with no prompt_id")
        return
    try:
        if buffer_manager:
            metadata = {"source": "websocket_session"}
            if session_id:
                metadata["session_id"] = session_id
            buffer_manager.update_prompt(
                prompt_id=prompt_id, 
                response=response_data, 
                metadata=metadata
            )
    except Exception as e:
        logger.warning(f"[SESSION-DISPATCH] update_prompt failed: {e}")

async def dispatch_session_message(msg: Dict[str, Any], user_id: str, session_id: Optional[str], cfg_mgr) -> Dict[str, Any]:
    """
    Enhanced dispatch with state machine and action separation.
    
    Returns response message to send back to client.
    """
    # Initialize hybrid system components
    session_manager = get_session_manager()
    action_processor = get_action_processor(buffer_manager, cfg_mgr)
    response_generator = get_response_generator()
    
    try:
        # Ensure session exists and get/create session_id
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        if session_id not in session_manager.sessions:
            current_state = session_manager.create_session(session_id, user_id)
            logger.info(f"[SESSION-DISPATCH] Created new session {session_id} for user {user_id}")
        else:
            current_state = session_manager.get_state(session_id)
            logger.debug(f"[SESSION-DISPATCH] Using existing session {session_id} in state {current_state.value}")
        
        # Process action through state machine
        action_result = await action_processor.process_action(
            msg, user_id, session_id, current_state
        )
        
        # Update session state if needed
        if action_result.new_state and action_result.success:
            transition_success = session_manager.transition_state(
                session_id, 
                action_result.new_state, 
                trigger=msg.get("type")
            )
            if transition_success:
                logger.debug(f"[SESSION-DISPATCH] State transition: {current_state.value} -> {action_result.new_state.value}")
            else:
                logger.warning(f"[SESSION-DISPATCH] Failed state transition: {current_state.value} -> {action_result.new_state.value}")
        
        # Generate response using response generator
        response = response_generator.generate_response(
            msg, action_result, session_id, user_id
        )
        
        # Add state information for debugging/monitoring
        current_session_state = session_manager.get_state(session_id)
        response["session_state"] = current_session_state.value
        
        # Add performance/debugging info
        if action_result.side_effects:
            response["debug_info"] = {
                "side_effects": action_result.side_effects,
                "action_success": action_result.success
            }
        
        # Log the interaction
        if action_result.success:
            logger.info(f"[SESSION-DISPATCH:{session_id}] {msg.get('type')} processed successfully -> {response.get('type')}")
        else:
            logger.warning(f"[SESSION-DISPATCH:{session_id}] {msg.get('type')} failed: {action_result.error}")
        
        return response
        
    except Exception as e:
        logger.error(f"[SESSION-DISPATCH:{session_id}] fatal error processing message: {e}")
        
        # Create fallback error response
        fallback_response = {
            "type": "error",
            "message_id": msg.get("message_id"),
            "timestamp": _now_iso(),
            "session_id": session_id,
            "user_id": user_id,
            "data": {
                "error": f"Internal dispatch error: {str(e)}",
                "error_type": "dispatch_failure"
            },
            "session_state": "error"
        }
        
        return fallback_response

def get_session_stats() -> Dict[str, Any]:
    """Get current session statistics from hybrid system."""
    session_manager = get_session_manager()
    
    # Get session manager stats
    session_stats = session_manager.get_stats()
    
    # Get buffer manager stats if available
    buffer_stats = {}
    if buffer_manager:
        try:
            buffer_stats = buffer_manager.get_stats()
        except Exception as e:
            logger.warning(f"[SESSION-DISPATCH] Failed to get buffer stats: {e}")
            buffer_stats = {"error": str(e)}
    else:
        buffer_stats = {"error": "buffer_manager not available"}
    
    return {
        "session_manager": session_stats,
        "buffer_manager": buffer_stats,
        "timestamp": _now_iso()
    }

def cleanup_expired_sessions(timeout_seconds: int = 1800) -> int:
    """Clean up expired sessions and return count of cleaned sessions."""
    session_manager = get_session_manager()
    
    try:
        # Clean up session manager
        session_cleaned = session_manager.cleanup_expired_sessions(timeout_seconds)
        
        # Clean up buffer manager if available
        buffer_cleaned = 0
        if buffer_manager:
            try:
                before_count = len(buffer_manager.active_users)
                buffer_manager.cleanup_expired(timeout_seconds)
                after_count = len(buffer_manager.active_users)
                buffer_cleaned = before_count - after_count
            except Exception as e:
                logger.warning(f"[SESSION-DISPATCH] Buffer cleanup failed: {e}")
        
        total_cleaned = session_cleaned + buffer_cleaned
        
        if total_cleaned > 0:
            logger.info(f"[SESSION-DISPATCH] cleaned up {total_cleaned} expired sessions (session_mgr: {session_cleaned}, buffer: {buffer_cleaned})")
        
        return total_cleaned
        
    except Exception as e:
        logger.error(f"[SESSION-DISPATCH] session cleanup failed: {e}")
        return 0

__all__ = [
    "dispatch_session_message", 
    "get_session_stats", 
    "cleanup_expired_sessions"
]