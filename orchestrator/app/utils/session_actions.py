# Session Action Processing System
from __future__ import annotations
import logging
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone

from .session_state import SessionState, SessionManager, get_session_manager

logger = logging.getLogger("session_actions")

@dataclass
class ActionResult:
    """Result of processing a session action."""
    success: bool
    action_type: str
    data: Dict[str, Any]
    side_effects: List[str]  # What happened (buffer_update, persistence, etc.)
    new_state: Optional[SessionState] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

class SessionActionProcessor:
    """Processes session actions with state validation."""
    
    def __init__(self, buffer_manager=None, config_manager=None):
        self.buffer_manager = buffer_manager
        self.config_manager = config_manager
        self.session_manager = get_session_manager()
        
        # Map action types to handler methods
        self.action_handlers = {
            "connect": self._handle_connect_action,
            "send_message": self._handle_message_action,
            "join_conversation": self._handle_join_conversation_action,
            "heartbeat": self._handle_heartbeat_action,
            "disconnect": self._handle_disconnect_action,
            "analytics_subscribe": self._handle_analytics_subscribe_action,
            "analytics_unsubscribe": self._handle_analytics_unsubscribe_action,
            "analytics_request": self._handle_analytics_request_action
        }
    
    async def process_action(self, 
                           msg: Dict[str, Any], 
                           user_id: str, 
                           session_id: str, 
                           current_state: SessionState) -> ActionResult:
        """Process action with state validation."""
        action_type = msg.get("type")
        
        try:
            # Validate action is allowed in current state
            if not self._is_action_valid(action_type, current_state):
                logger.warning(f"[ACTION-PROC] Invalid action '{action_type}' in state '{current_state.value}' for session {session_id}")
                return ActionResult(
                    success=False,
                    action_type=action_type,
                    data={},
                    side_effects=[],
                    error=f"Action '{action_type}' not valid in state '{current_state.value}'"
                )
            
            # Get the appropriate handler
            handler = self.action_handlers.get(action_type)
            if not handler:
                logger.warning(f"[ACTION-PROC] Unknown action type: {action_type}")
                return ActionResult(
                    success=False,
                    action_type=action_type,
                    data={},
                    side_effects=[],
                    error=f"Unknown action type: {action_type}"
                )
            
            # Process the action
            logger.debug(f"[ACTION-PROC] Processing {action_type} for session {session_id}")
            result = await handler(msg, user_id, session_id, current_state)
            
            # Update session activity
            if result.success:
                self.session_manager.update_activity(
                    session_id,
                    increment_messages=(action_type == "send_message"),
                    custom_data=result.metadata
                )
            
            return result
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Error processing {action_type} for session {session_id}: {e}")
            return ActionResult(
                success=False,
                action_type=action_type,
                data={},
                side_effects=[],
                error=f"Internal error: {str(e)}"
            )
    
    def _is_action_valid(self, action_type: str, state: SessionState) -> bool:
        """Validate if action is allowed in current state."""
        action_state_map = {
            "connect": [SessionState.INITIALIZING],
            "send_message": [SessionState.ACTIVE, SessionState.IDLE],
            "join_conversation": [SessionState.ACTIVE, SessionState.IDLE],
            "heartbeat": [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.IDLE],
            "disconnect": [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.IDLE, SessionState.DISCONNECTING],
            "analytics_subscribe": [SessionState.ACTIVE, SessionState.IDLE],
            "analytics_unsubscribe": [SessionState.ACTIVE, SessionState.IDLE],
            "analytics_request": [SessionState.ACTIVE, SessionState.IDLE]
        }
        
        allowed_states = action_state_map.get(action_type, [])
        return state in allowed_states
    
    async def _handle_connect_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle session connection with buffer manager integration."""
        side_effects = []
        
        try:
            # Update buffer manager if available
            if self.buffer_manager:
                metadata = msg.get("metadata", {})
                metadata.update({
                    "session_id": session_id,
                    "connected_at": _now_iso(),
                    "connection_type": "websocket_session"
                })
                
                self.buffer_manager.update_active_user(
                    user_id=user_id,
                    orch_id=session_id,
                    metadata=metadata
                )
                side_effects.append("buffer_manager_updated")
            
            # Update config manager presence if available
            if self.config_manager:
                try:
                    self.config_manager.touch_presence(status="active")
                    self.config_manager.update_session_activity(1)
                    side_effects.append("presence_updated")
                except Exception as e:
                    logger.warning(f"[ACTION-PROC] Config manager update failed: {e}")
            
            # Load existing conversations (placeholder for now)
            conversations = []  # TODO: Load from database
            
            return ActionResult(
                success=True,
                action_type="connect",
                data={
                    "session_id": session_id,
                    "user_id": user_id,
                    "created_at": _now_iso(),
                    "conversations": conversations,
                    "session_metadata": {
                        "connection_method": "websocket",
                        "established_at": _now_iso()
                    }
                },
                side_effects=side_effects,
                new_state=SessionState.ACTIVE,
                metadata={"connection_established": True}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Connect action failed: {e}")
            return ActionResult(
                success=False,
                action_type="connect",
                data={},
                side_effects=side_effects,
                error=str(e)
            )
    
    async def _handle_message_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle message sending with conversation management."""
        side_effects = []
        
        try:
            # Extract message details
            conversation_id = msg.get("conversation_id") or str(uuid.uuid4())
            message_content = msg.get("message", "")
            message_id = str(uuid.uuid4())
            
            # Add to buffer manager if available
            if self.buffer_manager:
                metadata = {
                    "session_id": session_id,
                    "conversation_id": conversation_id,
                    "message_type": "user",
                    "source": "websocket_session"
                }
                
                self.buffer_manager.add_prompt(
                    prompt_id=message_id,
                    user_id=user_id,
                    prompt=message_content,
                    response=None,
                    metadata=metadata
                )
                side_effects.append("message_buffered")
            
            # Update session conversation context
            self.session_manager.update_activity(
                session_id,
                conversation_id=conversation_id,
                custom_data={"last_message_id": message_id}
            )
            side_effects.append("session_updated")
            
            # Get sequence number (placeholder)
            sequence_number = 1  # TODO: Get actual sequence from conversation history
            
            return ActionResult(
                success=True,
                action_type="send_message",
                data={
                    "message_id": message_id,
                    "conversation_id": conversation_id,
                    "sequence_number": sequence_number,
                    "processing_status": "received",
                    "content_length": len(message_content)
                },
                side_effects=side_effects,
                new_state=SessionState.PROCESSING,
                metadata={
                    "message_processed": True,
                    "conversation_id": conversation_id
                }
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Message action failed: {e}")
            return ActionResult(
                success=False,
                action_type="send_message",
                data={},
                side_effects=side_effects,
                error=str(e)
            )
    
    async def _handle_join_conversation_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle conversation join with history loading."""
        side_effects = []
        
        try:
            conversation_id = msg.get("conversation_id")
            if not conversation_id:
                return ActionResult(
                    success=False,
                    action_type="join_conversation",
                    data={},
                    side_effects=[],
                    error="conversation_id required"
                )
            
            # Update session conversation context
            self.session_manager.update_activity(
                session_id,
                conversation_id=conversation_id,
                custom_data={"joined_conversation": conversation_id}
            )
            side_effects.append("conversation_context_updated")
            
            # Load conversation history (placeholder)
            messages = []  # TODO: Load actual messages from database
            conversation_title = f"Conversation {conversation_id[:8]}"  # Placeholder
            
            return ActionResult(
                success=True,
                action_type="join_conversation",
                data={
                    "conversation_id": conversation_id,
                    "title": conversation_title,
                    "messages": messages,
                    "message_count": len(messages),
                    "joined_at": _now_iso()
                },
                side_effects=side_effects,
                new_state=SessionState.ACTIVE,
                metadata={"active_conversation": conversation_id}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Join conversation action failed: {e}")
            return ActionResult(
                success=False,
                action_type="join_conversation",
                data={},
                side_effects=side_effects,
                error=str(e)
            )
    
    async def _handle_heartbeat_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle heartbeat/keepalive with presence update."""
        side_effects = []
        
        try:
            # Update buffer manager if available
            if self.buffer_manager:
                metadata = {
                    "heartbeat": _now_iso(),
                    "session_id": session_id
                }
                self.buffer_manager.update_active_user(
                    user_id=user_id,
                    orch_id=session_id,
                    metadata=metadata
                )
                side_effects.append("heartbeat_recorded")
            
            # Update presence (throttled)
            if self.config_manager:
                try:
                    self.config_manager.touch_presence(status="active")
                    side_effects.append("presence_updated")
                except Exception as e:
                    logger.debug(f"[ACTION-PROC] Presence update throttled or failed: {e}")
            
            return ActionResult(
                success=True,
                action_type="heartbeat",
                data={
                    "timestamp": _now_iso(),
                    "session_status": "active"
                },
                side_effects=side_effects,
                metadata={"heartbeat_processed": True}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Heartbeat action failed: {e}")
            return ActionResult(
                success=False,
                action_type="heartbeat",
                data={},
                side_effects=side_effects,
                error=str(e)
            )
    
    async def _handle_disconnect_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle session disconnect with cleanup."""
        side_effects = []
        
        try:
            # Remove from buffer manager if available
            if self.buffer_manager:
                self.buffer_manager.remove_active_user(user_id)
                side_effects.append("user_removed_from_buffer")
            
            # Update config manager
            if self.config_manager:
                try:
                    self.config_manager.update_session_activity(0)
                    side_effects.append("session_activity_cleared")
                except Exception as e:
                    logger.warning(f"[ACTION-PROC] Config manager cleanup failed: {e}")
            
            return ActionResult(
                success=True,
                action_type="disconnect",
                data={
                    "session_id": session_id,
                    "disconnected_at": _now_iso(),
                    "cleanup_completed": True
                },
                side_effects=side_effects,
                new_state=SessionState.EXPIRED,
                metadata={"disconnection_processed": True}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Disconnect action failed: {e}")
            return ActionResult(
                success=False,
                action_type="disconnect",
                data={},
                side_effects=side_effects,
                error=str(e)
            )

    async def _handle_analytics_subscribe_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle analytics subscription request."""
        side_effects = []
        
        try:
            # Mark session as subscribed to analytics
            self.session_manager.update_activity(
                session_id,
                custom_data={"analytics_subscribed": True, "subscribed_at": _now_iso()}
            )
            side_effects.append("analytics_subscription_recorded")
            
            # Update buffer manager if available
            if self.buffer_manager:
                metadata = {
                    "analytics_subscribed": True,
                    "session_id": session_id,
                    "subscribed_at": _now_iso()
                }
                self.buffer_manager.update_active_user(
                    user_id=user_id,
                    orch_id=session_id,
                    metadata=metadata
                )
                side_effects.append("buffer_subscription_updated")
            
            return ActionResult(
                success=True,
                action_type="analytics_subscribe",
                data={
                    "subscribed": True,
                    "session_id": session_id,
                    "subscribed_at": _now_iso()
                },
                side_effects=side_effects,
                metadata={"analytics_subscription": True}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Analytics subscribe action failed: {e}")
            return ActionResult(
                success=False,
                action_type="analytics_subscribe",
                data={},
                side_effects=side_effects,
                error=str(e)
            )

    async def _handle_analytics_unsubscribe_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle analytics unsubscribe request."""
        side_effects = []
        
        try:
            # Mark session as unsubscribed from analytics
            self.session_manager.update_activity(
                session_id,
                custom_data={"analytics_subscribed": False, "unsubscribed_at": _now_iso()}
            )
            side_effects.append("analytics_unsubscription_recorded")
            
            # Update buffer manager if available
            if self.buffer_manager:
                metadata = {
                    "analytics_subscribed": False,
                    "session_id": session_id,
                    "unsubscribed_at": _now_iso()
                }
                self.buffer_manager.update_active_user(
                    user_id=user_id,
                    orch_id=session_id,
                    metadata=metadata
                )
                side_effects.append("buffer_unsubscription_updated")
            
            return ActionResult(
                success=True,
                action_type="analytics_unsubscribe",
                data={
                    "subscribed": False,
                    "session_id": session_id,
                    "unsubscribed_at": _now_iso()
                },
                side_effects=side_effects,
                metadata={"analytics_subscription": False}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Analytics unsubscribe action failed: {e}")
            return ActionResult(
                success=False,
                action_type="analytics_unsubscribe",
                data={},
                side_effects=side_effects,
                error=str(e)
            )

    async def _handle_analytics_request_action(self, msg: Dict[str, Any], user_id: str, session_id: str, current_state: SessionState) -> ActionResult:
        """Handle analytics data request."""
        side_effects = []
        
        try:
            # Extract request parameters
            request_data = msg.get("data", {})
            start_date = request_data.get("start_date")
            end_date = request_data.get("end_date")
            
            # Record analytics request
            self.session_manager.update_activity(
                session_id,
                custom_data={
                    "last_analytics_request": _now_iso(),
                    "request_params": request_data
                }
            )
            side_effects.append("analytics_request_recorded")
            
            return ActionResult(
                success=True,
                action_type="analytics_request",
                data={
                    "request_accepted": True,
                    "session_id": session_id,
                    "request_params": request_data,
                    "requested_at": _now_iso()
                },
                side_effects=side_effects,
                metadata={"analytics_request": request_data}
            )
            
        except Exception as e:
            logger.error(f"[ACTION-PROC] Analytics request action failed: {e}")
            return ActionResult(
                success=False,
                action_type="analytics_request",
                data={},
                side_effects=side_effects,
                error=str(e)
            )

# Global action processor instance will be created with dependencies
_action_processor_instance = None

def get_action_processor(buffer_manager=None, config_manager=None) -> SessionActionProcessor:
    """Get action processor instance with dependencies."""
    global _action_processor_instance
    
    if _action_processor_instance is None:
        _action_processor_instance = SessionActionProcessor(buffer_manager, config_manager)
    
    return _action_processor_instance

def reset_action_processor():
    """Reset global instance (for testing)."""
    global _action_processor_instance
    _action_processor_instance = None

__all__ = [
    "ActionResult",
    "SessionActionProcessor", 
    "get_action_processor",
    "reset_action_processor"
]