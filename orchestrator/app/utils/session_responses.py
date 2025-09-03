# Session Response Generation System
from __future__ import annotations
import logging
from typing import Dict, Any, Callable
from datetime import datetime, timezone

from .session_actions import ActionResult

logger = logging.getLogger("session_responses")

def _now_iso() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

class SessionResponseGenerator:
    """Generates consistent WebSocket responses based on action results."""
    
    def __init__(self):
        # Map action types to response generator functions
        self.response_templates = {
            "connect": self._create_session_established_response,
            "send_message": self._create_message_received_response,
            "join_conversation": self._create_conversation_joined_response,
            "heartbeat": self._create_heartbeat_ack_response,
            "disconnect": self._create_disconnected_response,
            "analytics_subscribe": self._create_analytics_subscribe_response,
            "analytics_unsubscribe": self._create_analytics_unsubscribe_response,
            "analytics_request": self._create_analytics_request_response
        }
    
    def generate_response(self, 
                         msg: Dict[str, Any], 
                         action_result: ActionResult, 
                         session_id: str, 
                         user_id: str) -> Dict[str, Any]:
        """Generate response based on action result."""
        # Create base response structure
        base_response = {
            "message_id": msg.get("message_id"),
            "timestamp": _now_iso(),
            "session_id": session_id,
            "user_id": user_id
        }
        
        # Handle error cases
        if not action_result.success:
            return self._create_error_response(base_response, action_result)
        
        # Generate success response using appropriate template
        response_generator = self.response_templates.get(action_result.action_type)
        if response_generator:
            return response_generator(base_response, action_result, msg)
        
        # Fallback for unknown action types
        logger.warning(f"[RESPONSE-GEN] No response template for action: {action_result.action_type}")
        return self._create_generic_success_response(base_response, action_result)
    
    def _create_error_response(self, base_response: Dict[str, Any], action_result: ActionResult) -> Dict[str, Any]:
        """Create error response."""
        return {
            **base_response,
            "type": "error",
            "data": {
                "error": action_result.error,
                "action_type": action_result.action_type,
                "side_effects": action_result.side_effects
            }
        }
    
    def _create_generic_success_response(self, base_response: Dict[str, Any], action_result: ActionResult) -> Dict[str, Any]:
        """Create generic success response."""
        return {
            **base_response,
            "type": "ack",
            "data": {
                **action_result.data,
                "action_type": action_result.action_type,
                "side_effects": action_result.side_effects
            }
        }
    
    def _create_session_established_response(self, 
                                           base_response: Dict[str, Any], 
                                           action_result: ActionResult, 
                                           original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate session established response."""
        return {
            **base_response,
            "type": "session_established",
            "data": {
                "session_id": action_result.data.get("session_id", base_response["session_id"]),
                "user_id": action_result.data.get("user_id", base_response["user_id"]),
                "created_at": action_result.data.get("created_at", base_response["timestamp"]),
                "conversations": action_result.data.get("conversations", []),
                "session_metadata": action_result.data.get("session_metadata", {}),
                "capabilities": {
                    "send_message": True,
                    "join_conversation": True,
                    "heartbeat": True,
                    "disconnect": True
                }
            },
            "side_effects": action_result.side_effects
        }
    
    def _create_message_received_response(self, 
                                        base_response: Dict[str, Any], 
                                        action_result: ActionResult, 
                                        original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate message received response."""
        return {
            **base_response,
            "type": "message_received",
            "data": {
                "message_id": action_result.data.get("message_id"),
                "conversation_id": action_result.data.get("conversation_id"),
                "sequence_number": action_result.data.get("sequence_number", 1),
                "processing_status": action_result.data.get("processing_status", "received"),
                "timestamp": base_response["timestamp"],
                "content_length": action_result.data.get("content_length", 0),
                "next_steps": ["processing", "ai_response_expected"]
            },
            "side_effects": action_result.side_effects
        }
    
    def _create_conversation_joined_response(self, 
                                           base_response: Dict[str, Any], 
                                           action_result: ActionResult, 
                                           original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate conversation joined response."""
        return {
            **base_response,
            "type": "conversation_joined",
            "data": {
                "conversation_id": action_result.data.get("conversation_id"),
                "title": action_result.data.get("title", "Untitled Conversation"),
                "messages": action_result.data.get("messages", []),
                "message_count": action_result.data.get("message_count", 0),
                "joined_at": action_result.data.get("joined_at", base_response["timestamp"]),
                "conversation_metadata": {
                    "last_activity": base_response["timestamp"],
                    "participant_count": 1  # Just the user for now
                }
            },
            "side_effects": action_result.side_effects
        }
    
    def _create_heartbeat_ack_response(self, 
                                     base_response: Dict[str, Any], 
                                     action_result: ActionResult, 
                                     original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate heartbeat acknowledgment response."""
        return {
            **base_response,
            "type": "heartbeat_ack",
            "data": {
                "timestamp": action_result.data.get("timestamp", base_response["timestamp"]),
                "session_status": action_result.data.get("session_status", "active"),
                "server_time": base_response["timestamp"],
                "next_heartbeat_expected": 30  # seconds
            },
            "side_effects": action_result.side_effects
        }
    
    def _create_disconnected_response(self, 
                                    base_response: Dict[str, Any], 
                                    action_result: ActionResult, 
                                    original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate disconnection response."""
        return {
            **base_response,
            "type": "disconnected",
            "data": {
                "session_id": action_result.data.get("session_id", base_response["session_id"]),
                "disconnected_at": action_result.data.get("disconnected_at", base_response["timestamp"]),
                "cleanup_completed": action_result.data.get("cleanup_completed", True),
                "reason": "user_requested",
                "final_message": True
            },
            "side_effects": action_result.side_effects
        }
    
    def create_system_message(self, 
                            session_id: str, 
                            user_id: str, 
                            message_type: str, 
                            data: Dict[str, Any]) -> Dict[str, Any]:
        """Create system-generated message (not in response to user action)."""
        return {
            "message_id": f"system_{message_type}_{int(datetime.now().timestamp())}",
            "timestamp": _now_iso(),
            "session_id": session_id,
            "user_id": user_id,
            "type": message_type,
            "data": data,
            "source": "system"
        }
    
    def create_ai_response_message(self, 
                                 session_id: str, 
                                 user_id: str, 
                                 conversation_id: str, 
                                 content: str, 
                                 message_id: str = None, 
                                 is_complete: bool = True, 
                                 sequence_number: int = None) -> Dict[str, Any]:
        """Create AI assistant response message."""
        import uuid
        
        return {
            "message_id": message_id or str(uuid.uuid4()),
            "timestamp": _now_iso(),
            "session_id": session_id,
            "user_id": user_id,
            "type": "assistant_response",
            "data": {
                "conversation_id": conversation_id,
                "content": content,
                "content_delta": content,  # For streaming compatibility
                "is_complete": is_complete,
                "sequence_number": sequence_number or 2,
                "model_metadata": {
                    "model": "assistant",
                    "response_time": 0.0,  # Placeholder
                    "tokens_used": len(content.split())  # Rough estimate
                }
            },
            "source": "ai_assistant"
        }
    
    def create_error_message(self, 
                           session_id: str, 
                           user_id: str, 
                           error: str, 
                           error_code: str = None, 
                           correlation_id: str = None) -> Dict[str, Any]:
        """Create standalone error message."""
        return {
            "message_id": correlation_id or f"error_{int(datetime.now().timestamp())}",
            "timestamp": _now_iso(),
            "session_id": session_id,
            "user_id": user_id,
            "type": "error",
            "data": {
                "error": error,
                "error_code": error_code or "GENERAL_ERROR",
                "recoverable": True,
                "suggested_action": "retry_or_contact_support"
            },
            "source": "system"
        }

    def _create_analytics_subscribe_response(self,
                                           base_response: Dict[str, Any],
                                           action_result: ActionResult,
                                           original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics subscription response."""
        return {
            **base_response,
            "type": "ack",
            "data": {
                "action": "analytics_subscribe",
                "subscribed": action_result.data.get("subscribed", True),
                "session_id": action_result.data.get("session_id"),
                "status": "success"
            },
            "side_effects": action_result.side_effects
        }

    def _create_analytics_unsubscribe_response(self,
                                             base_response: Dict[str, Any],
                                             action_result: ActionResult,
                                             original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics unsubscription response."""
        return {
            **base_response,
            "type": "ack",
            "data": {
                "action": "analytics_unsubscribe",
                "subscribed": action_result.data.get("subscribed", False),
                "session_id": action_result.data.get("session_id"),
                "status": "success"
            },
            "side_effects": action_result.side_effects
        }

    def _create_analytics_request_response(self,
                                         base_response: Dict[str, Any],
                                         action_result: ActionResult,
                                         original_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics request response."""
        return {
            **base_response,
            "type": "ack",
            "data": {
                "action": "analytics_request",
                "request_accepted": action_result.data.get("request_accepted", True),
                "session_id": action_result.data.get("session_id"),
                "status": "success"
            },
            "side_effects": action_result.side_effects
        }

# Global response generator instance
response_generator = SessionResponseGenerator()

def get_response_generator() -> SessionResponseGenerator:
    """Get global response generator instance."""
    return response_generator

__all__ = [
    "SessionResponseGenerator",
    "response_generator", 
    "get_response_generator"
]