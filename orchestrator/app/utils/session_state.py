# Session State Management for WebSocket Communication
from __future__ import annotations
import logging
import time
from enum import Enum
from typing import Dict, Optional, Any, Set, List
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger("session_state")

class SessionState(Enum):
    """Session lifecycle states."""
    INITIALIZING = "initializing"     # Session created but not established
    ACTIVE = "active"                 # Session established and ready
    PROCESSING = "processing"         # Handling message/action
    IDLE = "idle"                    # No active conversation
    DISCONNECTING = "disconnecting"  # Clean shutdown in progress
    EXPIRED = "expired"              # Session timeout

@dataclass
class SessionMetadata:
    """Session metadata and state information."""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    state: SessionState
    conversation_id: Optional[str] = None
    connection_count: int = 0
    message_count: int = 0
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
    
    def is_expired(self, timeout_seconds: int = 1800) -> bool:
        """Check if session has expired."""
        if self.state == SessionState.EXPIRED:
            return True
        
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_activity).total_seconds()
        return elapsed > timeout_seconds

class SessionTransition:
    """Defines valid state transitions and their triggers."""
    
    VALID_TRANSITIONS: Dict[SessionState, List[SessionState]] = {
        SessionState.INITIALIZING: [SessionState.ACTIVE, SessionState.EXPIRED],
        SessionState.ACTIVE: [SessionState.PROCESSING, SessionState.IDLE, SessionState.DISCONNECTING],
        SessionState.PROCESSING: [SessionState.ACTIVE, SessionState.IDLE, SessionState.DISCONNECTING],
        SessionState.IDLE: [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.DISCONNECTING],
        SessionState.DISCONNECTING: [SessionState.EXPIRED],
        SessionState.EXPIRED: []  # Terminal state
    }
    
    @classmethod
    def is_valid_transition(cls, from_state: SessionState, to_state: SessionState) -> bool:
        """Check if state transition is valid."""
        allowed_states = cls.VALID_TRANSITIONS.get(from_state, [])
        return to_state in allowed_states
    
    @classmethod
    def get_allowed_transitions(cls, from_state: SessionState) -> List[SessionState]:
        """Get list of allowed transitions from current state."""
        return cls.VALID_TRANSITIONS.get(from_state, [])

class SessionManager:
    """Manages session states and transitions."""
    
    def __init__(self):
        self.sessions: Dict[str, SessionMetadata] = {}
        self._lock = {}  # Simple locking mechanism per session
        
    def create_session(self, session_id: str, user_id: str) -> SessionState:
        """Create new session in INITIALIZING state."""
        now = datetime.now(timezone.utc)
        
        if session_id in self.sessions:
            # Session already exists, return current state
            logger.info(f"[SESSION-MGR] Session {session_id} already exists")
            return self.sessions[session_id].state
        
        self.sessions[session_id] = SessionMetadata(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_activity=now,
            state=SessionState.INITIALIZING
        )
        
        logger.info(f"[SESSION-MGR] Created session {session_id} for user {user_id}")
        return SessionState.INITIALIZING
    
    def transition_state(self, session_id: str, new_state: SessionState, trigger: str = None) -> bool:
        """Attempt state transition with validation."""
        if session_id not in self.sessions:
            logger.warning(f"[SESSION-MGR] Cannot transition unknown session: {session_id}")
            return False
        
        session = self.sessions[session_id]
        current_state = session.state
        
        # Check if transition is valid
        if not SessionTransition.is_valid_transition(current_state, new_state):
            logger.warning(f"[SESSION-MGR] Invalid transition: {current_state.value} -> {new_state.value}")
            return False
        
        # Update state and activity
        session.state = new_state
        session.update_activity()
        
        logger.info(f"[SESSION-MGR] Session {session_id}: {current_state.value} -> {new_state.value}" + 
                   (f" (trigger: {trigger})" if trigger else ""))
        
        return True
    
    def get_state(self, session_id: str) -> SessionState:
        """Get current session state."""
        if session_id not in self.sessions:
            logger.warning(f"[SESSION-MGR] Unknown session: {session_id}")
            return SessionState.EXPIRED
        
        return self.sessions[session_id].state
    
    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata."""
        return self.sessions.get(session_id)
    
    def update_activity(self, session_id: str, **kwargs):
        """Update session activity and metadata."""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        session.update_activity()
        
        # Update specific metadata
        if 'conversation_id' in kwargs:
            session.conversation_id = kwargs['conversation_id']
        if 'increment_messages' in kwargs and kwargs['increment_messages']:
            session.message_count += 1
        if 'increment_connections' in kwargs and kwargs['increment_connections']:
            session.connection_count += 1
        if 'custom_data' in kwargs:
            session.custom_data.update(kwargs['custom_data'])
    
    def is_valid_action(self, session_id: str, action_type: str) -> bool:
        """Check if action is valid in current state."""
        if session_id not in self.sessions:
            return False
        
        current_state = self.sessions[session_id].state
        
        # Define valid actions per state
        action_state_map = {
            "connect": [SessionState.INITIALIZING],
            "send_message": [SessionState.ACTIVE, SessionState.IDLE],
            "join_conversation": [SessionState.ACTIVE, SessionState.IDLE],
            "heartbeat": [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.IDLE],
            "disconnect": [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.IDLE, SessionState.DISCONNECTING]
        }
        
        allowed_states = action_state_map.get(action_type, [])
        return current_state in allowed_states
    
    def cleanup_expired_sessions(self, timeout_seconds: int = 1800) -> int:
        """Clean up expired sessions and return count."""
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if session.is_expired(timeout_seconds):
                expired_sessions.append(session_id)
        
        # Remove expired sessions
        for session_id in expired_sessions:
            self.sessions[session_id].state = SessionState.EXPIRED
            logger.info(f"[SESSION-MGR] Expired session: {session_id}")
        
        # Actually remove them after a grace period
        to_remove = []
        for session_id, session in self.sessions.items():
            if session.state == SessionState.EXPIRED:
                # Remove if expired for more than 5 minutes
                if session.is_expired(300):  # 5 minutes
                    to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
            logger.info(f"[SESSION-MGR] Removed expired session: {session_id}")
        
        return len(expired_sessions)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        state_counts = {}
        for state in SessionState:
            state_counts[state.value] = 0
        
        for session in self.sessions.values():
            state_counts[session.state.value] += 1
        
        return {
            "total_sessions": len(self.sessions),
            "state_distribution": state_counts,
            "active_sessions": len([s for s in self.sessions.values() 
                                  if s.state in [SessionState.ACTIVE, SessionState.PROCESSING, SessionState.IDLE]])
        }
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions."""
        active_sessions = []
        for session in self.sessions.values():
            if session.state not in [SessionState.EXPIRED]:
                active_sessions.append({
                    "session_id": session.session_id,
                    "user_id": session.user_id,
                    "state": session.state.value,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "message_count": session.message_count,
                    "conversation_id": session.conversation_id
                })
        
        return active_sessions

# Global session manager instance
session_manager = SessionManager()

def get_session_manager() -> SessionManager:
    """Get global session manager instance."""
    return session_manager

__all__ = [
    "SessionState",
    "SessionMetadata", 
    "SessionTransition",
    "SessionManager",
    "session_manager",
    "get_session_manager"
]