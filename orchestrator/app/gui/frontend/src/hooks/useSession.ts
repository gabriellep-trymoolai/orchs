/**
 * Session Management Hook for MoolAI Orchestrator
 * Integrates with backend session state management
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { webSocketService, SessionData, ConnectionState } from '../services/websocket';
import { apiClient } from '../services/api-client';

export interface SessionState {
  // Connection state
  connectionState: ConnectionState;
  isConnected: boolean;
  isConnecting: boolean;
  
  // Session data
  sessionData: SessionData | null;
  sessionId: string | null;
  userId: string | null;
  
  // Error handling
  lastError: string | null;
  
  // Statistics
  reconnectAttempts: number;
  lastActivity: Date | null;
}

export interface SessionActions {
  // Connection management
  connect: (params?: { user_id?: string; session_id?: string; token?: string }) => Promise<SessionData>;
  disconnect: () => void;
  
  // Session operations
  createSession: (sessionName?: string) => Promise<void>;
  joinSession: (sessionId: string) => Promise<void>;
  
  // Messaging
  sendMessage: (message: string, conversationId?: string) => Promise<void>;
  sendRawMessage: (message: any) => void;
  
  // Event listeners
  addEventListener: (eventType: string, callback: (data: any) => void) => () => void;
  
  // Utilities
  clearError: () => void;
  refreshSession: () => Promise<void>;
}

export interface UseSessionOptions {
  autoConnect?: boolean;
  autoReconnect?: boolean;
  userId?: string;
  sessionId?: string;
  onSessionEstablished?: (session: SessionData) => void;
  onSessionLost?: () => void;
  onError?: (error: string) => void;
}

export function useSession(options: UseSessionOptions = {}): SessionState & SessionActions {
  const {
    autoConnect = false,
    autoReconnect = true,
    userId: initialUserId,
    sessionId: initialSessionId,
    onSessionEstablished,
    onSessionLost,
    onError
  } = options;

  // State management
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected');
  const [sessionData, setSessionData] = useState<SessionData | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const [lastActivity, setLastActivity] = useState<Date | null>(null);
  
  // Refs for stable callbacks
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const eventListenersRef = useRef<(() => void)[]>([]);

  // Derived state
  const isConnected = connectionState === 'connected';
  const isConnecting = connectionState === 'connecting';
  const sessionId = sessionData?.session_id || null;
  const userId = sessionData?.user_id || initialUserId || null;

  // Clear error
  const clearError = useCallback(() => {
    setLastError(null);
  }, []);

  // WebSocket event handling
  useEffect(() => {
    const cleanupFunctions: (() => void)[] = [];

    // Connection state listener
    const stateCleanup = webSocketService.onStateChange((state) => {
      setConnectionState(state);
      
      if (state === 'error' && autoReconnect) {
        setReconnectAttempts(prev => prev + 1);
        
        // Schedule reconnect after delay
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        
        reconnectTimeoutRef.current = setTimeout(() => {
          if (connectionState === 'error') {
            connect().catch(error => {
              setLastError(error.message);
              onError?.(error.message);
            });
          }
        }, Math.min(5000 * Math.pow(2, reconnectAttempts), 30000));
      }
      
      if (state === 'connected') {
        setReconnectAttempts(0);
        setLastActivity(new Date());
      }
    });
    cleanupFunctions.push(stateCleanup);

    // Session data listener
    const sessionCleanup = webSocketService.onSessionChange((session) => {
      setSessionData(session);
      
      if (session) {
        onSessionEstablished?.(session);
        setLastActivity(new Date());
      } else {
        onSessionLost?.();
      }
    });
    cleanupFunctions.push(sessionCleanup);

    // Error handling
    const errorCleanup = webSocketService.addEventListener('error', (errorData) => {
      const errorMessage = errorData?.error || errorData?.message || 'Unknown WebSocket error';
      setLastError(errorMessage);
      onError?.(errorMessage);
    });
    cleanupFunctions.push(errorCleanup);

    // Activity tracking
    const activityEvents = ['message_received', 'assistant_response', 'heartbeat_ack'];
    activityEvents.forEach(eventType => {
      const cleanup = webSocketService.addEventListener(eventType, () => {
        setLastActivity(new Date());
      });
      cleanupFunctions.push(cleanup);
    });

    // Store cleanup functions
    eventListenersRef.current = cleanupFunctions;

    return () => {
      cleanupFunctions.forEach(cleanup => cleanup());
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [autoReconnect, reconnectAttempts, onSessionEstablished, onSessionLost, onError]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && connectionState === 'disconnected') {
      connect().catch(error => {
        setLastError(error.message);
        onError?.(error.message);
      });
    }
  }, [autoConnect]);

  // Connection management
  const connect = useCallback(async (params?: { 
    user_id?: string; 
    session_id?: string; 
    token?: string 
  }) => {
    try {
      setLastError(null);
      
      const connectParams = {
        user_id: params?.user_id || initialUserId,
        session_id: params?.session_id || initialSessionId,
        token: params?.token
      };

      const session = await webSocketService.connect(connectParams);
      return session;
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Connection failed';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    }
  }, [initialUserId, initialSessionId, onError]);

  const disconnect = useCallback(() => {
    try {
      webSocketService.disconnect();
      setLastError(null);
      setReconnectAttempts(0);
      
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Disconnect failed';
      setLastError(errorMessage);
    }
  }, []);

  // Session operations
  const createSession = useCallback(async (sessionName?: string) => {
    try {
      if (!userId) {
        throw new Error('User ID required to create session');
      }

      const sessionData = await apiClient.createChatSession({
        user_id: userId,
        session_name: sessionName,
        metadata: { created_via: 'ui' }
      });

      // Reconnect with new session ID
      if (isConnected) {
        await webSocketService.disconnect();
      }
      
      await connect({ 
        user_id: userId, 
        session_id: sessionData.session_id 
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create session';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    }
  }, [userId, isConnected, connect, onError]);

  const joinSession = useCallback(async (targetSessionId: string) => {
    try {
      if (!userId) {
        throw new Error('User ID required to join session');
      }

      // Disconnect current session
      if (isConnected) {
        webSocketService.disconnect();
      }
      
      // Connect to specific session
      await connect({ 
        user_id: userId, 
        session_id: targetSessionId 
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to join session';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    }
  }, [userId, isConnected, connect, onError]);

  // Messaging
  const sendMessage = useCallback(async (message: string, conversationId?: string) => {
    try {
      if (!isConnected) {
        throw new Error('Not connected to session');
      }

      await webSocketService.sendChatMessage({
        conversation_id: conversationId || sessionId || 'default',
        message,
        metadata: { timestamp: new Date().toISOString() }
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    }
  }, [isConnected, sessionId, onError]);

  const sendRawMessage = useCallback((message: any) => {
    try {
      if (!isConnected) {
        throw new Error('Not connected to session');
      }
      
      webSocketService.send(message);
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send raw message';
      setLastError(errorMessage);
      onError?.(errorMessage);
    }
  }, [isConnected, onError]);

  // Event management
  const addEventListener = useCallback((eventType: string, callback: (data: any) => void) => {
    return webSocketService.addEventListener(eventType, callback);
  }, []);

  // Utilities
  const refreshSession = useCallback(async () => {
    try {
      if (!sessionId || !userId) {
        throw new Error('No active session to refresh');
      }

      // Reconnect with current session
      if (isConnected) {
        webSocketService.disconnect();
      }
      
      await connect({ 
        user_id: userId, 
        session_id: sessionId 
      });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to refresh session';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    }
  }, [sessionId, userId, isConnected, connect, onError]);

  return {
    // State
    connectionState,
    isConnected,
    isConnecting,
    sessionData,
    sessionId,
    userId,
    lastError,
    reconnectAttempts,
    lastActivity,
    
    // Actions
    connect,
    disconnect,
    createSession,
    joinSession,
    sendMessage,
    sendRawMessage,
    addEventListener,
    clearError,
    refreshSession
  };
}