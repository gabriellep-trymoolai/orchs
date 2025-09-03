/**
 * Conversation Hook for MoolAI Orchestrator
 * Integrates conversation management with session and WebSocket communication
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { conversationService, Conversation, Message, ConversationSummary } from '../services/conversation';
import { useSession } from './useSession';

export interface ConversationState {
  // Current conversation
  currentConversation: Conversation | null;
  messages: Message[];
  
  // All conversations
  conversations: Conversation[];
  conversationSummaries: ConversationSummary[];
  
  // Input state
  isTyping: boolean;
  isSending: boolean;
  
  // Error handling
  lastError: string | null;
  
  // Statistics
  totalConversations: number;
  totalMessages: number;
}

export interface ConversationActions {
  // Conversation management
  createConversation: (name: string) => Conversation | null;
  switchToConversation: (id: string) => boolean;
  deleteConversation: (id: string) => boolean;
  renameConversation: (id: string, newName: string) => boolean;
  
  // Message management
  sendMessage: (content: string) => Promise<void>;
  queryLLM: (query: string, enableEvaluation?: boolean) => Promise<void>;
  addMessage: (content: string, role: 'user' | 'assistant' | 'system', metadata?: any) => Message | null;
  
  // UI state management
  setIsTyping: (typing: boolean) => void;
  clearError: () => void;
  
  // Utilities
  clearAllConversations: () => void;
  exportConversations: () => string;
  importConversations: (jsonData: string) => boolean;
}

export interface UseConversationOptions {
  session: ReturnType<typeof useSession>; // Require session to be passed in
  autoCreateConversation?: boolean;
  defaultConversationName?: string;
  enableAutoResponse?: boolean;
  onNewMessage?: (message: Message) => void;
  onError?: (error: string) => void;
}

export function useConversation(options: UseConversationOptions): ConversationState & ConversationActions {
  const {
    session,
    autoCreateConversation = true,
    defaultConversationName = 'New Chat',
    enableAutoResponse = true,
    onNewMessage,
    onError
  } = options;

  // Note: Session management is handled by AppProvider - don't create another connection here

  // State management
  const [currentConversation, setCurrentConversation] = useState<Conversation | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  // Refs for cleanup
  const cleanupFunctionsRef = useRef<(() => void)[]>([]);

  // Derived state
  const messages = currentConversation?.messages || [];
  const conversationSummaries = conversationService.getConversationSummaries();
  const totalConversations = conversations.length;
  const totalMessages = conversations.reduce((total, conv) => total + conv.messages.length, 0);

  // Initialize conversation service listeners
  useEffect(() => {
    const cleanupFunctions: (() => void)[] = [];

    // Listen to conversation changes
    const conversationCleanup = conversationService.onConversationsChange((convs) => {
      setConversations(convs);
      
      // Update current conversation reference
      const current = conversationService.getCurrentConversation();
      setCurrentConversation(current);
    });
    cleanupFunctions.push(conversationCleanup);

    // Listen to new messages
    const messageCleanup = conversationService.onNewMessage((message) => {
      onNewMessage?.(message);
      
      // Update current conversation if it's the target
      const current = conversationService.getCurrentConversation();
      if (current && current.id === message.conversation_id) {
        setCurrentConversation({...current});
      }
    });
    cleanupFunctions.push(messageCleanup);

    cleanupFunctionsRef.current = cleanupFunctions;

    return () => {
      cleanupFunctions.forEach(cleanup => cleanup());
    };
  }, [onNewMessage]);

  // WebSocket message handling
  useEffect(() => {
    if (!session.isConnected) return;

    const cleanup = session.addEventListener('assistant_response', (data) => {
      conversationService.handleWebSocketMessage({
        type: 'assistant_response',
        data
      });
    });

    const messageCleanup = session.addEventListener('message_received', (data) => {
      conversationService.handleWebSocketMessage({
        type: 'message_received',
        data
      });
    });

    return () => {
      cleanup();
      messageCleanup();
    };
  }, [session.isConnected]);

  // Auto-create conversation when session is established
  useEffect(() => {
    if (session.isConnected && session.userId && autoCreateConversation) {
      const current = conversationService.getCurrentConversation();
      
      if (!current) {
        const newConversation = conversationService.createConversation(
          defaultConversationName,
          session.userId,
          session.sessionId || undefined
        );
        setCurrentConversation(newConversation);
      }
    }
  }, [session.isConnected, session.userId, session.sessionId, autoCreateConversation, defaultConversationName]);

  // Conversation management actions
  const createConversation = useCallback((name: string): Conversation | null => {
    try {
      if (!session.userId) {
        setLastError('User ID required to create conversation');
        return null;
      }

      const conversation = conversationService.createConversation(
        name,
        session.userId,
        session.sessionId || undefined
      );
      
      setCurrentConversation(conversation);
      setLastError(null);
      return conversation;
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to create conversation';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return null;
    }
  }, [session.userId, session.sessionId, onError]);

  const switchToConversation = useCallback((id: string): boolean => {
    try {
      const success = conversationService.setCurrentConversation(id);
      
      if (success) {
        const conversation = conversationService.getConversation(id);
        setCurrentConversation(conversation);
        setLastError(null);
      } else {
        setLastError('Conversation not found');
      }
      
      return success;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to switch conversation';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return false;
    }
  }, [onError]);

  const deleteConversation = useCallback((id: string): boolean => {
    try {
      const success = conversationService.deleteConversation(id);
      
      if (success) {
        // Update current conversation after deletion
        const current = conversationService.getCurrentConversation();
        setCurrentConversation(current);
        setLastError(null);
      }
      
      return success;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to delete conversation';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return false;
    }
  }, [onError]);

  const renameConversation = useCallback((id: string, newName: string): boolean => {
    try {
      const success = conversationService.renameConversation(id, newName);
      
      if (success) {
        // Update current conversation if it was renamed
        if (currentConversation?.id === id) {
          const updated = conversationService.getConversation(id);
          setCurrentConversation(updated);
        }
        setLastError(null);
      }
      
      return success;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to rename conversation';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return false;
    }
  }, [currentConversation?.id, onError]);

  // Message management actions
  const sendMessage = useCallback(async (content: string): Promise<void> => {
    try {
      if (!currentConversation) {
        throw new Error('No active conversation');
      }

      if (!session.userId || !session.isConnected) {
        throw new Error('Session not connected');
      }

      setIsSending(true);
      setLastError(null);

      // Send via conversation service which handles API calls
      await conversationService.sendMessageToAPI(
        currentConversation.id,
        content,
        session.userId,
        session.sessionId || undefined
      );

      // The response will be handled via WebSocket events
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send message';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    } finally {
      setIsSending(false);
    }
  }, [currentConversation, session.userId, session.sessionId, session.isConnected, onError]);

  const queryLLM = useCallback(async (query: string, enableEvaluation: boolean = false): Promise<void> => {
    try {
      if (!currentConversation) {
        throw new Error('No active conversation');
      }

      if (!session.userId || !session.isConnected) {
        throw new Error('Session not connected');
      }

      setIsSending(true);
      setLastError(null);

      // Send via conversation service
      await conversationService.queryLLM(
        currentConversation.id,
        query,
        session.userId,
        session.sessionId || undefined,
        enableEvaluation
      );

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to query LLM';
      setLastError(errorMessage);
      onError?.(errorMessage);
      throw error;
    } finally {
      setIsSending(false);
    }
  }, [currentConversation, session.userId, session.sessionId, session.isConnected, onError]);

  const addMessage = useCallback((
    content: string, 
    role: 'user' | 'assistant' | 'system', 
    metadata?: any
  ): Message | null => {
    try {
      if (!currentConversation) {
        setLastError('No active conversation');
        return null;
      }

      const message = conversationService.addMessage(
        currentConversation.id,
        content,
        role,
        metadata
      );
      
      setLastError(null);
      return message;
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to add message';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return null;
    }
  }, [currentConversation, onError]);

  // Utility actions
  const clearError = useCallback(() => {
    setLastError(null);
  }, []);

  const clearAllConversations = useCallback(() => {
    try {
      conversationService.clearAllConversations();
      setCurrentConversation(null);
      setLastError(null);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to clear conversations';
      setLastError(errorMessage);
      onError?.(errorMessage);
    }
  }, [onError]);

  const exportConversations = useCallback((): string => {
    try {
      return conversationService.exportConversations();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to export conversations';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return '';
    }
  }, [onError]);

  const importConversations = useCallback((jsonData: string): boolean => {
    try {
      const success = conversationService.importConversations(jsonData);
      
      if (success) {
        const current = conversationService.getCurrentConversation();
        setCurrentConversation(current);
        setLastError(null);
      } else {
        setLastError('Failed to import conversations');
      }
      
      return success;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to import conversations';
      setLastError(errorMessage);
      onError?.(errorMessage);
      return false;
    }
  }, [onError]);

  return {
    // State
    currentConversation,
    messages,
    conversations,
    conversationSummaries,
    isTyping,
    isSending,
    lastError,
    totalConversations,
    totalMessages,
    
    // Actions
    createConversation,
    switchToConversation,
    deleteConversation,
    renameConversation,
    sendMessage,
    queryLLM,
    addMessage,
    setIsTyping,
    clearError,
    clearAllConversations,
    exportConversations,
    importConversations
  };
}