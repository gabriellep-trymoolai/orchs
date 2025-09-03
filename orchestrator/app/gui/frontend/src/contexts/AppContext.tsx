/**
 * App Context Provider for MoolAI Orchestrator
 * Provides session and conversation management across the application
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { useSession, SessionState, SessionActions } from '../hooks/useSession';
import { useConversation, ConversationState, ConversationActions } from '../hooks/useConversation';

interface AppContextType {
  session: SessionState & SessionActions;
  conversation: ConversationState & ConversationActions;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  // Initialize session with auto-connect
  const session = useSession({
    autoConnect: true,
    autoReconnect: true,
    userId: localStorage.getItem('moolai_user_id') || undefined,
    onSessionEstablished: (sessionData) => {
      console.log('Session established:', sessionData);
      // Store session info in localStorage for persistence
      localStorage.setItem('moolai_session_id', sessionData.session_id);
      localStorage.setItem('moolai_user_id', sessionData.user_id);
    },
    onSessionLost: () => {
      console.log('Session lost');
    },
    onError: (error) => {
      console.error('Session error:', error);
    }
  });

  // Initialize conversation management
  const conversation = useConversation({
    session, // Pass the single session instance
    autoCreateConversation: true,
    defaultConversationName: 'New Chat',
    enableAutoResponse: true,
    onNewMessage: (message) => {
      console.log('New message:', message);
    },
    onError: (error) => {
      console.error('Conversation error:', error);
    }
  });

  const contextValue: AppContextType = {
    session,
    conversation
  };

  return (
    <AppContext.Provider value={contextValue}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

// Individual hook exports for convenience
export function useAppSession() {
  const { session } = useApp();
  return session;
}

export function useAppConversation() {
  const { conversation } = useApp();
  return conversation;
}