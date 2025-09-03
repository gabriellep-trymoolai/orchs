/**
 * Conversation Service for MoolAI Orchestrator
 * Manages chat conversations with localStorage persistence and real-time updates
 */

import { apiClient } from './api-client';

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant' | 'system';
  timestamp: string;
  conversation_id: string;
  metadata?: {
    model?: string;
    tokens?: number;
    cached?: boolean;
    sequence_number?: number;
    is_complete?: boolean;
    [key: string]: any;
  };
}

export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  created_at: string;
  updated_at: string;
  user_id: string;
  session_id?: string;
  metadata?: {
    model?: string;
    message_count?: number;
    [key: string]: any;
  };
}

export interface ConversationSummary {
  id: string;
  name: string;
  message_count: number;
  last_message: string;
  updated_at: string;
}

export class ConversationService {
  private conversations: Map<string, Conversation> = new Map();
  private currentConversation: Conversation | null = null;
  private listeners: Set<(conversations: Conversation[]) => void> = new Set();
  private messageListeners: Set<(message: Message) => void> = new Set();
  
  private readonly STORAGE_KEY = 'moolai_conversations';
  private readonly CURRENT_KEY = 'moolai_current_conversation';

  constructor() {
    this.loadFromStorage();
  }

  // Storage management
  private saveToStorage(): void {
    try {
      const conversationsArray = Array.from(this.conversations.values());
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(conversationsArray));
      
      if (this.currentConversation) {
        localStorage.setItem(this.CURRENT_KEY, this.currentConversation.id);
      } else {
        localStorage.removeItem(this.CURRENT_KEY);
      }
    } catch (error) {
      console.warn('Failed to save conversations to localStorage:', error);
    }
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const conversationsArray: Conversation[] = JSON.parse(stored);
        this.conversations.clear();
        
        conversationsArray.forEach(conv => {
          this.conversations.set(conv.id, conv);
        });
      }

      const currentId = localStorage.getItem(this.CURRENT_KEY);
      if (currentId && this.conversations.has(currentId)) {
        this.currentConversation = this.conversations.get(currentId)!;
      }
      
      this.notifyListeners();
    } catch (error) {
      console.warn('Failed to load conversations from localStorage:', error);
    }
  }

  // Event management
  public onConversationsChange(callback: (conversations: Conversation[]) => void): () => void {
    this.listeners.add(callback);
    // Immediately call with current data
    callback(this.getAllConversations());
    
    return () => {
      this.listeners.delete(callback);
    };
  }

  public onNewMessage(callback: (message: Message) => void): () => void {
    this.messageListeners.add(callback);
    return () => {
      this.messageListeners.delete(callback);
    };
  }

  private notifyListeners(): void {
    const conversations = this.getAllConversations();
    this.listeners.forEach(callback => {
      try {
        callback(conversations);
      } catch (error) {
        console.error('Error in conversation listener:', error);
      }
    });
  }

  private notifyMessageListeners(message: Message): void {
    this.messageListeners.forEach(callback => {
      try {
        callback(message);
      } catch (error) {
        console.error('Error in message listener:', error);
      }
    });
  }

  // Conversation management
  public createConversation(
    name: string, 
    userId: string, 
    sessionId?: string
  ): Conversation {
    const conversation: Conversation = {
      id: `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name,
      messages: [],
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      user_id: userId,
      session_id: sessionId,
      metadata: {
        message_count: 0
      }
    };

    this.conversations.set(conversation.id, conversation);
    this.currentConversation = conversation;
    this.saveToStorage();
    this.notifyListeners();

    return conversation;
  }

  public getConversation(id: string): Conversation | null {
    return this.conversations.get(id) || null;
  }

  public getCurrentConversation(): Conversation | null {
    return this.currentConversation;
  }

  public setCurrentConversation(id: string): boolean {
    const conversation = this.conversations.get(id);
    if (conversation) {
      this.currentConversation = conversation;
      this.saveToStorage();
      this.notifyListeners();
      return true;
    }
    return false;
  }

  public getAllConversations(): Conversation[] {
    return Array.from(this.conversations.values())
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());
  }

  public getConversationSummaries(): ConversationSummary[] {
    return this.getAllConversations().map(conv => ({
      id: conv.id,
      name: conv.name,
      message_count: conv.messages.length,
      last_message: conv.messages.length > 0 
        ? conv.messages[conv.messages.length - 1].content.slice(0, 100) + '...'
        : 'No messages',
      updated_at: conv.updated_at
    }));
  }

  public deleteConversation(id: string): boolean {
    const deleted = this.conversations.delete(id);
    
    if (deleted) {
      if (this.currentConversation?.id === id) {
        // Set current to most recent conversation or null
        const remaining = this.getAllConversations();
        this.currentConversation = remaining.length > 0 ? remaining[0] : null;
      }
      
      this.saveToStorage();
      this.notifyListeners();
    }
    
    return deleted;
  }

  public renameConversation(id: string, newName: string): boolean {
    const conversation = this.conversations.get(id);
    if (conversation) {
      conversation.name = newName;
      conversation.updated_at = new Date().toISOString();
      
      this.saveToStorage();
      this.notifyListeners();
      return true;
    }
    return false;
  }

  // Message management
  public addMessage(
    conversationId: string,
    content: string,
    role: 'user' | 'assistant' | 'system',
    metadata?: Message['metadata']
  ): Message | null {
    const conversation = this.conversations.get(conversationId);
    if (!conversation) {
      return null;
    }

    const message: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      content,
      role,
      timestamp: new Date().toISOString(),
      conversation_id: conversationId,
      metadata
    };

    conversation.messages.push(message);
    conversation.updated_at = new Date().toISOString();
    conversation.metadata = {
      ...conversation.metadata,
      message_count: conversation.messages.length
    };

    this.saveToStorage();
    this.notifyListeners();
    this.notifyMessageListeners(message);

    return message;
  }

  public addUserMessage(conversationId: string, content: string): Message | null {
    return this.addMessage(conversationId, content, 'user');
  }

  public addAssistantMessage(
    conversationId: string, 
    content: string, 
    metadata?: Message['metadata']
  ): Message | null {
    return this.addMessage(conversationId, content, 'assistant', metadata);
  }

  public updateMessage(messageId: string, updates: Partial<Message>): boolean {
    for (const conversation of this.conversations.values()) {
      const messageIndex = conversation.messages.findIndex(m => m.id === messageId);
      if (messageIndex !== -1) {
        conversation.messages[messageIndex] = {
          ...conversation.messages[messageIndex],
          ...updates
        };
        conversation.updated_at = new Date().toISOString();
        
        this.saveToStorage();
        this.notifyListeners();
        return true;
      }
    }
    return false;
  }

  public getConversationMessages(conversationId: string): Message[] {
    const conversation = this.conversations.get(conversationId);
    return conversation ? [...conversation.messages] : [];
  }

  // API integration methods
  public async sendMessageToAPI(
    conversationId: string,
    message: string,
    userId: string,
    sessionId?: string
  ): Promise<any> {
    try {
      // Add user message locally first
      this.addUserMessage(conversationId, message);

      // Send to backend
      const response = await apiClient.processPrompt({
        prompt: message,
        user_id: userId,
        session_id: sessionId,
        metadata: {
          conversation_id: conversationId,
          timestamp: new Date().toISOString()
        }
      });

      // Process the API response and add assistant message
      if (response && response.success && response.data) {
        const assistantContent = response.data.response || response.data.content || response.data.text || 'Response received';
        this.addAssistantMessage(conversationId, assistantContent, {
          model: response.data.model || 'unknown',
          timestamp: new Date().toISOString(),
          api_response: true
        });
      }

      return response;
    } catch (error) {
      console.error('Failed to send message to API:', error);
      throw error;
    }
  }

  public async queryLLM(
    conversationId: string,
    query: string,
    userId: string,
    sessionId?: string,
    enableEvaluation: boolean = false
  ): Promise<any> {
    try {
      // Add user message locally first
      this.addUserMessage(conversationId, query);

      // Send to LLM endpoint
      const response = await apiClient.queryLLM({
        query,
        user_id: userId,
        session_id: sessionId,
        enable_evaluation: enableEvaluation
      });

      return response;
    } catch (error) {
      console.error('Failed to query LLM:', error);
      throw error;
    }
  }

  public handleWebSocketMessage(message: any): void {
    try {
      switch (message.type) {
        case 'message_received':
          if (message.data?.conversation_id && message.data?.content) {
            this.addMessage(
              message.data.conversation_id,
              message.data.content,
              message.data.role || 'system',
              message.data.metadata
            );
          }
          break;

        case 'assistant_response':
          if (message.data?.conversation_id && message.data?.content_delta) {
            // Handle streaming responses
            const conversationId = message.data.conversation_id;
            const content = message.data.content_delta;
            const isComplete = message.data.is_complete || false;
            
            if (isComplete) {
              this.addAssistantMessage(conversationId, content, {
                model: message.data.metadata?.model,
                sequence_number: message.data.sequence_number,
                is_complete: true
              });
            }
          }
          break;

        default:
          // Handle other message types as needed
          break;
      }
    } catch (error) {
      console.error('Error handling WebSocket message:', error);
    }
  }

  // Utility methods
  public clearAllConversations(): void {
    this.conversations.clear();
    this.currentConversation = null;
    
    try {
      localStorage.removeItem(this.STORAGE_KEY);
      localStorage.removeItem(this.CURRENT_KEY);
    } catch (error) {
      console.warn('Failed to clear localStorage:', error);
    }
    
    this.notifyListeners();
  }

  public exportConversations(): string {
    const conversations = this.getAllConversations();
    return JSON.stringify(conversations, null, 2);
  }

  public importConversations(jsonData: string): boolean {
    try {
      const conversations: Conversation[] = JSON.parse(jsonData);
      
      // Validate structure
      if (!Array.isArray(conversations)) {
        throw new Error('Invalid format: expected array');
      }

      // Clear current data and import
      this.conversations.clear();
      conversations.forEach(conv => {
        if (conv.id && conv.name && Array.isArray(conv.messages)) {
          this.conversations.set(conv.id, conv);
        }
      });

      this.currentConversation = null;
      this.saveToStorage();
      this.notifyListeners();
      
      return true;
    } catch (error) {
      console.error('Failed to import conversations:', error);
      return false;
    }
  }
}

// Export singleton instance
export const conversationService = new ConversationService();