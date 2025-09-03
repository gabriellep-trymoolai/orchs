/**
 * WebSocket Service for MoolAI Orchestrator
 * Integrates with backend session management and provides real-time communication
 */

export interface WebSocketConfig {
  baseUrl?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  keepaliveInterval?: number;
  authTimeout?: number;
  debug?: boolean;
}

export interface SessionData {
  session_id: string;
  user_id: string;
  created_at: string;
  organization_id?: string;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
  message_id?: string;
  correlation_id?: string;
  timestamp?: string;
}

export interface ChatMessage {
  conversation_id: string;
  message: string;
  metadata?: Record<string, any>;
}

export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting' | 'error';

export class MoolAIWebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private connectionState: ConnectionState = 'disconnected';
  private sessionData: SessionData | null = null;
  private reconnectAttempts = 0;
  private messageIdCounter = 0;
  private keepaliveInterval: NodeJS.Timeout | null = null;
  private pendingResponses = new Map<string, {
    resolve: (value: any) => void;
    reject: (error: Error) => void;
    timeout: NodeJS.Timeout;
  }>();
  
  // Event listeners
  private listeners = new Map<string, Set<(data: any) => void>>();
  private stateListeners = new Set<(state: ConnectionState) => void>();
  private sessionListeners = new Set<(session: SessionData | null) => void>();

  constructor(config: WebSocketConfig = {}) {
    this.config = {
      baseUrl: config.baseUrl || this.getDefaultBaseUrl(),
      reconnectInterval: config.reconnectInterval || 5000,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      keepaliveInterval: config.keepaliveInterval || 30000,
      authTimeout: config.authTimeout || 10000,
      debug: config.debug || false
    };

    this.log('WebSocket service initialized', this.config);
  }

  private getDefaultBaseUrl(): string {
    // Use environment variable first
    const envUrl = import.meta.env.VITE_WS_BASE_URL;
    if (envUrl) {
      return envUrl;
    }
    
    // Fallback to dynamic URL based on current location
    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${protocol}//${window.location.host}`;
    }
    return 'ws://localhost:8000';
  }

  private log(message: string, ...args: any[]): void {
    if (this.config.debug) {
      console.log(`[MoolAI WS] ${message}`, ...args);
    }
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${++this.messageIdCounter}`;
  }

  private setState(state: ConnectionState): void {
    if (this.connectionState !== state) {
      this.connectionState = state;
      this.log(`State changed to: ${state}`);
      this.stateListeners.forEach(listener => listener(state));
    }
  }

  private setSession(session: SessionData | null): void {
    this.sessionData = session;
    this.log('Session updated:', session);
    this.sessionListeners.forEach(listener => listener(session));
  }

  private emit(eventType: string, data: any): void {
    const eventListeners = this.listeners.get(eventType);
    if (eventListeners) {
      eventListeners.forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`[MoolAI WS] Error in listener for ${eventType}:`, error);
        }
      });
    }
  }

  // Public API
  public getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  public getSession(): SessionData | null {
    return this.sessionData;
  }

  public addEventListener(eventType: string, callback: (data: any) => void): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(callback);

    // Return cleanup function
    return () => {
      const eventListeners = this.listeners.get(eventType);
      if (eventListeners) {
        eventListeners.delete(callback);
      }
    };
  }

  public onStateChange(callback: (state: ConnectionState) => void): () => void {
    this.stateListeners.add(callback);
    return () => this.stateListeners.delete(callback);
  }

  public onSessionChange(callback: (session: SessionData | null) => void): () => void {
    this.sessionListeners.add(callback);
    return () => this.sessionListeners.delete(callback);
  }

  public async connect(params: {
    user_id?: string;
    session_id?: string;
    token?: string;
  } = {}): Promise<SessionData> {
    if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
      throw new Error('Connection already in progress or established');
    }

    this.setState('connecting');
    this.log('Connecting to WebSocket...', params);

    try {
      const wsEndpoint = import.meta.env.VITE_WS_ENDPOINT || '/ws/v1/session';
      const url = new URL(wsEndpoint, this.config.baseUrl);
      
      // Add query parameters
      if (params.user_id) url.searchParams.set('user_id', params.user_id);
      if (params.session_id) url.searchParams.set('session_id', params.session_id);
      if (params.token) url.searchParams.set('token', params.token);

      this.ws = new WebSocket(url.toString());
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
          this.cleanup();
        }, this.config.authTimeout);

        this.ws!.onopen = () => {
          this.log('WebSocket connection opened');
          this.reconnectAttempts = 0;
        };

        this.ws!.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);

            // Handle session establishment
            if (message.type === 'session_established') {
              clearTimeout(timeout);
              this.setState('connected');
              this.setSession(message.data as SessionData);
              this.startKeepalive();
              resolve(message.data as SessionData);
            }
          } catch (error) {
            console.error('[MoolAI WS] Error parsing message:', error);
          }
        };

        this.ws!.onclose = (event) => {
          clearTimeout(timeout);
          this.log(`WebSocket closed: ${event.code} ${event.reason}`);
          this.handleDisconnection();
        };

        this.ws!.onerror = (error) => {
          clearTimeout(timeout);
          this.log('WebSocket error:', error);
          this.setState('error');
          reject(new Error('WebSocket connection failed'));
        };
      });
    } catch (error) {
      this.setState('error');
      throw error;
    }
  }

  public disconnect(): void {
    this.log('Disconnecting WebSocket');
    
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval);
      this.keepaliveInterval = null;
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      // Send disconnect message
      this.send({
        type: 'disconnect',
        session_id: this.sessionData?.session_id
      });
      
      this.ws.close(1000, 'Client disconnect');
    }

    this.cleanup();
  }

  public send(message: Partial<WebSocketMessage>): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }

    const fullMessage: WebSocketMessage = {
      message_id: this.generateMessageId(),
      timestamp: new Date().toISOString(),
      ...message
    };

    this.log('Sending message:', fullMessage);
    this.ws.send(JSON.stringify(fullMessage));
  }

  public async sendAndWait<T = any>(message: Partial<WebSocketMessage>, timeout = 30000): Promise<T> {
    const messageId = this.generateMessageId();
    const fullMessage = {
      message_id: messageId,
      timestamp: new Date().toISOString(),
      ...message
    };

    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pendingResponses.delete(messageId);
        reject(new Error('Response timeout'));
      }, timeout);

      this.pendingResponses.set(messageId, {
        resolve: (data) => {
          clearTimeout(timeoutId);
          this.pendingResponses.delete(messageId);
          resolve(data);
        },
        reject: (error) => {
          clearTimeout(timeoutId);
          this.pendingResponses.delete(messageId);
          reject(error);
        },
        timeout: timeoutId
      });

      this.send(fullMessage);
    });
  }

  // Chat-specific methods
  public async sendChatMessage(chatMessage: ChatMessage): Promise<void> {
    return this.sendAndWait({
      type: 'send_message',
      ...chatMessage
    });
  }

  // Analytics-specific methods
  public async requestAnalytics(params: {
    start_date?: string;
    end_date?: string;
  } = {}): Promise<any> {
    return this.sendAndWait({
      type: 'analytics_request',
      data: params
    });
  }

  public async subscribeToAnalytics(): Promise<void> {
    return this.sendAndWait({
      type: 'analytics_subscribe'
    });
  }

  public async unsubscribeFromAnalytics(): Promise<void> {
    return this.sendAndWait({
      type: 'analytics_unsubscribe'
    });
  }

  public async joinConversation(conversationId: string): Promise<void> {
    return this.sendAndWait({
      type: 'join_conversation',
      conversation_id: conversationId
    });
  }

  public sendHeartbeat(): void {
    if (this.connectionState === 'connected') {
      this.send({
        type: 'heartbeat'
      });
    }
  }

  // Private methods
  private handleMessage(message: WebSocketMessage): void {
    this.log('Received message:', message);

    // Handle responses to pending requests
    if (message.correlation_id && this.pendingResponses.has(message.correlation_id)) {
      const handler = this.pendingResponses.get(message.correlation_id)!;
      if (message.type === 'error') {
        handler.reject(new Error(message.data?.error || 'Unknown error'));
      } else {
        handler.resolve(message.data);
      }
      return;
    }

    // Handle specific message types
    switch (message.type) {
      case 'session_established':
        // Handled in connect method
        break;
      
      case 'message_received':
        this.emit('message_received', message.data);
        break;
      
      case 'assistant_response':
        this.emit('assistant_response', message.data);
        break;
      
      case 'analytics_response':
        this.emit('analytics_response', message.data);
        break;
      
      case 'analytics_error':
        this.emit('analytics_error', message.data);
        break;
      
      case 'analytics_subscription_confirmed':
        this.emit('analytics_subscription_confirmed', message.data);
        break;
      
      case 'analytics_unsubscribed':
        this.emit('analytics_unsubscribed', message.data);
        break;
      
      case 'heartbeat_ack':
        this.emit('heartbeat_ack', message.data);
        break;
      
      case 'disconnected':
        this.emit('disconnected', message.data);
        this.handleDisconnection();
        break;
      
      case 'error':
        this.emit('error', message.data);
        break;
      
      default:
        this.emit(message.type, message.data);
    }
  }

  private startKeepalive(): void {
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval);
    }

    this.keepaliveInterval = setInterval(() => {
      try {
        this.sendHeartbeat();
      } catch (error) {
        this.log('Keepalive failed:', error);
      }
    }, this.config.keepaliveInterval);
  }

  private handleDisconnection(): void {
    this.cleanup();
    
    if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
      this.scheduleReconnect();
    } else {
      this.log('Max reconnection attempts reached');
      this.setState('error');
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnectInterval * Math.pow(2, Math.min(this.reconnectAttempts - 1, 4)),
      30000
    );
    
    this.log(`Scheduling reconnection attempt ${this.reconnectAttempts} in ${delay}ms`);
    this.setState('reconnecting');
    
    setTimeout(() => {
      if (this.connectionState === 'reconnecting') {
        this.connect().catch(error => {
          this.log('Reconnection failed:', error);
          this.handleDisconnection();
        });
      }
    }, delay);
  }

  private cleanup(): void {
    if (this.keepaliveInterval) {
      clearInterval(this.keepaliveInterval);
      this.keepaliveInterval = null;
    }

    // Clear pending responses
    this.pendingResponses.forEach(handler => {
      clearTimeout(handler.timeout);
      handler.reject(new Error('Connection closed'));
    });
    this.pendingResponses.clear();

    this.ws = null;
    this.setState('disconnected');
    this.setSession(null);
  }
}

// Export singleton instance
export const webSocketService = new MoolAIWebSocketService({
  debug: true // Temporarily enable debug logging
});