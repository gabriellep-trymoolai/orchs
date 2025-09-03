import React, { useState, useRef, useEffect } from 'react';
import { ChatSidebar } from '@/components/ChatSidebar';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Menu, Plus, Send, Minimize2, Maximize2, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';
import { useAppSession, useAppConversation } from '@/contexts/AppContext';
import { Message } from '@/services/conversation';

// Message Component
const MessageBubble: React.FC<{ message: Message; isUser: boolean }> = ({ message, isUser }) => {
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[70%] ${isUser ? 'order-2' : 'order-1'}`}>
        {!isUser && (
          <div className="flex items-center mb-2">
            <div className="w-8 h-8 bg-orange-primary rounded-full flex items-center justify-center mr-2">
              <span className="text-sm font-medium text-white">AI</span>
            </div>
            <span className="text-sm text-muted-foreground">Assistant</span>
          </div>
        )}
        <div
          className={`p-3 rounded-lg ${
            isUser
              ? 'bg-orange-primary text-white'
              : 'bg-muted text-foreground'
          }`}
        >
          <p className="whitespace-pre-wrap">{message.content}</p>
          <div className={`flex items-center justify-between mt-2 text-xs ${
            isUser ? 'text-orange-100' : 'text-muted-foreground'
          }`}>
            <span>{formatTime(message.timestamp)}</span>
            {message.metadata?.cached && (
              <Badge variant="secondary" className="text-xs">Cached</Badge>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export const Dashboard: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Hooks for session and conversation management
  const session = useAppSession();
  const conversation = useAppConversation();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation.messages]);

  // Auto-create conversation if needed
  useEffect(() => {
    if (session.isConnected && session.userId && !conversation.currentConversation) {
      conversation.createConversation('New Chat');
    }
  }, [session.isConnected, session.userId, conversation.currentConversation]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && conversation.currentConversation && !conversation.isSending) {
      try {
        await conversation.sendMessage(message.trim());
        setMessage('');
      } catch (error) {
        console.error('Failed to send message:', error);
      }
    }
  };

  const handleCreateNewChat = () => {
    if (session.userId) {
      const newConversation = conversation.createConversation(
        `Chat ${conversation.totalConversations + 1}`
      );
      if (newConversation) {
        setMessage('');
      }
    }
  };

  const toggleSidebarCollapse = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  // Connection status indicator
  const getConnectionStatus = () => {
    if (session.connectionState === 'connected') {
      return <Badge variant="outline" className="text-green-600 border-green-600"><CheckCircle className="w-3 h-3 mr-1" />Connected</Badge>;
    } else if (session.connectionState === 'connecting' || session.connectionState === 'reconnecting') {
      return <Badge variant="outline" className="text-yellow-600 border-yellow-600"><Loader2 className="w-3 h-3 mr-1 animate-spin" />Connecting</Badge>;
    } else {
      return <Badge variant="outline" className="text-red-600 border-red-600"><AlertCircle className="w-3 h-3 mr-1" />Disconnected</Badge>;
    }
  };

  return (
    <div className="min-h-screen bg-background flex">
      <ChatSidebar 
        isOpen={sidebarOpen} 
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        isCollapsed={sidebarCollapsed}
      />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <header className="h-16 border-b border-border flex items-center px-6 bg-card">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="lg:hidden mr-4"
          >
            <Menu className="h-5 w-5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleSidebarCollapse}
            className="hidden lg:flex mr-4"
          >
            {sidebarCollapsed ? <Maximize2 className="h-5 w-5" /> : <Minimize2 className="h-5 w-5" />}
          </Button>
          
          {/* Conversation Info */}
          <div className="flex-1 flex items-center justify-center">
            {conversation.currentConversation && (
              <div className="text-center">
                <h2 className="text-sm font-medium text-foreground">
                  {conversation.currentConversation.name}
                </h2>
                <p className="text-xs text-muted-foreground">
                  {conversation.messages.length} messages
                </p>
              </div>
            )}
          </div>

          {/* Status indicators */}
          <div className="flex items-center gap-3">
            {getConnectionStatus()}
            <Button
              variant="outline"
              size="sm"
              onClick={handleCreateNewChat}
              disabled={!session.isConnected}
              className="hidden sm:flex"
            >
              <Plus className="h-4 w-4 mr-1" />
              New Chat
            </Button>
          </div>
        </header>

        {/* Error Display */}
        {(session.lastError || conversation.lastError) && (
          <div className="px-6 pt-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {session.lastError || conversation.lastError}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    session.clearError();
                    conversation.clearError();
                  }}
                  className="ml-2 h-auto p-1 text-xs"
                >
                  Dismiss
                </Button>
              </AlertDescription>
            </Alert>
          </div>
        )}

        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto">
            {conversation.messages.length === 0 ? (
              <div className="h-full flex items-center justify-center p-6">
                <div className="text-center max-w-md">
                  <div className="w-16 h-16 bg-orange-primary rounded-full flex items-center justify-center mx-auto mb-6">
                    <span className="text-2xl">👋</span>
                  </div>
                  <h1 className="text-2xl font-semibold text-foreground mb-4">
                    Hello, {session.userId ? session.userId.split('_').slice(-1)[0] : 'User'}
                  </h1>
                  <p className="text-muted-foreground mb-8">How can I help you today?</p>
                  {!session.isConnected && (
                    <Button
                      onClick={() => session.connect()}
                      disabled={session.isConnecting}
                      className="bg-orange-primary hover:bg-orange-dark text-white"
                    >
                      {session.isConnecting ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Connecting...
                        </>
                      ) : (
                        'Connect to Start Chatting'
                      )}
                    </Button>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-6">
                {conversation.messages.map((msg) => (
                  <MessageBubble
                    key={msg.id}
                    message={msg}
                    isUser={msg.role === 'user'}
                  />
                ))}
                {conversation.isSending && (
                  <div className="flex justify-start mb-4">
                    <div className="flex items-center space-x-2 bg-muted p-3 rounded-lg">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">Assistant is typing...</span>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-6 border-t border-border bg-card">
            <form onSubmit={handleSendMessage} className="flex gap-3 max-w-4xl mx-auto">
              <div className="flex-1 relative">
                <Input
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder={
                    !session.isConnected 
                      ? "Connect to start chatting..." 
                      : conversation.isTyping 
                        ? "Typing..." 
                        : "How can I help you today?"
                  }
                  disabled={!session.isConnected || conversation.isSending}
                  className="h-12 bg-input border-border text-foreground placeholder:text-muted-foreground pr-12"
                  onFocus={() => conversation.setIsTyping(true)}
                  onBlur={() => conversation.setIsTyping(false)}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={handleCreateNewChat}
                  disabled={!session.isConnected}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground sm:hidden"
                >
                  <Plus className="h-5 w-5" />
                </Button>
              </div>
              <Button
                type="submit"
                className="h-12 px-6 bg-orange-primary hover:bg-orange-dark text-white"
                disabled={!message.trim() || !session.isConnected || conversation.isSending}
              >
                {conversation.isSending ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </Button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};