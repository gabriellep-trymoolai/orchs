/**
 * WebSocket Test Page
 * For testing the integrated session and conversation management
 */

import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { AlertCircle, Send, Trash2, Plus } from 'lucide-react';
import { useAppSession, useAppConversation } from '@/contexts/AppContext';
import { ConnectionStatus } from '@/components/ConnectionStatus';

export const WebSocketTest: React.FC = () => {
  const session = useAppSession();
  const conversation = useAppConversation();
  const [testMessage, setTestMessage] = useState('');
  const [testQuery, setTestQuery] = useState('');
  const [rawMessage, setRawMessage] = useState('');

  const handleSendTestMessage = async () => {
    if (!testMessage.trim() || !conversation.currentConversation) return;
    
    try {
      await conversation.sendMessage(testMessage);
      setTestMessage('');
    } catch (error) {
      console.error('Error sending test message:', error);
    }
  };

  const handleTestQuery = async () => {
    if (!testQuery.trim() || !conversation.currentConversation) return;
    
    try {
      await conversation.queryLLM(testQuery, true);
      setTestQuery('');
    } catch (error) {
      console.error('Error sending test query:', error);
    }
  };

  const handleSendRawMessage = () => {
    if (!rawMessage.trim()) return;
    
    try {
      const message = JSON.parse(rawMessage);
      session.sendRawMessage(message);
      setRawMessage('');
    } catch (error) {
      console.error('Error sending raw message:', error);
    }
  };

  const handleCreateTestConversation = () => {
    const name = `Test Chat ${conversation.totalConversations + 1}`;
    conversation.createConversation(name);
  };

  const handleAddTestMessage = (role: 'user' | 'assistant' | 'system', content: string) => {
    if (!conversation.currentConversation) return;
    
    conversation.addMessage(content, role, {
      timestamp: new Date().toISOString(),
      test_message: true
    });
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">WebSocket Integration Test</h1>
        <p className="text-muted-foreground">
          Test the MoolAI WebSocket session and conversation management
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Connection Status */}
        <div className="lg:col-span-1">
          <ConnectionStatus />
        </div>

        {/* Main Test Area */}
        <div className="lg:col-span-2">
          <Tabs defaultValue="conversation" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="conversation">Conversation</TabsTrigger>
              <TabsTrigger value="messages">Messages</TabsTrigger>
              <TabsTrigger value="raw">Raw WebSocket</TabsTrigger>
              <TabsTrigger value="debug">Debug</TabsTrigger>
            </TabsList>

            {/* Conversation Tab */}
            <TabsContent value="conversation" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Conversation Management</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex gap-2">
                    <Button 
                      onClick={handleCreateTestConversation}
                      disabled={!session.isConnected}
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Create Test Conversation
                    </Button>
                    <Button 
                      variant="outline"
                      onClick={() => conversation.clearAllConversations()}
                    >
                      <Trash2 className="w-4 h-4 mr-2" />
                      Clear All
                    </Button>
                  </div>

                  {conversation.currentConversation && (
                    <div className="p-3 bg-muted rounded">
                      <h4 className="font-medium">{conversation.currentConversation.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {conversation.messages.length} messages
                      </p>
                    </div>
                  )}

                  <div className="space-y-2">
                    <h4 className="font-medium">All Conversations:</h4>
                    {conversation.conversationSummaries.map((conv) => (
                      <div key={conv.id} className="flex items-center justify-between p-2 border rounded">
                        <div>
                          <span className="font-medium">{conv.name}</span>
                          <Badge variant="secondary" className="ml-2">
                            {conv.message_count}
                          </Badge>
                        </div>
                        <div className="flex gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => conversation.switchToConversation(conv.id)}
                          >
                            Switch
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => conversation.deleteConversation(conv.id)}
                          >
                            <Trash2 className="w-3 h-3" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Messages Tab */}
            <TabsContent value="messages" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Message Testing</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Send Chat Message:</label>
                    <div className="flex gap-2">
                      <Input
                        value={testMessage}
                        onChange={(e) => setTestMessage(e.target.value)}
                        placeholder="Type a test message..."
                        disabled={!session.isConnected || !conversation.currentConversation}
                      />
                      <Button 
                        onClick={handleSendTestMessage}
                        disabled={!testMessage.trim() || !session.isConnected || !conversation.currentConversation}
                      >
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">LLM Query (with evaluation):</label>
                    <div className="flex gap-2">
                      <Input
                        value={testQuery}
                        onChange={(e) => setTestQuery(e.target.value)}
                        placeholder="Type a test query..."
                        disabled={!session.isConnected || !conversation.currentConversation}
                      />
                      <Button 
                        onClick={handleTestQuery}
                        disabled={!testQuery.trim() || !session.isConnected || !conversation.currentConversation}
                      >
                        Query
                      </Button>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Add Test Messages:</label>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        onClick={() => handleAddTestMessage('user', 'Hello, this is a test user message!')}
                        disabled={!conversation.currentConversation}
                      >
                        Add User Message
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handleAddTestMessage('assistant', 'Hello! This is a test assistant response.')}
                        disabled={!conversation.currentConversation}
                      >
                        Add Assistant Message
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handleAddTestMessage('system', 'System message: Test completed successfully.')}
                        disabled={!conversation.currentConversation}
                      >
                        Add System Message
                      </Button>
                    </div>
                  </div>

                  {/* Message Display */}
                  <div className="border rounded p-3 max-h-60 overflow-y-auto">
                    <h4 className="font-medium mb-2">Current Conversation Messages:</h4>
                    {conversation.messages.length === 0 ? (
                      <p className="text-muted-foreground text-sm">No messages yet</p>
                    ) : (
                      <div className="space-y-2">
                        {conversation.messages.map((msg) => (
                          <div key={msg.id} className="p-2 bg-muted rounded text-sm">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant={msg.role === 'user' ? 'default' : msg.role === 'assistant' ? 'secondary' : 'outline'}>
                                {msg.role}
                              </Badge>
                              <span className="text-xs text-muted-foreground">
                                {new Date(msg.timestamp).toLocaleTimeString()}
                              </span>
                            </div>
                            <p>{msg.content}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Raw WebSocket Tab */}
            <TabsContent value="raw" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Raw WebSocket Testing</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Send Raw Message (JSON):</label>
                    <textarea
                      value={rawMessage}
                      onChange={(e) => setRawMessage(e.target.value)}
                      placeholder='{"type": "heartbeat"}'
                      className="w-full h-24 p-2 border rounded text-sm font-mono"
                      disabled={!session.isConnected}
                    />
                    <Button 
                      onClick={handleSendRawMessage}
                      disabled={!rawMessage.trim() || !session.isConnected}
                    >
                      Send Raw Message
                    </Button>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Quick Raw Messages:</label>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setRawMessage('{"type": "heartbeat"}')}
                      >
                        Heartbeat
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setRawMessage('{"type": "ping", "data": {"timestamp": "' + new Date().toISOString() + '"}}')}
                      >
                        Ping
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setRawMessage('{"type": "test", "data": {"message": "WebSocket test"}}')}
                      >
                        Test Message
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            {/* Debug Tab */}
            <TabsContent value="debug" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Debug Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-2">Session State:</h4>
                      <pre className="text-xs bg-muted p-2 rounded overflow-auto">
                        {JSON.stringify({
                          connectionState: session.connectionState,
                          isConnected: session.isConnected,
                          sessionId: session.sessionId,
                          userId: session.userId,
                          reconnectAttempts: session.reconnectAttempts,
                          lastError: session.lastError
                        }, null, 2)}
                      </pre>
                    </div>
                    <div>
                      <h4 className="font-medium mb-2">Conversation State:</h4>
                      <pre className="text-xs bg-muted p-2 rounded overflow-auto">
                        {JSON.stringify({
                          currentConversationId: conversation.currentConversation?.id,
                          totalConversations: conversation.totalConversations,
                          totalMessages: conversation.totalMessages,
                          isSending: conversation.isSending,
                          lastError: conversation.lastError
                        }, null, 2)}
                      </pre>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Current Conversation Data:</h4>
                    <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-40">
                      {JSON.stringify(conversation.currentConversation, null, 2)}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};