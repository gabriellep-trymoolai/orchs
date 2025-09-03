/**
 * Connection Status Component
 * Shows real-time WebSocket connection status and session info
 */

import React, { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  Wifi, 
  WifiOff,
  RefreshCw,
  MessageSquare,
  User,
  Clock
} from 'lucide-react';
import { useAppSession, useAppConversation } from '@/contexts/AppContext';

export const ConnectionStatus: React.FC = () => {
  const session = useAppSession();
  const conversation = useAppConversation();
  const [lastHeartbeat, setLastHeartbeat] = useState<Date | null>(null);

  // Listen for heartbeat events
  useEffect(() => {
    if (!session.isConnected) return;

    const cleanup = session.addEventListener('heartbeat_ack', () => {
      setLastHeartbeat(new Date());
    });

    return cleanup;
  }, [session.isConnected]);

  const getConnectionStatusBadge = () => {
    switch (session.connectionState) {
      case 'connected':
        return (
          <Badge variant="outline" className="text-green-600 border-green-600">
            <CheckCircle className="w-3 h-3 mr-1" />
            Connected
          </Badge>
        );
      case 'connecting':
      case 'reconnecting':
        return (
          <Badge variant="outline" className="text-yellow-600 border-yellow-600">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            {session.connectionState === 'connecting' ? 'Connecting' : 'Reconnecting'}
          </Badge>
        );
      case 'error':
        return (
          <Badge variant="outline" className="text-red-600 border-red-600">
            <AlertCircle className="w-3 h-3 mr-1" />
            Error
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-gray-600 border-gray-600">
            <WifiOff className="w-3 h-3 mr-1" />
            Disconnected
          </Badge>
        );
    }
  };

  const formatTime = (date: Date | null) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString();
  };

  const formatDuration = (date: Date | null) => {
    if (!date) return 'N/A';
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Wifi className="w-4 h-4" />
          Connection Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Connection Status */}
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">Status:</span>
          {getConnectionStatusBadge()}
        </div>

        {/* Session Info */}
        {session.sessionData && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Session ID:</span>
              <span className="text-xs font-mono bg-muted px-2 py-1 rounded">
                {session.sessionData.session_id.slice(-8)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">User ID:</span>
              <span className="text-xs font-mono bg-muted px-2 py-1 rounded">
                {session.sessionData.user_id.slice(-8)}
              </span>
            </div>
          </div>
        )}

        {/* Activity */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Last Activity:</span>
            <span className="text-xs">{formatTime(session.lastActivity)}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Last Heartbeat:</span>
            <span className="text-xs">{formatTime(lastHeartbeat)}</span>
          </div>
          {session.lastActivity && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Uptime:</span>
              <span className="text-xs">{formatDuration(session.lastActivity)}</span>
            </div>
          )}
        </div>

        {/* Conversation Stats */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Conversations:</span>
            <Badge variant="secondary">{conversation.totalConversations}</Badge>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Total Messages:</span>
            <Badge variant="secondary">{conversation.totalMessages}</Badge>
          </div>
          {conversation.currentConversation && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Current Chat:</span>
              <span className="text-xs max-w-24 truncate">
                {conversation.currentConversation.name}
              </span>
            </div>
          )}
        </div>

        {/* Reconnect Attempts */}
        {session.reconnectAttempts > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Reconnect Attempts:</span>
            <Badge variant="outline" className="text-yellow-600 border-yellow-600">
              {session.reconnectAttempts}
            </Badge>
          </div>
        )}

        {/* Error Display */}
        {(session.lastError || conversation.lastError) && (
          <div className="p-2 bg-destructive/10 border border-destructive/20 rounded">
            <p className="text-xs text-destructive">
              {session.lastError || conversation.lastError}
            </p>
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          {!session.isConnected && (
            <Button
              size="sm"
              onClick={() => session.connect()}
              disabled={session.isConnecting}
              className="flex-1"
            >
              {session.isConnecting ? (
                <>
                  <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  <Wifi className="w-3 h-3 mr-1" />
                  Connect
                </>
              )}
            </Button>
          )}
          
          {session.isConnected && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => session.refreshSession()}
              className="flex-1"
            >
              <RefreshCw className="w-3 h-3 mr-1" />
              Refresh
            </Button>
          )}

          {conversation.totalConversations === 0 && session.isConnected && (
            <Button
              size="sm"
              onClick={() => conversation.createConversation('Test Chat')}
              className="flex-1"
            >
              <MessageSquare className="w-3 h-3 mr-1" />
              Test Chat
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};