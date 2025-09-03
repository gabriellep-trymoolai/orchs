import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { TrendingUp, TrendingDown, Clock, Target, Server, AlertCircle, CheckCircle, Activity } from 'lucide-react';
import { apiClient } from '@/services/api-client';

export const SystemMonitoringDashboard: React.FC = () => {
  // Fetch latest system metrics
  const { data: systemMetrics, isLoading, error, refetch } = useQuery({
    queryKey: ['systemMetrics'],
    queryFn: () => apiClient.getLatestSystemMetrics(),
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: 3,
  });

  // Fetch system health
  const { data: systemHealth } = useQuery({
    queryKey: ['systemHealth'],
    queryFn: () => apiClient.getSystemHealth(),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  // Calculate system health score
  const calculateHealthScore = () => {
    if (!systemMetrics) return 0;
    
    const cpuHealth = systemMetrics.cpu_usage_percent ? (100 - systemMetrics.cpu_usage_percent) : 100;
    const memoryHealth = systemMetrics.memory_percent ? (100 - systemMetrics.memory_percent) : 100;
    const storageHealth = systemMetrics.storage_percent ? (100 - systemMetrics.storage_percent) : 100;
    
    return Math.round((cpuHealth + memoryHealth + storageHealth) / 3);
  };

  // Get organization ID for display
  const orgId = apiClient.getConfig().organizationId;

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground flex items-center gap-2">
            <Server className="h-6 w-6 text-orange-primary" />
            System Performance Monitor
          </h1>
          <p className="text-muted-foreground mt-1">
            Real-time system metrics and performance monitoring for {orgId.toUpperCase()}
          </p>
        </div>
        <div className="flex items-center gap-3">
          {systemHealth?.status === 'healthy' ? (
            <Badge variant="outline" className="text-green-600 border-green-600">
              <CheckCircle className="w-3 h-3 mr-1" />
              System Healthy
            </Badge>
          ) : (
            <Badge variant="outline" className="text-red-600 border-red-600">
              <AlertCircle className="w-3 h-3 mr-1" />
              Issues Detected
            </Badge>
          )}
          <Button onClick={() => refetch()} variant="outline" size="sm">
            <Activity className="h-4 w-4 mr-1" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Failed to load system metrics: {error.message}
          </AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-6 bg-card border-border">
              <div className="space-y-2 animate-pulse">
                <div className="h-4 bg-muted rounded w-2/3"></div>
                <div className="h-8 bg-muted rounded w-1/2"></div>
                <div className="h-3 bg-muted rounded w-3/4"></div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Metrics Grid */}
      {systemMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* System Health Score */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">System Health Score</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-foreground">{calculateHealthScore()}%</span>
                <div className="flex items-center gap-1 text-sm text-green-400">
                  <TrendingUp className="h-4 w-4" />
                  <span>Optimal</span>
                </div>
              </div>
            </div>
          </Card>

          {/* API Response Time */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">API Response Time</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-foreground">
                  {systemMetrics.api_latency_ms ? `${systemMetrics.api_latency_ms}ms` : 'N/A'}
                </span>
                <div className="flex items-center gap-1 text-sm text-green-400">
                  <Clock className="h-4 w-4" />
                  <span>Normal</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Memory Usage */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Memory Usage</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-foreground">
                  {systemMetrics.memory_percent ? `${systemMetrics.memory_percent.toFixed(1)}%` : 'N/A'}
                </span>
                <div className={`flex items-center gap-1 text-sm ${
                  systemMetrics.memory_percent > 80 ? 'text-red-400' : 'text-green-400'
                }`}>
                  {systemMetrics.memory_percent > 80 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  <span>{systemMetrics.memory_percent > 80 ? 'High' : 'Normal'}</span>
                </div>
              </div>
            </div>
          </Card>

          {/* Active Connections */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">Network Connections</p>
              <div className="flex items-center gap-2">
                <span className="text-2xl font-bold text-foreground">
                  {systemMetrics.network_connections || 'N/A'}
                </span>
                <div className="flex items-center gap-1 text-sm text-green-400">
                  <Target className="h-4 w-4" />
                  <span>Active</span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Resource Usage Cards */}
      {systemMetrics && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* CPU & Memory Overview */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-foreground">CPU & Memory Overview</h3>
                <span className="text-2xl font-bold text-foreground">
                  {systemMetrics.cpu_cores_used ? `${systemMetrics.cpu_cores_used.toFixed(1)} cores` : 'N/A'}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-border">
                  <div className="text-sm text-muted-foreground">CPU Usage</div>
                  <div className="text-xl font-bold text-orange-primary">
                    {systemMetrics.cpu_usage_percent ? `${systemMetrics.cpu_usage_percent.toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Load: {systemMetrics.cpu_load_1min ? systemMetrics.cpu_load_1min.toFixed(2) : 'N/A'}
                  </div>
                </div>
                <div className="p-4 rounded-lg border border-border">
                  <div className="text-sm text-muted-foreground">Memory</div>
                  <div className="text-xl font-bold text-orange-primary">
                    {systemMetrics.memory_percent ? `${systemMetrics.memory_percent.toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {systemMetrics.memory_usage_mb && systemMetrics.memory_total_mb 
                      ? `${(systemMetrics.memory_usage_mb / 1024).toFixed(1)}GB / ${(systemMetrics.memory_total_mb / 1024).toFixed(1)}GB`
                      : 'N/A'
                    }
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Storage & Network */}
          <Card className="p-6 bg-card border-border">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium text-foreground">Storage & Network</h3>
                <span className="text-2xl font-bold text-foreground">
                  {systemMetrics.network_in_mb_s && systemMetrics.network_out_mb_s
                    ? `${(systemMetrics.network_in_mb_s + systemMetrics.network_out_mb_s).toFixed(1)} MB/s`
                    : 'N/A'
                  }
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-border">
                  <div className="text-sm text-muted-foreground">Storage</div>
                  <div className="text-xl font-bold text-orange-primary">
                    {systemMetrics.storage_percent ? `${systemMetrics.storage_percent.toFixed(1)}%` : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {systemMetrics.storage_usage_gb && systemMetrics.storage_total_gb
                      ? `${systemMetrics.storage_usage_gb.toFixed(1)}GB / ${systemMetrics.storage_total_gb.toFixed(1)}GB`
                      : 'N/A'
                    }
                  </div>
                </div>
                <div className="p-4 rounded-lg border border-border">
                  <div className="text-sm text-muted-foreground">Network I/O</div>
                  <div className="text-xl font-bold text-orange-primary">
                    {systemMetrics.network_in_mb_s ? `${systemMetrics.network_in_mb_s.toFixed(1)}` : 'N/A'}
                    <span className="text-sm text-muted-foreground"> / </span>
                    {systemMetrics.network_out_mb_s ? `${systemMetrics.network_out_mb_s.toFixed(1)}` : 'N/A'}
                  </div>
                  <div className="text-xs text-muted-foreground">In / Out MB/s</div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};