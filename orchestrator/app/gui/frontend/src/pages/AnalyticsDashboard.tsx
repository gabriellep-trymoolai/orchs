import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { TrendingUp, TrendingDown, Clock, Target, Loader2, Wifi, WifiOff, Shield, AlertTriangle, Hash } from 'lucide-react';
import { apiClient } from '@/services/api-client';
import { useAppSession } from '@/contexts/AppContext';

// Define AnalyticsMetrics interface locally since we removed the analytics-websocket service
interface AnalyticsMetrics {
  total_api_calls: number;
  total_cost: number;
  total_tokens: number;
  cache_hit_rate: number;
  avg_response_time_ms: number;
  firewall_blocks: number;
  provider_breakdown: Array<{
    provider: string;
    calls: number;
    cost: number;
    tokens: number;
  }>;
  data_source?: string;
  phoenix_available?: boolean;
  phoenix_connected?: boolean;
}

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ComponentType<{ className?: string }>;
  trend?: {
    value: string;
    isPositive: boolean;
    description: string;
  };
  isLoading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, icon: IconComponent, trend, isLoading }) => (
  <Card className="p-6 bg-card border-border">
    <div className="space-y-2">
      <p className="text-sm text-muted-foreground">{title}</p>
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2">
          <IconComponent className="h-5 w-5 text-muted-foreground" />
          {isLoading ? (
            <Loader2 className="h-6 w-6 animate-spin" />
          ) : (
            <>
            <span className="text-2xl font-bold text-foreground">{value}</span>
              {trend && (
                <div className={`flex items-center gap-1 text-sm ${trend.isPositive ? 'text-green-400' : 'text-red-400'}`}>
                  {trend.isPositive ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />}
                  <span>{trend.description}</span>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  </Card>
);

export const AnalyticsDashboard: React.FC = () => {
  const session = useAppSession(); // Use shared session
  const [timeRange, setTimeRange] = useState('30d');
  const [realTimeMetrics, setRealTimeMetrics] = useState<AnalyticsMetrics | null>(null);
  const [wsError, setWsError] = useState<string | null>(null);
  const [isSubscribed, setIsSubscribed] = useState(false);

  // Calculate date range based on selection
  const getDateRange = () => {
    const end = new Date();
    const start = new Date();
    
    switch (timeRange) {
      case '1h':
        start.setHours(end.getHours() - 1);
        break;
      case '24h':
        start.setHours(end.getHours() - 24);
        break;
      case '7d':
        start.setDate(end.getDate() - 7);
        break;
      case '30d':
        start.setDate(end.getDate() - 30);
        break;
      case '90d':
        start.setDate(end.getDate() - 90);
        break;
      default:
        start.setDate(end.getDate() - 30);
    }
    
    // For date-based queries, set start to beginning of day and end to end of day
    if (['7d', '30d', '90d'].includes(timeRange)) {
      start.setHours(0, 0, 0, 0);
      end.setHours(23, 59, 59, 999);
      
      // Return date-only strings for date-based ranges
      return {
        start: start.toISOString().split('T')[0],
        end: end.toISOString().split('T')[0]
      };
    }
    
    // For time-based ranges (1h, 24h), return full ISO strings
    return {
      start: start.toISOString(),
      end: end.toISOString()
    };
  };

  // WebSocket connection management using shared session
  useEffect(() => {
    // Set up WebSocket listeners for analytics using shared session
    const unsubscribeAnalytics = session.addEventListener('analytics_response', (data) => {
      console.log('Analytics data received:', data);
      setRealTimeMetrics(data);
      setWsError(null); // Clear any previous errors
    });
    
    const unsubscribeError = session.addEventListener('analytics_error', (error) => {
      console.error('Analytics error received:', error);
      setWsError(error.error || 'Analytics error');
    });
    
    const unsubscribeConfirmation = session.addEventListener('analytics_subscription_confirmed', (data) => {
      console.log('Analytics subscription confirmed:', data);
      setIsSubscribed(data.subscribed || false);
    });
    
    const unsubscribeUnsubscribed = session.addEventListener('analytics_unsubscribed', (data) => {
      console.log('Analytics unsubscribed:', data);
      setIsSubscribed(false);
      setRealTimeMetrics(null);
    });

    // Subscribe to analytics if session is connected
    if (session.isConnected && !isSubscribed) {
      console.log('Subscribing to analytics via session WebSocket');
      // Use session's sendRawMessage to subscribe to analytics
      session.sendRawMessage({
        type: 'analytics_subscribe',
        data: {
          metrics: ['overview', 'providers', 'cache'],
          interval: 30 // 30 second updates
        }
      });
    }

    // Cleanup on unmount
    return () => {
      unsubscribeAnalytics();
      unsubscribeError();
      unsubscribeConfirmation();
      unsubscribeUnsubscribed();
      
      if (session.isConnected && isSubscribed) {
        console.log('Unsubscribing from analytics');
        session.sendRawMessage({
          type: 'analytics_unsubscribe'
        });
      }
    };
  }, [session.isConnected, isSubscribed]);

  // Handle time range changes - request new analytics data
  useEffect(() => {
    if (session.isConnected && isSubscribed) {
      const { start, end } = getDateRange();
      console.log('Requesting analytics data for time range:', timeRange);
      session.sendRawMessage({
        type: 'analytics_request',
        data: {
          start_date: start,
          end_date: end,
          time_range: timeRange
        }
      });
    }
  }, [timeRange, session.isConnected, isSubscribed]);

  // Calculate date range based on selection (for API calls)
  const getApiDateRange = () => {
    const end = new Date();
    const start = new Date();
    
    switch (timeRange) {
      case '1h':
        start.setHours(end.getHours() - 1);
        break;
      case '24h':
        start.setHours(end.getHours() - 24);
        break;
      case '7d':
        start.setDate(end.getDate() - 7);
        break;
      case '30d':
        start.setDate(end.getDate() - 30);
        break;
      case '90d':
        start.setDate(end.getDate() - 90);
        break;
      default:
        start.setDate(end.getDate() - 30);
    }
    
    // For date-based queries, set start to beginning of day and end to end of day
    if (['7d', '30d', '90d'].includes(timeRange)) {
      start.setHours(0, 0, 0, 0);  // Start of start date
      end.setHours(23, 59, 59, 999);  // End of end date
    }
    
    return {
      start_date: start.toISOString(),
      end_date: end.toISOString()
    };
  };

  // Fetch analytics data
  const { data: analyticsOverview, isLoading: overviewLoading } = useQuery({
    queryKey: ['analytics-overview', timeRange],
    queryFn: () => apiClient.getAnalyticsOverview(getApiDateRange()),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: cachePerformance, isLoading: cacheLoading } = useQuery({
    queryKey: ['cache-performance', timeRange],
    queryFn: () => apiClient.getCachePerformance(getApiDateRange()),
    refetchInterval: 30000,
  });

  const { isLoading: providerLoading } = useQuery({
    queryKey: ['provider-breakdown', timeRange],
    queryFn: () => apiClient.getProviderBreakdown(getApiDateRange()),
    refetchInterval: 30000,
  });

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground flex items-center gap-2">
            Analytics Dashboard 📊
            {/* WebSocket Connection Status */}
            <div className="flex items-center gap-2 ml-4">
              {session.isConnected ? (
                isSubscribed ? (
                  <div className="flex items-center gap-1 text-green-500 text-sm">
                    <Wifi className="h-4 w-4" />
                    <span>Live Analytics</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1 text-yellow-500 text-sm">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Subscribing</span>
                  </div>
                )
              ) : session.isConnecting ? (
                <div className="flex items-center gap-1 text-yellow-500 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Connecting</span>
                </div>
              ) : session.connectionState === 'reconnecting' ? (
                <div className="flex items-center gap-1 text-orange-500 text-sm">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Reconnecting</span>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-red-500 text-sm">
                  <WifiOff className="h-4 w-4" />
                  <span>Offline</span>
                </div>
              )}
            </div>
          </h1>
          <p className="text-muted-foreground mt-1">
            Track your model usage and compare costs in real-time.
            {wsError && <span className="text-red-500 ml-2">(Error: {wsError})</span>}
            {realTimeMetrics && realTimeMetrics.data_source && (
              <span className="text-blue-500 ml-2">(Source: {realTimeMetrics.data_source})</span>
            )}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-32">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="1h">Last 1 hour</SelectItem>
              <SelectItem value="24h">Last 24 hours</SelectItem>
              <SelectItem value="7d">Last 7 days</SelectItem>
              <SelectItem value="30d">Last 30 days</SelectItem>
              <SelectItem value="90d">Last 90 days</SelectItem>
            </SelectContent>
          </Select>
          
          <div className="flex gap-2">
            {session.connectionState !== 'connected' && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => session.connect()}
                disabled={session.connectionState === 'connecting'}
              >
                {session.connectionState === 'connecting' ? 'Connecting...' : 'Reconnect'}
              </Button>
            )}
            
            {session.isConnected && !isSubscribed && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  session.sendRawMessage({
                    type: 'analytics_subscribe',
                    data: { metrics: ['overview', 'providers', 'cache'], interval: 30 }
                  });
                }}
              >
                Subscribe to Analytics
              </Button>
            )}
            
            {isSubscribed && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  session.sendRawMessage({ type: 'analytics_unsubscribe' });
                }}
              >
                Unsubscribe
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Cache Hit Rate"
          value={
            realTimeMetrics 
              ? `${realTimeMetrics.cache_hit_rate.toFixed(1)}%`
              : cacheLoading 
                ? "Loading..." 
                : `${cachePerformance?.cache_performance?.cache_hit_rate || 0}%`
          }
          icon={Target}
          isLoading={!realTimeMetrics && cacheLoading}
          trend={undefined} // Remove trend for now - can be added later
        />
        
        <MetricCard
          title="Avg. Response Time"
          value={
            realTimeMetrics 
              ? `${realTimeMetrics.avg_response_time_ms}ms`
              : overviewLoading 
                ? "Loading..." 
                : `${analyticsOverview?.overview?.avg_response_time_ms || 0}ms`
          }
          icon={Clock}
          isLoading={!realTimeMetrics && overviewLoading}
        />
        
        <MetricCard
          title="Total API Calls"
          value={
            realTimeMetrics 
              ? realTimeMetrics.total_api_calls.toLocaleString()
              : overviewLoading 
                ? "Loading..." 
                : (analyticsOverview?.overview?.total_api_calls || 0).toLocaleString()
          }
          icon={TrendingUp}
          isLoading={!realTimeMetrics && overviewLoading}
        />
        
        <MetricCard
          title="Total Tokens Used"
          value={
            realTimeMetrics 
              ? `${(realTimeMetrics.total_tokens || 0).toLocaleString()} tokens`
              : overviewLoading 
                ? "Loading..." 
                : `${(analyticsOverview?.overview?.total_tokens || 0).toLocaleString()} tokens`
          }
          icon={Hash}
          isLoading={!realTimeMetrics && overviewLoading}
        />
      </div>

      {/* Enhanced Security & Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard
          title="Firewall Blocks"
          value={
            realTimeMetrics 
              ? realTimeMetrics.firewall_blocks.toLocaleString()
              : overviewLoading 
                ? "Loading..." 
                : (analyticsOverview?.overview?.firewall_blocks || 0).toLocaleString()
          }
          icon={Shield}
          isLoading={!realTimeMetrics && overviewLoading}
        />
        
        <MetricCard
          title="Cache Efficiency"
          value={
            realTimeMetrics 
              ? `${realTimeMetrics.cache_hit_rate.toFixed(1)}%`
              : cacheLoading 
                ? "Loading..." 
                : `${cachePerformance?.cache_performance?.cache_hit_rate || 0}%`
          }
          icon={Target}
          isLoading={!realTimeMetrics && cacheLoading}
          trend={realTimeMetrics && realTimeMetrics.cache_hit_rate > 50 ? {
            value: "Optimized",
            isPositive: true,
            description: "High cache hit rate"
          } : undefined}
        />
        
        <MetricCard
          title="Request Security"
          value={
            realTimeMetrics 
              ? `${((realTimeMetrics.total_api_calls - realTimeMetrics.firewall_blocks) / Math.max(realTimeMetrics.total_api_calls, 1) * 100).toFixed(1)}%`
              : overviewLoading 
                ? "Loading..." 
                : `${analyticsOverview?.overview ? ((analyticsOverview.overview.total_api_calls - analyticsOverview.overview.firewall_blocks) / Math.max(analyticsOverview.overview.total_api_calls, 1) * 100).toFixed(1) : 100}%`
          }
          icon={AlertTriangle}
          isLoading={!realTimeMetrics && overviewLoading}
        />
      </div>

      {/* Provider Breakdown Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">Total API Calls</h3>
              <span className="text-2xl font-bold text-foreground">
                {overviewLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : (analyticsOverview?.overview?.total_api_calls || 0).toLocaleString()}
              </span>
            </div>
{(realTimeMetrics?.provider_breakdown || !providerLoading) ? (
              <div className="grid grid-cols-2 gap-4">
                {(realTimeMetrics?.provider_breakdown || analyticsOverview?.provider_breakdown)?.slice(0, 2).map((provider: any, index: number) => (
                  <div key={index} className="p-4 rounded-lg border border-border">
                    <div className="text-sm text-muted-foreground capitalize">{provider.provider}</div>
                    <div className="text-xl font-bold text-orange-primary">{provider.calls.toLocaleString()}</div>
                    <div className="text-xs text-muted-foreground">{provider.tokens.toLocaleString()} Tokens Used</div>
                  </div>
                )) || (
                  <div className="col-span-2 text-center text-muted-foreground">No data available</div>
                )}
              </div>
            ) : (
              <div className="flex items-center justify-center h-24">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            )}
          </div>
        </Card>

        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">Total Cost</h3>
              <span className="text-2xl font-bold text-foreground">
                {overviewLoading ? <Loader2 className="h-6 w-6 animate-spin" /> : `$${(analyticsOverview?.overview?.total_cost || 0).toFixed(2)}`}
              </span>
            </div>
            {providerLoading ? (
              <div className="flex items-center justify-center h-24">
                <Loader2 className="h-6 w-6 animate-spin" />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-4">
                {analyticsOverview?.provider_breakdown?.slice(0, 2).map((provider: any, index: number) => (
                  <div key={index} className="p-4 rounded-lg border border-border">
                    <div className="text-sm text-muted-foreground capitalize">{provider.provider}</div>
                    <div className="text-xl font-bold text-orange-primary">${provider.cost.toFixed(2)}</div>
                    <div className="text-xs text-muted-foreground">
                      Avg ${(provider.cost / Math.max(provider.calls, 1)).toFixed(4)} per query
                    </div>
                  </div>
                )) || (
                  <div className="col-span-2 text-center text-muted-foreground">No data available</div>
                )}
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">Cost Optimization Impact</h3>
              <Select defaultValue="2025">
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2025">2025</SelectItem>
                  <SelectItem value="2024">2024</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">$5,184 saved</div>
              <div className="text-sm text-muted-foreground">800k tokens used</div>
            </div>
            <div className="h-40 bg-muted/20 rounded-lg flex items-center justify-center">
              <span className="text-muted-foreground">Chart Placeholder</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span className="text-muted-foreground">Potential Cost: $11,520</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-primary rounded"></div>
                <span className="text-muted-foreground">Actual Cost: $6336</span>
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">Monthly Cost Breakdown</h3>
              <Select defaultValue="jan-2025">
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="jan-2025">Jan 2025</SelectItem>
                  <SelectItem value="dec-2024">Dec 2024</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">$3,960 spent</div>
              <div className="text-sm text-muted-foreground">250,500 tokens</div>
            </div>
            <div className="h-40 bg-muted/20 rounded-lg flex items-center justify-center">
              <span className="text-muted-foreground">Chart Placeholder</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span className="text-muted-foreground">OpenAI Cost: $3,600</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-primary rounded"></div>
                <span className="text-muted-foreground">MoolAI Cost: $360</span>
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Bottom Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">Monthly API Calls Comparison</h3>
              <Select defaultValue="jan-2025">
                <SelectTrigger className="w-32">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="jan-2025">Jan 2025</SelectItem>
                  <SelectItem value="dec-2024">Dec 2024</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <div className="text-2xl font-bold text-foreground">500 calls</div>
              <div className="text-sm text-muted-foreground">2,242 tokens</div>
            </div>
            <div className="h-40 bg-muted/20 rounded-lg flex items-center justify-center">
              <span className="text-muted-foreground">Chart Placeholder</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-500 rounded"></div>
                <span className="text-muted-foreground">OpenAI: 245</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-orange-primary rounded"></div>
                <span className="text-muted-foreground">MoolAI: 255</span>
              </div>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-card border-border">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-medium text-foreground">API Call vs Cost Distribution</h3>
              <Select defaultValue="2025">
                <SelectTrigger className="w-24">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="2025">2025</SelectItem>
                  <SelectItem value="2024">2024</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center space-y-2">
                <div className="text-sm text-muted-foreground">Call</div>
                <div className="w-24 h-24 mx-auto bg-muted/20 rounded-full flex items-center justify-center">
                  <span className="text-xs text-muted-foreground">Pie Chart</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded"></div>
                      <span>OpenAI calls</span>
                    </div>
                    <span>50%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-orange-primary rounded"></div>
                      <span>MoolAI calls</span>
                    </div>
                    <span>50%</span>
                  </div>
                </div>
              </div>
              <div className="text-center space-y-2">
                <div className="text-sm text-muted-foreground">Cost</div>
                <div className="w-24 h-24 mx-auto bg-muted/20 rounded-full flex items-center justify-center">
                  <span className="text-xs text-muted-foreground">Pie Chart</span>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-blue-500 rounded"></div>
                      <span>OpenAI Cost</span>
                    </div>
                    <span>90%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-1">
                      <div className="w-2 h-2 bg-orange-primary rounded"></div>
                      <span>MoolAI Cost</span>
                    </div>
                    <span>10%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};