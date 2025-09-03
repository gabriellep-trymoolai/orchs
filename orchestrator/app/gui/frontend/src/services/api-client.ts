/**
 * Core API Client for MoolAI Orchestrator
 * Handles all HTTP requests to the backend with proper configuration
 */

export interface ApiConfig {
  baseUrl: string;
  version: string;
  organizationId: string;
  timeout?: number;
  debug?: boolean;
}

export interface ApiResponse<T = any> {
  data: T;
  status: number;
  message?: string;
  timestamp?: string;
}

export interface ApiError {
  error: string;
  message: string;
  status: number;
  details?: any;
}

class MoolAIApiClient {
  private config: ApiConfig;
  private controller: AbortController | null = null;

  constructor(config?: Partial<ApiConfig>) {
    this.config = {
      baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://20.125.25.170:8000',
      version: import.meta.env.VITE_API_VERSION || 'v1',
      organizationId: import.meta.env.VITE_ORGANIZATION_ID || 'org_001',
      timeout: 30000,
      debug: import.meta.env.VITE_DEBUG === 'true',
      ...config
    };
  }

  private log(message: string, ...args: any[]): void {
    if (this.config.debug) {
      console.log(`[MoolAI API] ${message}`, ...args);
    }
  }

  private getBaseUrl(): string {
    return `${this.config.baseUrl}/api/${this.config.version}`;
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.getBaseUrl()}${endpoint}`;
    
    // Create abort controller for timeout
    this.controller = new AbortController();
    const timeoutId = setTimeout(() => {
      this.controller?.abort();
    }, this.config.timeout);

    const defaultHeaders = {
      'Content-Type': 'application/json',
      'X-Organization-ID': this.config.organizationId,
    };

    const requestOptions: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
      signal: this.controller.signal,
    };

    this.log('Making request:', { url, options: requestOptions });

    try {
      const response = await fetch(url, requestOptions);
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          error: 'Unknown error',
          message: `HTTP ${response.status}: ${response.statusText}`,
          status: response.status
        }));
        
        this.log('API Error:', errorData);
        throw new Error(JSON.stringify(errorData));
      }

      const data = await response.json();
      this.log('API Response:', data);
      return data;

    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        
        try {
          const apiError = JSON.parse(error.message);
          throw apiError;
        } catch {
          throw new Error(error.message);
        }
      }
      
      throw error;
    }
  }

  // Core API Methods

  async get<T>(endpoint: string): Promise<T> {
    return this.makeRequest<T>(endpoint, { method: 'GET' });
  }

  async post<T>(endpoint: string, data?: any): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async put<T>(endpoint: string, data?: any): Promise<T> {
    return this.makeRequest<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.makeRequest<T>(endpoint, { method: 'DELETE' });
  }

  // Orchestrator-specific endpoints

  async getHealth(): Promise<{ status: string; timestamp: string }> {
    return this.get('/health');
  }

  async getUsers(): Promise<any[]> {
    return this.get(`/orchestrators/${this.config.organizationId}/users`);
  }

  async createUser(userData: any): Promise<any> {
    return this.post(`/orchestrators/${this.config.organizationId}/users`, userData);
  }

  async processPrompt(promptData: {
    prompt: string;
    user_id: string;
    session_id?: string;
    metadata?: Record<string, any>;
  }): Promise<any> {
    return this.post(`/orchestrators/${this.config.organizationId}/prompts`, promptData);
  }

  async createChatSession(sessionData: {
    user_id: string;
    session_name?: string;
    metadata?: Record<string, any>;
  }): Promise<any> {
    return this.post(`/orchestrators/${this.config.organizationId}/chat/sessions`, sessionData);
  }

  async getChatSessions(userId: string): Promise<any[]> {
    return this.get(`/orchestrators/${this.config.organizationId}/chat/sessions?user_id=${userId}`);
  }

  async getConfiguration(): Promise<any> {
    return this.get(`/orchestrators/${this.config.organizationId}/config`);
  }

  // Monitoring endpoints

  async getSystemHealth(): Promise<any> {
    return this.get('/system/health');
  }

  async getSystemMetrics(): Promise<any> {
    return this.get(`/system/metrics/organization/${this.config.organizationId}`);
  }

  async getLatestSystemMetrics(): Promise<any> {
    return this.get(`/system/metrics/organization/${this.config.organizationId}/latest`);
  }

  async getCpuSummary(): Promise<any> {
    return this.get(`/system/metrics/summary/cpu?organization_id=${this.config.organizationId}`);
  }

  async getMemorySummary(): Promise<any> {
    return this.get(`/system/metrics/summary/memory?organization_id=${this.config.organizationId}`);
  }

  async getStorageSummary(): Promise<any> {
    return this.get(`/system/metrics/summary/storage?organization_id=${this.config.organizationId}`);
  }

  async getSystemAlerts(): Promise<any> {
    return this.get(`/system/alerts/organization/${this.config.organizationId}`);
  }

  async getCollectionStatus(): Promise<any> {
    return this.get(`/system/status/collection?organization_id=${this.config.organizationId}`);
  }

  async getVersionHistory(): Promise<any> {
    return this.get(`/system/versions/organization/${this.config.organizationId}`);
  }

  async collectMetrics(): Promise<any> {
    return this.post('/system/collect/immediate');
  }

  // Cache endpoints

  async getCacheStatistics(): Promise<any> {
    return this.get('/cache/statistics');
  }

  async clearCache(): Promise<any> {
    return this.post('/cache/clear');
  }

  // LLM endpoints

  async queryLLM(queryData: {
    query: string;
    user_id: string;
    enable_evaluation?: boolean;
    session_id?: string;
  }): Promise<any> {
    return this.post('/llm/query', queryData);
  }

  async processLLMPrompt(promptData: {
    prompt: string;
    user_id: string;
    session_id?: string;
    use_cache?: boolean;
  }): Promise<any> {
    return this.post('/llm/prompt', promptData);
  }

  // Agent endpoints

  async queryAgent(queryData: {
    query: string;
    user_id: string;
    enable_evaluation?: boolean;
    session_id?: string;
  }): Promise<any> {
    return this.post('/agents/query', queryData);
  }

  // Analytics endpoints

  async getAnalyticsOverview(params?: {
    start_date?: string;
    end_date?: string;
    organization_id?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    queryParams.append('use_phoenix', 'true'); // Always use Phoenix backend
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.organization_id) queryParams.append('organization_id', params.organization_id);
    
    return this.get(`/analytics/overview?${queryParams.toString()}`);
  }

  async getProviderBreakdown(params?: {
    start_date?: string;
    end_date?: string;
    organization_id?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    queryParams.append('use_phoenix', 'true'); // Always use Phoenix backend
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.organization_id) queryParams.append('organization_id', params.organization_id);
    
    return this.get(`/analytics/provider-breakdown?${queryParams.toString()}`);
  }

  async getTimeSeriesData(params: {
    metric: 'cost' | 'calls' | 'tokens' | 'latency';
    interval: 'hour' | 'day';
    start_date?: string;
    end_date?: string;
    organization_id?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    queryParams.append('use_phoenix', 'true'); // Always use Phoenix backend
    queryParams.append('metric', params.metric);
    queryParams.append('interval', params.interval);
    if (params.start_date) queryParams.append('start_date', params.start_date);
    if (params.end_date) queryParams.append('end_date', params.end_date);
    if (params.organization_id) queryParams.append('organization_id', params.organization_id);
    
    return this.get(`/analytics/time-series?${queryParams.toString()}`);
  }

  async getCachePerformance(params?: {
    start_date?: string;
    end_date?: string;
    organization_id?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    queryParams.append('use_phoenix', 'true'); // Always use Phoenix backend
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.organization_id) queryParams.append('organization_id', params.organization_id);
    
    return this.get(`/analytics/cache-performance?${queryParams.toString()}`);
  }

  async getFirewallActivity(params?: {
    start_date?: string;
    end_date?: string;
    organization_id?: string;
  }): Promise<any> {
    const queryParams = new URLSearchParams();
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    if (params?.organization_id) queryParams.append('organization_id', params.organization_id);
    
    const queryString = queryParams.toString();
    return this.get(`/analytics/firewall-activity${queryString ? `?${queryString}` : ''}`);
  }

  // Utility methods

  updateConfig(newConfig: Partial<ApiConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  getConfig(): ApiConfig {
    return { ...this.config };
  }

  abort(): void {
    if (this.controller) {
      this.controller.abort();
    }
  }
}

// Export singleton instance
export const apiClient = new MoolAIApiClient();

// Export class for custom instances
export default MoolAIApiClient;
