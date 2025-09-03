import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Search, Plus } from 'lucide-react';

interface User {
  user_id: string;
  username: string;
  email: string;
  full_name?: string;
  role: string;
  active: boolean;
  department?: string;
  job_title?: string;
  created_at?: string;
  last_login?: string;
}

interface APIResponse {
  success: boolean;
  data: {
    items: User[];
    total_items: number;
    page: number;
    total_pages: number;
  };
  message: string;
}

export const Users: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [users, setUsers] = useState<User[]>([]);
  const [totalUsers, setTotalUsers] = useState(0);
  const [activeUsers, setActiveUsers] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch users from API
  const fetchUsers = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('/api/v1/orchestrators/org_001/users?page=1&page_size=50');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data: APIResponse = await response.json();
      
      if (data.success) {
        setUsers(data.data.items);
        setTotalUsers(data.data.total_items);
        setActiveUsers(data.data.items.filter(user => user.active).length);
      } else {
        throw new Error(data.message || 'Failed to fetch users');
      }
    } catch (err) {
      console.error('Error fetching users:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch users');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, []);

  // Format relative time
  const formatRelativeTime = (dateString?: string) => {
    if (!dateString) return 'Never';
    
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffMins < 60) return `${diffMins} mins ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    return `${diffDays} days ago`;
  };

  // Filter users based on search term
  const filteredUsers = users.filter(user =>
    user.username.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.full_name?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) {
    return (
      <div className="flex-1 p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading users...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 p-6 space-y-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center space-y-2">
            <div className="text-red-500">Error loading users</div>
            <div className="text-sm text-muted-foreground">{error}</div>
            <Button onClick={fetchUsers} variant="outline" size="sm">
              Retry
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">User Management</h1>
        <Button className="bg-orange-primary hover:bg-orange-dark text-white">
          <Plus className="h-4 w-4 mr-2" />
          Add User
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="p-6">
          <div className="space-y-2">
            <h3 className="text-sm text-muted-foreground">Total Users</h3>
            <div className="flex items-center gap-2">
              <span className="text-3xl font-bold text-foreground">{totalUsers}</span>
              <span className="text-sm text-muted-foreground">registered users</span>
            </div>
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="space-y-2">
            <h3 className="text-sm text-muted-foreground">Active Users</h3>
            <div className="flex items-center gap-2">
              <span className="text-3xl font-bold text-foreground">{activeUsers}</span>
              <span className="text-sm text-muted-foreground">currently active</span>
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <div className="space-y-2">
            <h3 className="text-sm text-muted-foreground">Avg. Session Duration</h3>
            <div className="flex items-center gap-2">
              <span className="text-3xl font-bold text-foreground">45 min</span>
              <span className="text-sm text-red-400">-15.3% from last month</span>
            </div>
          </div>
        </Card>
      </div>

      {/* All Users Section */}
      <div className="space-y-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground">All Users</h2>
          <p className="text-sm text-muted-foreground">Manage the users of your organization logged into the system</p>
        </div>

        {/* Search */}
        <div className="flex gap-3">
          <div className="flex-1 relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Table */}
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-muted/50">
                <tr className="border-b border-border">
                  <th className="text-left p-4 text-sm font-medium text-muted-foreground">USER</th>
                  <th className="text-left p-4 text-sm font-medium text-muted-foreground">ROLE</th>
                  <th className="text-left p-4 text-sm font-medium text-muted-foreground">STATUS</th>
                  <th className="text-left p-4 text-sm font-medium text-muted-foreground">LAST ACTIVE</th>
                  <th className="text-left p-4 text-sm font-medium text-muted-foreground">ACTIONS</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map((user) => (
                  <tr key={user.user_id} className="border-b border-border hover:bg-muted/30">
                    <td className="p-4">
                      <div className="space-y-1">
                        <div className="text-sm font-medium text-foreground">
                          {user.full_name || user.username}
                        </div>
                        <div className="text-xs text-muted-foreground">{user.email}</div>
                        {user.job_title && user.department && (
                          <div className="text-xs text-muted-foreground">
                            {user.job_title} • {user.department}
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="p-4">
                      <span className="text-sm text-foreground">{user.role}</span>
                    </td>
                    <td className="p-4">
                      <Badge 
                        variant="outline" 
                        className={user.active 
                          ? "border-green-500 text-green-400" 
                          : "border-red-500 text-red-400"
                        }
                      >
                        <div className={`w-2 h-2 rounded-full mr-2 ${
                          user.active ? "bg-green-500" : "bg-red-500"
                        }`}></div>
                        {user.active ? "Active" : "Inactive"}
                      </Badge>
                    </td>
                    <td className="p-4">
                      <span className="text-sm text-foreground">
                        {formatRelativeTime(user.last_login)}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className="text-sm text-orange-primary cursor-pointer hover:underline">
                        Manage User
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between p-4 border-t border-border">
            <span className="text-sm text-muted-foreground">
              Showing {filteredUsers.length} of {totalUsers} users
              {searchTerm && ` (filtered by "${searchTerm}")`}
            </span>
            {totalUsers > 0 && (
              <div className="flex items-center gap-2">
                <Button 
                  onClick={fetchUsers} 
                  variant="outline" 
                  size="sm"
                  disabled={loading}
                >
                  Refresh
                </Button>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};