import React from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Plus, Edit, Trash2, ChevronDown } from 'lucide-react';

export const Organizations: React.FC = () => {
  const organizations = [
    {
      id: 1,
      name: 'Organization 1',
      orgId: 'org-1',
      created: '2025-01-16'
    },
    {
      id: 2,
      name: 'Organization 2',
      orgId: 'org-2',
      created: '2025-01-16'
    },
    {
      id: 3,
      name: 'Organization 3',
      orgId: 'org-3',
      created: '2025-01-16'
    },
    {
      id: 4,
      name: 'Organization 4',
      orgId: 'org-4',
      created: '2025-01-16'
    },
    {
      id: 5,
      name: 'Organization 5',
      orgId: 'org-5',
      created: '2025-01-16'
    }
  ];

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Organizations</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Create, configure, and manage XXX orchestrator instances for organizations
          </p>
        </div>
        <Button className="bg-orange-primary hover:bg-orange-dark text-white">
          <Plus className="h-4 w-4 mr-2" />
          Add New Organization
        </Button>
      </div>

      {/* Organizations List */}
      <div className="space-y-4">
        {organizations.map((org) => (
          <Card key={org.id} className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <ChevronDown className="h-4 w-4" />
                </Button>
                
                <div className="flex items-center gap-3">
                  <div className="w-6 h-6 bg-orange-primary/20 rounded border border-orange-primary/30 flex items-center justify-center">
                    <div className="w-3 h-3 bg-orange-primary rounded-sm"></div>
                  </div>
                  <div className="space-y-1">
                    <h3 className="text-lg font-medium text-foreground">{org.name}</h3>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span>ID: {org.orgId}</span>
                      <span>Created: {org.created}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <Edit className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0 text-red-400 hover:text-red-300">
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};