import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Search, Filter, Download } from 'lucide-react';

export const Logs: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');

  const interactions = [
    {
      id: 1,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 2,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 3,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 4,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 5,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 6,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 7,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 8,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
    {
      id: 9,
      timestamp: 'May 13, 2025, 7:45 AM',
      user: 'Arun Krishnaswamy',
      email: 'admin@trymool.ai',
      model: 'OpenAI',
      input: 'Tell me more about modern deep learning.',
      output: 'Deep learning is a subfield of machine learning that focuses on the use of artificial neural networks to model and solve complex...',
      filesAttached: 'samplefile.pdf',
    },
  ];

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">Logs</h1>
        <Button className="bg-orange-primary hover:bg-orange-dark text-white">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>
      </div>

      <Tabs defaultValue="interactions" className="w-full">
        <TabsList className="grid w-full grid-cols-4 lg:w-96">
          <TabsTrigger value="interactions" className="data-[state=active]:bg-orange-primary data-[state=active]:text-white">
            Interactions
          </TabsTrigger>
          <TabsTrigger value="users">Users</TabsTrigger>
          <TabsTrigger value="errors">Errors</TabsTrigger>
          <TabsTrigger value="topics">Topics</TabsTrigger>
        </TabsList>

        <TabsContent value="interactions" className="space-y-6">
          {/* Stats */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold text-foreground">1200</span>
              <span className="text-sm text-muted-foreground">Interactions</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Track the interactions of your organization, generate reports and assess performance here.
            </p>
          </div>

          {/* Search and Filter */}
          <div className="flex gap-3">
            <Button variant="outline" size="sm" className="border-border">
              <Filter className="h-4 w-4 mr-2" />
              Filter
            </Button>
            <div className="flex-1 relative max-w-md">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search..."
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
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">TIMESTAMP ↓</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">USER</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">MODEL</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">INPUT</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">OUTPUT</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">FILES ATTACHED</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">FOR MORE</th>
                  </tr>
                </thead>
                <tbody>
                  {interactions.map((interaction) => (
                    <tr key={interaction.id} className="border-b border-border hover:bg-muted/30">
                      <td className="p-4">
                        <div className="text-sm text-foreground">{interaction.timestamp}</div>
                      </td>
                      <td className="p-4">
                        <div className="space-y-1">
                          <div className="text-sm font-medium text-foreground">{interaction.user}</div>
                          <div className="text-xs text-muted-foreground">{interaction.email}</div>
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge variant="outline" className="border-green-500 text-green-400">
                          <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                          {interaction.model}
                        </Badge>
                      </td>
                      <td className="p-4 max-w-xs">
                        <div className="text-sm text-foreground truncate">
                          {interaction.input}
                        </div>
                      </td>
                      <td className="p-4 max-w-xs">
                        <div className="text-sm text-foreground truncate">
                          {interaction.output}
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="text-sm text-orange-primary underline cursor-pointer">
                          {interaction.filesAttached}
                        </div>
                      </td>
                      <td className="p-4">
                        <Button variant="ghost" size="sm" className="text-muted-foreground">
                          ⋯
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="users" className="space-y-6">
          {/* Stats */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold text-foreground">6 Users</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Track the interactions of your organization users, generate reports and assess performance here.
            </p>
          </div>

          {/* Search and Export */}
          <div className="flex justify-between items-center">
            <div className="flex gap-3">
              <Button variant="outline" size="sm" className="border-border">
                <Filter className="h-4 w-4 mr-2" />
                Filter
              </Button>
              <div className="flex-1 relative max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search..."
                  className="pl-10"
                />
              </div>
            </div>
            <Button className="bg-orange-primary hover:bg-orange-dark text-white">
              Export
            </Button>
          </div>

          {/* Table */}
          <Card className="overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-muted/50">
                  <tr className="border-b border-border">
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">USER</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">INTERACTIONS</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">RISKY BEHAVIOUR</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">USER SATISFACTION</th>
                    <th className="text-left p-4 text-sm font-medium text-muted-foreground">AVERAGE RESPONSE TIME</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: 6 }).map((_, index) => (
                    <tr key={index} className="border-b border-border hover:bg-muted/30">
                      <td className="p-4">
                        <div className="space-y-1">
                          <div className="text-sm font-medium text-foreground">Arun Krishnaswamy</div>
                          <div className="text-xs text-muted-foreground">admin@trymool.ai</div>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="flex flex-col gap-1">
                            <div className="w-12 h-2 bg-muted rounded-full overflow-hidden">
                              <div className="w-8 h-full bg-orange-primary rounded-full"></div>
                            </div>
                          </div>
                          <span className="text-sm text-foreground">245</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 bg-red-500/20 rounded-full flex items-center justify-center">
                            <span className="text-xs text-red-400">⚠</span>
                          </div>
                          <span className="text-sm text-red-400">21</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 bg-green-500/20 rounded-full flex items-center justify-center">
                            <span className="text-xs text-green-400">●</span>
                          </div>
                          <span className="text-sm text-green-400">87%</span>
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 bg-gray-500/20 rounded-full flex items-center justify-center">
                            <span className="text-xs text-gray-400">●</span>
                          </div>
                          <span className="text-sm text-foreground">82ms</span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center justify-between p-4 border-t border-border">
              <span className="text-sm text-muted-foreground">1-6 of 6</span>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">Rows per page: 10</span>
                <div className="flex gap-1">
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">‹</Button>
                  <span className="text-sm text-muted-foreground">1/1</span>
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">›</Button>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="errors">
          <div className="text-center py-12">
            <p className="text-muted-foreground">Error logs will be displayed here</p>
          </div>
        </TabsContent>

        <TabsContent value="topics">
          <div className="text-center py-12">
            <p className="text-muted-foreground">Topic analysis will be displayed here</p>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};