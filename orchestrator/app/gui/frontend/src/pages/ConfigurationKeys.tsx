import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Eye, EyeOff, Copy, Info } from 'lucide-react';

export const ConfigurationKeys: React.FC = () => {
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showClaudeKey, setShowClaudeKey] = useState(false);

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">API Keys</h1>
        <Button variant="outline" className="border-border text-foreground">
          Update
        </Button>
      </div>

      {/* API Keys Section */}
      <Card className="p-6 bg-card border-border">
        <h2 className="text-lg font-medium text-foreground mb-6">API Keys</h2>
        
        <div className="space-y-6">
          {/* OpenAI Configuration */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label className="text-sm font-medium text-foreground">Currently used language model</label>
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <span className="text-sm text-muted-foreground">OpenAI</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">API key</span>
                <Info className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
            
            <div className="flex gap-3">
              <Select defaultValue="openai">
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="claude">Claude</SelectItem>
                </SelectContent>
              </Select>
              
              <div className="flex-1 relative">
                <Input
                  type={showOpenAIKey ? "text" : "password"}
                  value="sk-...xxxxxxxxxxxxxxxxxxxxxxxxxxxx73"
                  className="pr-20"
                  readOnly
                />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                    className="h-8 w-8 p-0"
                  >
                    {showOpenAIKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="text-xs text-orange-primary">
              <a href="#" className="underline">How to get your OpenAI key?</a>
            </div>
            
            <div className="text-xs text-green-500 font-medium">
              Valid Key ✓
            </div>
          </div>

          {/* Claude Configuration */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <label className="text-sm font-medium text-foreground">Currently used language model</label>
                <div className="flex items-center gap-2">
                  <div className="w-5 h-5 rounded-full bg-orange-primary flex items-center justify-center">
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                  </div>
                  <span className="text-sm text-muted-foreground">Claude</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-muted-foreground">API key</span>
                <Info className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
            
            <div className="flex gap-3">
              <Select defaultValue="claude">
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="claude">Claude</SelectItem>
                  <SelectItem value="openai">OpenAI</SelectItem>
                </SelectContent>
              </Select>
              
              <div className="flex-1 relative">
                <Input
                  type={showClaudeKey ? "text" : "password"}
                  value="sk-...xxxxxxxxxxxxxxxxxxxxxxxxxxxx73"
                  className="pr-20"
                  readOnly
                />
                <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowClaudeKey(!showClaudeKey)}
                    className="h-8 w-8 p-0"
                  >
                    {showClaudeKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0"
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
            
            <div className="text-xs text-orange-primary">
              <a href="#" className="underline">How to get your Claude key?</a>
            </div>
            
            <div className="text-xs text-green-500 font-medium">
              Valid Key ✓
            </div>
          </div>

          {/* Add New API Key Button */}
          <Button 
            variant="outline" 
            className="border-border text-foreground bg-muted/20"
          >
            Add New API Key
          </Button>
        </div>
      </Card>
    </div>
  );
};