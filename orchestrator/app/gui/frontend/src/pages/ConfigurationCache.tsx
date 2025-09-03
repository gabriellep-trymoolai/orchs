import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';

export const ConfigurationCache: React.FC = () => {
  const [cacheEnabled, setCacheEnabled] = useState(true);
  const [cacheThreshold, setCacheThreshold] = useState([7]);
  const [defaultTTL, setDefaultTTL] = useState([3600]);

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">Cache Control</h1>
        <div className="flex gap-3">
          <Button variant="outline" className="border-border text-foreground">
            Update
          </Button>
          <Button className="bg-orange-primary hover:bg-orange-dark text-white">
            Export Cache
          </Button>
        </div>
      </div>

      <Card className="p-6 bg-card border-border">
        <div className="space-y-6">
          <p className="text-sm text-muted-foreground">
            Enable/disable and configure cache system
          </p>

          {/* Cache System Status */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Cache System Status</h3>
                <p className="text-sm text-muted-foreground">Toggle cache system on enabled/disabled</p>
              </div>
              <Switch
                checked={cacheEnabled}
                onCheckedChange={setCacheEnabled}
                className="data-[state=checked]:bg-orange-primary"
              />
            </div>
          </div>

          {/* Cache Threshold */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Cache Threshold (%)</h3>
              </div>
              <Button variant="outline" size="sm">
                Update
              </Button>
            </div>
            <div className="space-y-4">
              <div className="px-4">
                <Slider
                  value={cacheThreshold}
                  onValueChange={setCacheThreshold}
                  max={10}
                  min={0.1}
                  step={0.1}
                  className="w-full"
                />
              </div>
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>0.1</span>
                <span className="font-medium text-orange-primary">{cacheThreshold[0]}</span>
                <span>10</span>
              </div>
            </div>
          </div>

          {/* Default TTL */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Default TTL (seconds)</h3>
              </div>
              <Button variant="outline" size="sm">
                Update
              </Button>
            </div>
            <div className="space-y-4">
              <div className="px-4">
                <Slider
                  value={defaultTTL}
                  onValueChange={setDefaultTTL}
                  max={10000}
                  min={1}
                  step={100}
                  className="w-full"
                />
              </div>
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>1</span>
                <span className="font-medium text-orange-primary">{defaultTTL[0]}</span>
                <span>10000</span>
              </div>
            </div>
          </div>

          {/* Clear Cache */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Clear Cache</h3>
                <p className="text-sm text-muted-foreground">Toggle cache system enabled or disabled</p>
              </div>
              <Button variant="outline" className="border-destructive text-destructive hover:bg-destructive hover:text-white">
                Clear Cache
              </Button>
            </div>
          </div>

          {/* List Cache Keys */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">List Cache Keys</h3>
                <p className="text-sm text-muted-foreground">View all cached keys</p>
              </div>
              <Button variant="outline">
                View Keys
              </Button>
            </div>
          </div>

          {/* Warm Cache */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Warm Cache</h3>
                <p className="text-sm text-muted-foreground">Preload frequently accessed items</p>
                <p className="text-xs text-muted-foreground">Enter prompts to warm (one per line)</p>
              </div>
              <Button className="bg-orange-primary hover:bg-orange-dark text-white">
                Start Warmup
              </Button>
            </div>
          </div>

          {/* Cache Analytics */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-lg font-medium text-foreground">Cache Analytics</h3>
                <p className="text-sm text-muted-foreground">Detailed performance analytics</p>
              </div>
              <Button variant="outline">
                Refresh
              </Button>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="text-sm text-muted-foreground">Total Requests</div>
              </div>
              <div className="text-2xl font-bold text-foreground">45,230</div>
              <div className="text-sm text-green-400">+25% vs previous 24h</div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="text-sm text-muted-foreground">Hit Rate</div>
              </div>
              <div className="text-2xl font-bold text-foreground">87.4%</div>
              <div className="text-sm text-green-400">+21% from last hour</div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <div className="text-sm text-muted-foreground">Avg Response Time</div>
              </div>
              <div className="text-2xl font-bold text-foreground">12.3ms</div>
              <div className="text-sm text-green-400">+5.8ms vs previous 24h</div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};