import React, { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Trash2 } from 'lucide-react';

export const ConfigurationFirewall: React.FC = () => {
  const [firewallEnabled, setFirewallEnabled] = useState(true);
  const [autoBlockSuspicious, setAutoBlockSuspicious] = useState(true);
  const [secretsScanning, setSecretsScanning] = useState(true);
  const [toxicityDetection, setToxicityDetection] = useState(true);
  const [quarantineSystem, setQuarantineSystem] = useState(true);
  const [securityScanning, setSecurityScanning] = useState(true);
  const [enableMasking, setEnableMasking] = useState(true);
  const [enablePIIDetection, setEnablePIIDetection] = useState(true);
  const [hipaaCompliance, setHipaaCompliance] = useState(true);

  return (
    <div className="flex-1 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-foreground">LLM Firewall</h1>
        <Button variant="outline" className="border-border text-foreground">
          Update
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        Control the basic firewall settings
      </p>

      {/* Basic Firewall Settings */}
      <Card className="p-6 bg-card border-border">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-medium text-foreground">Enable Firewall</h3>
              <p className="text-sm text-muted-foreground">Turn on/off main firewall</p>
            </div>
            <Switch
              checked={firewallEnabled}
              onCheckedChange={setFirewallEnabled}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-medium text-foreground">Auto-block Suspicious Activity</h3>
              <p className="text-sm text-muted-foreground">Automatically block IPs with suspicious behavior</p>
            </div>
            <Switch
              checked={autoBlockSuspicious}
              onCheckedChange={setAutoBlockSuspicious}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-medium text-foreground">Enable Secrets Scanning</h3>
              <p className="text-sm text-muted-foreground">Scan content for exposed credentials and tokens</p>
            </div>
            <Switch
              checked={secretsScanning}
              onCheckedChange={setSecretsScanning}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-medium text-foreground">Enable Toxicity Detection</h3>
              <p className="text-sm text-muted-foreground">Automatically scan content for offensive material</p>
            </div>
            <Switch
              checked={toxicityDetection}
              onCheckedChange={setToxicityDetection}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="text-lg font-medium text-foreground">Enable Quarantine System</h3>
              <p className="text-sm text-muted-foreground">Automatically quarantine entries exceeding security thresholds</p>
            </div>
            <Switch
              checked={quarantineSystem}
              onCheckedChange={setQuarantineSystem}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>
        </div>
      </Card>

      {/* Rule Management */}
      <Card className="p-6 bg-card border-border">
        <div className="space-y-6">
          <h3 className="text-lg font-medium text-foreground">Rule Management</h3>
          <p className="text-sm text-muted-foreground">Configure firewall rules</p>

          {/* Default Policy */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h4 className="text-base font-medium text-foreground">Default Policy</h4>
                <p className="text-sm text-muted-foreground">Automatically block IPs with suspicious behavior</p>
              </div>
            </div>
            <Select defaultValue="allow">
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="allow">Allow (Permit unless explicitly blocked)</SelectItem>
                <SelectItem value="block">Block</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Add New Rule */}
          <div className="space-y-4">
            <h4 className="text-base font-medium text-foreground">Add New Rule</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Rule Type</label>
                <Select>
                  <SelectTrigger>
                    <SelectValue placeholder="Allow" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="allow">Allow</SelectItem>
                    <SelectItem value="block">Block</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Pattern Value</label>
                <Input placeholder="E.g. Prompt: Keyword/Topic" />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Description (Optional)</label>
                <Input placeholder="Rule Description" />
              </div>
            </div>
            <Button className="bg-orange-primary hover:bg-orange-dark text-white">
              Add New Rule
            </Button>
          </div>

          {/* Existing Rules */}
          <div className="space-y-4">
            <h4 className="text-base font-medium text-foreground">Allow Rules <span className="text-sm text-muted-foreground">1 rule</span></h4>
            <div className="p-4 border border-border rounded-lg">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <div className="text-sm font-medium text-foreground">organization/acme-corp</div>
                  <div className="text-xs text-muted-foreground">Allow Acme Corporation</div>
                </div>
                <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive">
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>

            <h4 className="text-base font-medium text-foreground">Block Rules <span className="text-sm text-muted-foreground">1 rule</span></h4>
            <div className="p-4 border border-border rounded-lg">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <div className="text-sm font-medium text-foreground">user:anonymous</div>
                  <div className="text-xs text-muted-foreground">Block anonymous users</div>
                </div>
                <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive">
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* PII Detection & HIPAA Compliance */}
      <Card className="p-6 bg-card border-border">
        <div className="space-y-6">
          <h3 className="text-lg font-medium text-foreground">PII Detection & HIPAA Compliance</h3>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h4 className="text-base font-medium text-foreground">Enable Security Scanning</h4>
              <p className="text-sm text-muted-foreground">Scan content for personally identifiable information</p>
            </div>
            <Switch
              checked={securityScanning}
              onCheckedChange={setSecurityScanning}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h4 className="text-base font-medium text-foreground">Enable Masking</h4>
              <p className="text-sm text-muted-foreground">Masking PII</p>
            </div>
            <Switch
              checked={enableMasking}
              onCheckedChange={setEnableMasking}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h4 className="text-base font-medium text-foreground">Enable PII Detection</h4>
              <p className="text-sm text-muted-foreground">Scan content for personally identifiable information</p>
            </div>
            <Switch
              checked={enablePIIDetection}
              onCheckedChange={setEnablePIIDetection}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>

          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h4 className="text-base font-medium text-foreground">HIPAA Compliance Mode</h4>
              <p className="text-sm text-muted-foreground">Enhanced protection for healthcare information</p>
            </div>
            <Switch
              checked={hipaaCompliance}
              onCheckedChange={setHipaaCompliance}
              className="data-[state=checked]:bg-orange-primary"
            />
          </div>
        </div>
      </Card>
    </div>
  );
};
