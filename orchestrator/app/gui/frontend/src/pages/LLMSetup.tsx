import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthCard } from '@/components/AuthCard';
import { AuthInput } from '@/components/AuthInput';
import { ProgressIndicator } from '@/components/ProgressIndicator';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Eye, EyeOff, HelpCircle } from 'lucide-react';

export const LLMSetup: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    llm: 'OpenAI',
    apiKey: ''
  });
  const [showApiKey, setShowApiKey] = useState(false);
  const [isValidKey, setIsValidKey] = useState(false);

  const handleApiKeyChange = (value: string) => {
    setFormData({ ...formData, apiKey: value });
    // Simulate API key validation
    setIsValidKey(value.length > 10);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/dashboard');
  };

  const handleBack = () => {
    navigate('/profile-setup');
  };

  return (
    <AuthCard>
      <ProgressIndicator 
        steps={['Set up Profile', 'Set up LLM']} 
        currentStep={1}
      />

      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-foreground">Set up your LLM</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">
            Choose your preferred language model
          </label>
          <Select value={formData.llm} onValueChange={(value) => setFormData({ ...formData, llm: value })}>
            <SelectTrigger className="h-12 bg-input border-border text-foreground">
              <div className="flex items-center gap-2">
                <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
                  <span className="text-xs font-bold text-black">AI</span>
                </div>
                <SelectValue placeholder="Select LLM" />
              </div>
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              <SelectItem value="OpenAI" className="text-foreground">OpenAI</SelectItem>
              <SelectItem value="Claude" className="text-foreground">Claude</SelectItem>
              <SelectItem value="Gemini" className="text-foreground">Gemini</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-sm font-medium text-foreground">Enter API key</label>
            <HelpCircle className="w-4 h-4 text-muted-foreground" />
          </div>
          <div className="relative">
            <AuthInput
              label=""
              type={showApiKey ? 'text' : 'password'}
              value={formData.apiKey}
              onChange={(e) => handleApiKeyChange(e.target.value)}
              placeholder="AUDBDKGS3473"
              className="pr-10"
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>
          
          <div className="flex justify-between items-center">
            {isValidKey && (
              <p className="text-sm text-green-500 flex items-center gap-1">
                <span>✓</span> Valid Key
              </p>
            )}
            <a href="#" className="text-sm text-orange-primary hover:text-orange-light ml-auto">
              How to get your OpenAI key?
            </a>
          </div>
        </div>

        <div className="flex gap-4 pt-4">
          <Button 
            type="button"
            variant="outline"
            onClick={handleBack}
            className="flex-1 h-12 bg-secondary border-border text-foreground hover:bg-input rounded-full"
          >
            Back
          </Button>
          <Button 
            type="submit" 
            className="flex-1 h-12 bg-orange-primary hover:bg-orange-dark text-white font-medium rounded-full"
          >
            Start using MoolAI
          </Button>
        </div>
      </form>
    </AuthCard>
  );
};