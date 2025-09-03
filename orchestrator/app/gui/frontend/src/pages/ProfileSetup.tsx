import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthCard } from '@/components/AuthCard';
import { AuthInput } from '@/components/AuthInput';
import { ProgressIndicator } from '@/components/ProgressIndicator';
import { Button } from '@/components/ui/button';

export const ProfileSetup: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/llm-setup');
  };

  const handleBack = () => {
    navigate('/verify-email');
  };

  return (
    <AuthCard>
      <ProgressIndicator 
        steps={['Set up Profile', 'Set up LLM']} 
        currentStep={0}
      />

      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-foreground">Set up your Profile</h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <AuthInput
          label="First Name"
          type="text"
          value={formData.firstName}
          onChange={(e) => setFormData({ ...formData, firstName: e.target.value })}
          placeholder=""
        />

        <AuthInput
          label="Last Name"
          type="text"
          value={formData.lastName}
          onChange={(e) => setFormData({ ...formData, lastName: e.target.value })}
          placeholder=""
        />

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
            Next
          </Button>
        </div>
      </form>
    </AuthCard>
  );
};