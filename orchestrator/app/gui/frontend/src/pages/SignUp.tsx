import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthCard } from '@/components/AuthCard';
import { AuthInput } from '@/components/AuthInput';
import { Button } from '@/components/ui/button';
import { Eye, EyeOff } from 'lucide-react';

export const SignUp: React.FC = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Simulate form validation
    if (!formData.email.includes('@')) {
      setErrors({ email: 'Invalid Email ID' });
      return;
    }
    navigate('/verify-email');
  };

  return (
    <AuthCard>
      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-foreground mb-2">Let's Get Started</h1>
        <p className="text-sm text-muted-foreground">Enter the details below to sign up</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <AuthInput
          label="Email ID"
          type="email"
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          placeholder="aisvarya@gmail.com"
          error={errors.email}
          showError={!!errors.email}
        />

        <div className="space-y-2">
          <AuthInput
            label="Password"
            type={showPassword ? 'text' : 'password'}
            value={formData.password}
            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
            placeholder="••••••••"
          />
          <button
            type="button"
            onClick={() => setShowPassword(!showPassword)}
            className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
            style={{ marginTop: '1.5rem' }}
          >
            {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
          </button>
        </div>

        <Button 
          type="submit" 
          className="w-full h-12 bg-orange-primary hover:bg-orange-dark text-white font-medium rounded-lg"
        >
          Create Account
        </Button>
      </form>

      <div className="text-center mt-6">
        <p className="text-sm text-muted-foreground">
          Already have an account?{' '}
          <Link to="/login" className="text-orange-primary hover:text-orange-light">
            Log in
          </Link>
        </p>
      </div>

      <div className="relative my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t border-border" />
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="bg-card px-3 text-muted-foreground">Or</span>
        </div>
      </div>

      <Button 
        variant="outline" 
        className="w-full h-12 bg-secondary border-border text-foreground hover:bg-input"
      >
        <div className="flex items-center gap-3">
          <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center">
            <span className="text-xs font-bold text-blue-600">G</span>
          </div>
          Sign up with Google
        </div>
      </Button>
    </AuthCard>
  );
};