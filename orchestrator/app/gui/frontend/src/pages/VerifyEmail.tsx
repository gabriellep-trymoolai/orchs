import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthCard } from '@/components/AuthCard';
import { AuthInput } from '@/components/AuthInput';
import { Button } from '@/components/ui/button';
import { InputOTP, InputOTPGroup, InputOTPSlot } from '@/components/ui/input-otp';

export const VerifyEmail: React.FC = () => {
  const navigate = useNavigate();
  const [email] = useState('aisvarya@gmail.com');
  const [otp, setOtp] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    navigate('/profile-setup');
  };

  return (
    <AuthCard>
      <div className="text-center mb-8">
        <h1 className="text-2xl font-semibold text-foreground mb-2">Verify your email</h1>
        <p className="text-sm text-muted-foreground">Enter the one-time code to create an account</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <AuthInput
          label="Email ID"
          type="email"
          value={email}
          readOnly
          className="bg-muted"
        />

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Verification Code</label>
          <div className="flex justify-center">
            <InputOTP
              maxLength={6}
              value={otp}
              onChange={setOtp}
              className="gap-2"
            >
              <InputOTPGroup className="gap-2">
                {[...Array(6)].map((_, index) => (
                  <InputOTPSlot
                    key={index}
                    index={index}
                    className="w-12 h-12 bg-input border-border text-foreground text-center rounded-lg"
                  />
                ))}
              </InputOTPGroup>
            </InputOTP>
          </div>
        </div>

        <Button 
          type="submit" 
          className="w-full h-12 bg-orange-primary hover:bg-orange-dark text-white font-medium rounded-lg"
        >
          Verify Email
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