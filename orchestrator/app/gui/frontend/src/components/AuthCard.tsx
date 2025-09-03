import React from 'react';
import { MoolaiLogo } from './MoolaiLogo';

interface AuthCardProps {
  children: React.ReactNode;
  className?: string;
}

export const AuthCard: React.FC<AuthCardProps> = ({ children, className = "" }) => {
  return (
    <div className={`min-h-screen bg-gradient-primary flex items-center justify-center p-6 ${className}`}>
      <div className="w-full max-w-md bg-card border border-border rounded-2xl p-8 shadow-2xl">
        <div className="flex justify-center mb-8">
          <MoolaiLogo />
        </div>
        {children}
      </div>
    </div>
  );
};