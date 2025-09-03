import React, { forwardRef } from 'react';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';

interface AuthInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  showError?: boolean;
}

export const AuthInput = forwardRef<HTMLInputElement, AuthInputProps>(
  ({ label, error, showError, className, ...props }, ref) => {
    return (
      <div className="space-y-2">
        <Label className="text-sm font-medium text-foreground">{label}</Label>
        <Input
          ref={ref}
          className={cn(
            "h-12 bg-input border-border text-foreground placeholder:text-muted-foreground",
            "focus:border-orange-primary focus:ring-orange-primary",
            showError && error && "border-destructive focus:border-destructive focus:ring-destructive",
            className
          )}
          {...props}
        />
        {showError && error && (
          <p className="text-sm text-destructive mt-1">{error}</p>
        )}
      </div>
    );
  }
);

AuthInput.displayName = "AuthInput";