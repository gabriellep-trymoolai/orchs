import React from 'react';

interface ProgressIndicatorProps {
  steps: string[];
  currentStep: number;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({ steps, currentStep }) => {
  return (
    <div className="flex items-center justify-between mb-8">
      {steps.map((step, index) => (
        <div key={step} className="flex items-center">
          <div className="flex flex-col items-center">
            <div
              className={`w-4 h-4 rounded-full flex items-center justify-center text-xs font-medium ${
                index <= currentStep
                  ? 'bg-orange-primary text-white'
                  : 'bg-muted text-muted-foreground'
              }`}
            />
            <span className="text-xs text-muted-foreground mt-2">{step}</span>
          </div>
          {index < steps.length - 1 && (
            <div className={`w-20 h-0.5 mx-4 ${
              index < currentStep ? 'bg-orange-primary' : 'bg-muted'
            }`} />
          )}
        </div>
      ))}
    </div>
  );
};