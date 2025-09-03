import React from 'react';

interface MoolaiLogoProps {
  className?: string;
  showText?: boolean;
}

export const MoolaiLogo: React.FC<MoolaiLogoProps> = ({ className = "", showText = true }) => {
  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="relative w-8 h-8 flex items-center justify-center">
        <svg
          width="32"
          height="32"
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#FF6B35" />
              <stop offset="100%" stopColor="#E63946" />
            </linearGradient>
          </defs>
          {/* Main chat bubble shape */}
          <path
            d="M20 20 Q20 10 30 10 L70 10 Q80 10 80 20 L80 50 Q80 60 70 60 L40 60 Q35 60 30 65 L25 70 Q20 75 20 70 L20 20 Z"
            fill="url(#logoGradient)"
          />
          {/* Inner curved elements */}
          <path
            d="M30 25 Q35 20 45 25 Q50 30 45 35 Q35 40 30 35 Q25 30 30 25 Z"
            fill="#1F2937"
          />
          <path
            d="M50 35 Q55 30 65 35 Q70 40 65 45 Q55 50 50 45 Q45 40 50 35 Z"
            fill="#1F2937"
          />
        </svg>
      </div>
      {showText && (
        <span className="text-xl font-semibold text-foreground tracking-wide">MoolAI</span>
      )}
    </div>
  );
};