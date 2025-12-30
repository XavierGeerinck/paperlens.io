import React from 'react';

interface SketchProps {
  className?: string;
  color?: string;
}

export const SketchCircle: React.FC<{ children: React.ReactNode; className?: string; color?: string }> = ({ children, className = "", color = "currentColor" }) => (
  <div className={`relative inline-block ${className}`}>
    <span className="relative z-10">{children}</span>
    <svg className="absolute inset-0 w-[110%] h-[120%] -left-[5%] -top-[10%] pointer-events-none overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none">
      <path
        d="M5,50 C15,5 85,5 95,50 C95,95 15,95 5,50 Z"
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeDasharray="400 0"
        style={{
             vectorEffect: 'non-scaling-stroke',
             filter: 'url(#pencilFilter)',
             d: "M4,50 Q20,2 90,8 Q98,40 92,85 Q60,98 10,88 Q-2,60 4,50 M6,48 Q18,4 88,10" // Rough circle path
        }}
      />
    </svg>
  </div>
);

export const SketchBox: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className = "" }) => (
    <div className={`relative ${className}`}>
        <div className="absolute inset-0 border-2 border-zinc-700/50 rounded-sm transform rotate-1 pointer-events-none" 
             style={{ clipPath: 'polygon(0% 0%, 100% 2%, 98% 100%, 2% 98%)' }}></div>
        <div className="absolute inset-0 border-2 border-zinc-700/50 rounded-sm transform -rotate-1 pointer-events-none" 
             style={{ clipPath: 'polygon(2% 2%, 98% 0%, 100% 98%, 0% 100%)' }}></div>
        <div className="relative z-10">{children}</div>
    </div>
);

export const SketchArrowRight: React.FC<SketchProps> = ({ className = "", color = "currentColor" }) => (
    <svg className={`overflow-visible ${className}`} width="60" height="20" viewBox="0 0 60 20">
        <path d="M5,10 C20,12 40,8 55,10 M45,5 C50,8 55,10 55,10 C50,12 45,15 45,15" 
              fill="none" 
              stroke={color} 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round" />
    </svg>
);

export const SketchUnderline: React.FC<{ className?: string, color?: string }> = ({ className = "", color = "currentColor" }) => (
    <svg className={`absolute bottom-0 left-0 w-full h-3 overflow-visible ${className}`} preserveAspectRatio="none" viewBox="0 0 100 10">
        <path d="M0,5 Q50,10 100,5" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
    </svg>
);

export const SketchHighlight: React.FC<{ className?: string, color?: string }> = ({ className = "", color = "rgba(255, 255, 0, 0.2)" }) => (
    <svg className={`absolute inset-0 w-full h-full -z-10 overflow-visible ${className}`} preserveAspectRatio="none" viewBox="0 0 100 100">
        <path d="M2,20 L98,15 L95,85 L5,90 Z" fill={color} />
    </svg>
);

// Add this to your root component or index.html to enable filters
export const SketchFilters = () => (
    <svg style={{ position: 'absolute', width: 0, height: 0 }}>
        <defs>
            <filter id="pencilFilter">
                <feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="3" result="noise" />
                <feDisplacementMap in="SourceGraphic" in2="noise" scale="2" />
            </filter>
        </defs>
    </svg>
);