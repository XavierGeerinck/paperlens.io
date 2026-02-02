import React, { useState, useEffect } from 'react';
import { Shield, Cpu, Lock, Server, Share2, Activity } from 'lucide-react';
import { useSimulation } from '../../../hooks/useSimulation';
import { SchematicCard } from '../../react/SketchElements';

// Define the architecture modes and their specs
const MODES = {
  plain: { 
    label: 'PLAIN',
    throughput: 100, 
    latency: 10, 
    trust: 'Provider',
    color: 'text-slate-400',
    bgColor: 'bg-slate-400',
    borderColor: 'border-slate-400',
    icon: Server,
    description: 'Unencrypted RAM'
  },
  tee: { 
    label: 'TEE',
    throughput: 80, 
    latency: 20, 
    trust: 'Hardware',
    color: 'text-amber-400',
    bgColor: 'bg-amber-400',
    borderColor: 'border-amber-400', // Amber
    icon: Lock,
    description: 'Trusted Enclave'
  },
  mpc: { 
    label: 'MPC',
    throughput: 40, 
    latency: 100, 
    trust: 'Non-collusion',
    color: 'text-blue-400',
    bgColor: 'bg-blue-400',
    borderColor: 'border-blue-400', // Blue
    icon: Share2,
    description: 'Secret Shares'
  },
  fhe: { 
    label: 'FHE',
    throughput: 5, // Made slightly faster for visualisation purposes (scaled)
    latency: 2000, 
    trust: 'Math',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-400',
    borderColor: 'border-emerald-400', // Emerald/Green
    icon: Shield,
    description: 'Homomorphic'
  }
};

const PrivacyPreservingInferenceSimulation: React.FC = () => {
  // We'll track progress for each lane independently
  // progress: 0 to 100%
  const { isRunning, start, stop, state } = useSimulation({
    initialState: { 
      plain: 0,
      tee: 0,
      mpc: 0,
      fhe: 0
    },
    onTick: (prevState) => {
      // Calculate speed relative to plain baseline
      // Throughput=100 -> speed=2.0 per tick
      // Throughput=5 -> speed=0.1 per tick
      
      const nextState = { ...prevState };
      
      (Object.keys(MODES) as Array<keyof typeof MODES>).forEach(key => {
        const speed = MODES[key].throughput / 30; // Scale factor
        nextState[key] = (prevState[key] + speed) % 100;
      });
      
      return nextState;
    }
  });

  useEffect(() => {
    start();
    return () => stop();
  }, []);

  return (
    <div className="w-full flex flex-col gap-6">
      <SchematicCard title="PRIVACY INFRASTRUCTURE RACE">
        <div className="p-2 flex flex-col gap-2 bg-slate-950 min-h-[400px]">
          
          {/* Header Legend */}
          <div className="grid grid-cols-[80px_1fr_100px] gap-4 px-4 py-2 border-b border-slate-800 text-[10px] uppercase text-slate-500 font-mono">
            <div>Mode</div>
            <div className="flex justify-between px-8">
               <span>Client</span>
               <span>Cloud Processing</span>
               <span>Result</span>
            </div>
            <div className="text-right">Metrics</div>
          </div>

          {/* Lanes */}
          {(Object.keys(MODES) as Array<keyof typeof MODES>).map((key) => {
             const spec = MODES[key];
             const progress = state[key] as number;
             const Icon = spec.icon;

             return (
               <div key={key} className="relative grid grid-cols-[80px_1fr_100px] gap-4 items-center p-4 border border-slate-800/50 rounded bg-slate-900/20 hover:bg-slate-900/40 transition-colors group">
                 
                 {/* Left: Label & Trust */}
                 <div className="flex flex-col gap-1">
                   <div className={`font-mono font-bold text-sm ${spec.color}`}>{spec.label}</div>
                   <div className="text-[10px] text-slate-500 truncate">{spec.trust}</div>
                 </div>

                 {/* Middle: The Pipe */}
                 <div className="relative h-16 flex items-center bg-slate-900/50 rounded-lg border border-slate-800 overflow-hidden">
                   {/* Background Grid */}
                   <div className="absolute inset-0 bg-[linear-gradient(90deg,#1e293b_1px,transparent_1px)] bg-[size:20px_100%] opacity-20" />
                   
                   {/* Center Processing Unit Icon */}
                   <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10 flex flex-col items-center">
                     <div className={`w-8 h-8 rounded flex items-center justify-center bg-slate-900 border ${spec.borderColor}`}>
                        <Icon className={`w-4 h-4 ${spec.color}`} />
                     </div>
                     <span className="text-[9px] text-slate-500 mt-1 bg-slate-950/80 px-1 rounded border border-slate-800/50">{spec.description}</span>
                   </div>

                   {/* User Node */}
                   <div className="absolute left-2 w-2 h-full bg-slate-800 rounded-full opacity-50" />

                   {/* Moving Packets */}
                   {isRunning && Array.from({ length: 3 }).map((_, idx) => {
                      // Create staggered packets
                      const offset = idx * 33; 
                      const localProgress = (progress + offset) % 100;
                      
                      // Opacity fade at edges
                      const opacity = localProgress < 10 ? localProgress / 10 : localProgress > 90 ? (100 - localProgress) / 10 : 1;
                      
                      // Determine content based on progress and mode
                      const getContent = () => {
                        const words = ["DATA", "CODE", "USER"];
                        const word = words[idx % words.length];
                        const isProcessing = localProgress > 40 && localProgress < 60;
                        
                        // Plain: Always visible
                        if (key === 'plain') return word;
                        
                        // TEE: Encrypted in transit, Clear in enclave (40-60%)
                        if (key === 'tee') return isProcessing ? word : '0x...';
                        
                        // MPC: Always Shares
                        if (key === 'mpc') return '[S]';
                        
                        // FHE: Always Encrypted
                        if (key === 'fhe') return '0x...';
                        
                        return word;
                      };

                      const content = getContent();
                      const isEncrypted = content === '0x...' || content === '[S]';

                      return (
                        <div 
                          key={idx}
                          className="absolute top-1/2 -translate-y-1/2 z-20 transition-all duration-200"
                          style={{ 
                            left: `${localProgress}%`,
                            opacity 
                          }}
                        >
                          <div className={`
                            px-1.5 py-0.5 rounded text-[9px] font-mono font-bold border shadow-[0_0_10px_rgba(0,0,0,0.3)]
                            ${isEncrypted ? 'bg-slate-900 border-slate-700 text-slate-500' : `${spec.bgColor} ${spec.borderColor.replace('border', 'text').replace('400', '900')} border-transparent`}
                          `}>
                            {content}
                          </div>
                        </div>
                      );
                   })}
                 </div>

                 {/* Right: Metrics */}
                 <div className="flex flex-col items-end gap-1">
                   <div className={`font-mono text-xs ${spec.color}`}>{spec.throughput}<span className="text-[9px] text-slate-600 ml-1">TPS</span></div>
                   <div className="text-[10px] text-slate-500">{spec.latency}ms</div>
                 </div>

               </div>
             );
          })}

        </div>
      </SchematicCard>
    </div>
  );
};

export default PrivacyPreservingInferenceSimulation;
