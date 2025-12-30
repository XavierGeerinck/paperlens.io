import React, { useRef, useEffect } from 'react';
import { Play, RotateCcw, Activity, Terminal, Cpu, Database, BrainCircuit, Zap, Variable, Pause } from 'lucide-react';
import { useSimulation } from '../hooks/useSimulation';

interface DemoViewProps {
  ideaId: string;
}

// --- Simulation Logic Definitions ---

interface BrainMimeticState {
  loss: number;
  surprise: number;
  memory: number;
  alpha: number; // Forget Gate
  eta: number;   // Momentum
  theta: number; // Learning Rate
}

const BRAIN_MIMETIC_INIT: BrainMimeticState = {
  loss: 2.5,
  surprise: 0.8,
  memory: 0,
  alpha: 0.1,
  eta: 0.9,
  theta: 0.01
};

const brainMimeticTick = (prev: BrainMimeticState, tick: number): Partial<BrainMimeticState> => {
  const newLoss = Math.max(0.1, prev.loss * 0.95 + (Math.random() * 0.1 - 0.05));
  const newSurprise = Math.max(0, prev.surprise * 0.9 + (Math.random() * 0.2));
  const newMemory = Math.min(100, prev.memory + 0.8);
  
  // Dynamic parameters based on "surprise"
  // When surprise is high, theta (learning rate) increases
  const newTheta = 0.01 + (newSurprise * 0.05);
  // When surprise is high, alpha (forgetting) might increase to make room, or decrease to retain? 
  // Let's say high surprise = plastic = update weights
  const newAlpha = 0.1 + (Math.sin(tick / 10) * 0.05); 
  const newEta = 0.9 - (newSurprise * 0.1);

  return { 
      loss: newLoss, 
      surprise: newSurprise, 
      memory: newMemory,
      alpha: newAlpha,
      eta: newEta,
      theta: newTheta
  };
};

const brainMimeticLog = (state: BrainMimeticState): string | null => {
   if (Math.random() > 0.6) {
       const messages = [
           `[TITANS] Surprise Gradient: ${state.surprise.toFixed(4)}`,
           `[MEMORY] Encoding pattern: 0x${Math.floor(Math.random()*16777215).toString(16)}`,
           `[PLASTICITY] Weight update. Theta: ${state.theta.toFixed(3)}`,
           `[ATTENTION] Retrieving context block`,
           `[OPTIMIZER] Backprop completed.`
       ];
       return messages[Math.floor(Math.random() * messages.length)];
   }
   return null;
};

// --- Components ---

const VariableWatch: React.FC<{ name: string; value: number; color?: string }> = ({ name, value, color = "text-zinc-400" }) => (
    <div className="flex items-center justify-between font-mono text-xs py-1 border-b border-zinc-800/50 last:border-0">
        <span className={color}>{name}</span>
        <span className="text-zinc-200">{value.toFixed(4)}</span>
    </div>
);

const DemoView: React.FC<DemoViewProps> = ({ ideaId }) => {
  const isBrainMimetic = ideaId === 'brain-mimetic-llm';
  const logsContainerRef = useRef<HTMLDivElement>(null);

  const { 
      isRunning, 
      state, 
      logs, 
      history, 
      epoch, 
      start, 
      stop, 
      reset 
  } = useSimulation<BrainMimeticState>({
      initialState: BRAIN_MIMETIC_INIT,
      onTick: brainMimeticTick,
      onLog: brainMimeticLog,
      tickRate: 200
  });

  // Use scrollTop for robust auto-scrolling that doesn't affect the viewport
  useEffect(() => {
    if (logsContainerRef.current) {
        logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  }, [logs]);

  if (!isBrainMimetic) {
      return (
          <div className="flex items-center justify-center h-96 border border-zinc-800 bg-zinc-900/50 rounded-xl">
              <div className="text-center text-zinc-500">
                  <Activity className="w-12 h-12 mx-auto mb-4 opacity-20" />
                  <p>Interactive simulation not available for this project.</p>
              </div>
          </div>
      );
  }

  // Helper to render SVG line path
  const renderPath = (data: number[] | undefined, max: number, color: string) => {
    if (!data || data.length < 2) return '';
    const width = 100;
    const height = 40;
    const step = width / (data.length - 1);
    
    const points = data.map((val, i) => {
        const x = i * step;
        const y = height - (Math.min(val, max) / max) * height; // Invert Y
        return `${x},${y}`;
    }).join(' ');

    return <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />;
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-500">
        {/* Left Column: Controls & Graphs */}
        <div className="lg:col-span-2 space-y-6">
            
            {/* Control Panel */}
            <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 relative overflow-hidden backdrop-blur-sm">
                <div className="flex items-center justify-between mb-8">
                    <h3 className="text-white font-mono uppercase tracking-widest flex items-center gap-2">
                        <Cpu className="w-5 h-5 text-indigo-500" />
                        Training Environment
                    </h3>
                    <div className="flex gap-2">
                        {!isRunning ? (
                            <button onClick={start} className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-mono uppercase tracking-widest transition-colors shadow-lg shadow-indigo-900/20">
                                <Play className="w-3 h-3" /> Initiate
                            </button>
                        ) : (
                            <button onClick={stop} className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-xs font-mono uppercase tracking-widest transition-colors border border-zinc-700">
                                <Pause className="w-3 h-3 text-yellow-500" /> Pause
                            </button>
                        )}
                        <button onClick={reset} className="p-2 border border-zinc-700 hover:border-zinc-500 text-zinc-400 hover:text-white rounded transition-colors" title="Reset Simulation">
                            <RotateCcw className="w-4 h-4" />
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                    {/* Graph 1: Loss */}
                    <div className="bg-black/40 border border-zinc-800 p-4 rounded-lg relative group">
                        <div className="text-[10px] font-mono text-zinc-500 uppercase mb-2 flex justify-between">
                            <span>Loss Function (MSE)</span>
                            <span className="text-white font-bold">{state.loss.toFixed(4)}</span>
                        </div>
                        <div className="h-24 w-full relative">
                            <div className="absolute inset-0 grid grid-cols-6 gap-0 opacity-10 pointer-events-none">
                                {[...Array(6)].map((_, i) => <div key={i} className="border-r border-zinc-500 h-full" />)}
                            </div>
                            <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="w-full h-full overflow-visible">
                                {renderPath(history['loss'], 3, '#f43f5e')} 
                            </svg>
                        </div>
                    </div>

                     {/* Graph 2: Surprise */}
                     <div className="bg-black/40 border border-zinc-800 p-4 rounded-lg relative group">
                        <div className="text-[10px] font-mono text-zinc-500 uppercase mb-2 flex justify-between">
                            <span>Surprise Metric (∇L)</span>
                            <span className="text-white font-bold">{state.surprise.toFixed(4)}</span>
                        </div>
                         <div className="h-24 w-full relative">
                            <div className="absolute inset-0 grid grid-cols-6 gap-0 opacity-10 pointer-events-none">
                                {[...Array(6)].map((_, i) => <div key={i} className="border-r border-zinc-500 h-full" />)}
                            </div>
                            <svg viewBox="0 0 100 40" preserveAspectRatio="none" className="w-full h-full overflow-visible">
                                {renderPath(history['surprise'], 1.5, '#22d3ee')} 
                            </svg>
                        </div>
                    </div>
                </div>

                <div className="mt-6 flex items-center gap-4 text-xs font-mono text-zinc-500 border-t border-zinc-800/50 pt-4">
                    <div className="flex items-center gap-2 min-w-[140px]">
                        <BrainCircuit className="w-4 h-4 text-indigo-500" />
                        Memory Load: {state.memory.toFixed(1)}%
                    </div>
                    <div className="flex-grow h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-indigo-600 to-purple-500 transition-all duration-200" style={{ width: `${state.memory}%` }} />
                    </div>
                    <div className="text-zinc-400">EPOCH {epoch.toString().padStart(4, '0')}</div>
                </div>
            </div>

            {/* Live Code & Variables Watcher */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Code View */}
                <div className="md:col-span-2 bg-[#0c0c0e] border border-zinc-800 rounded-xl p-6 font-mono text-xs overflow-hidden relative">
                    <div className="text-zinc-500 mb-4 flex items-center gap-2 border-b border-zinc-800 pb-2">
                        <Database className="w-4 h-4" />
                        LIVE KERNEL
                    </div>
                    <div className="text-green-500/80 space-y-1">
                        <div>class NeuralMemory(nn.Module):</div>
                        <div className="pl-4">def forward(self, x, state):</div>
                        <div className="pl-8 text-zinc-500"># Real-time weight update</div>
                        <div className="pl-8">M, S = state</div>
                        <div className="pl-8">
                            <span className="text-purple-400">surprise</span> = grad(loss, M)
                        </div>
                        <div className="pl-8 flex flex-wrap gap-x-1 items-center">
                            <span>M = (1 - </span>
                            <span className="text-blue-400 font-bold bg-blue-400/10 px-1 rounded flex items-center gap-1">
                                alpha
                                <span className="text-[9px] opacity-70">[{state.alpha.toFixed(3)}]</span>
                            </span>
                            <span>) * M + </span>
                            <span className="text-yellow-400 font-bold bg-yellow-400/10 px-1 rounded flex items-center gap-1">
                                eta
                                <span className="text-[9px] opacity-70">[{state.eta.toFixed(3)}]</span>
                            </span>
                            <span> * S</span>
                        </div>
                        <div className="pl-16 flex flex-wrap gap-x-1 items-center">
                             <span>- </span>
                             <span className="text-red-400 font-bold bg-red-400/10 px-1 rounded flex items-center gap-1">
                                theta
                                <span className="text-[9px] opacity-70">[{state.theta.toFixed(3)}]</span>
                             </span>
                             <span> * </span>
                             <span className="text-purple-400 font-bold bg-purple-400/10 px-0.5 rounded">surprise</span>
                        </div>
                        <div className="pl-8">return M</div>
                    </div>
                </div>

                {/* Variable Watcher */}
                <div className="md:col-span-1 bg-[#0c0c0e] border border-zinc-800 rounded-xl p-4 font-mono">
                    <div className="text-zinc-500 mb-4 flex items-center gap-2 border-b border-zinc-800 pb-2">
                        <Variable className="w-4 h-4" />
                        WATCH
                    </div>
                    <div className="space-y-1">
                        <VariableWatch name="loss" value={state.loss} color="text-red-400" />
                        <VariableWatch name="surprise" value={state.surprise} color="text-purple-400" />
                        <VariableWatch name="alpha" value={state.alpha} color="text-blue-400" />
                        <VariableWatch name="eta" value={state.eta} color="text-yellow-400" />
                        <VariableWatch name="theta" value={state.theta} color="text-red-400" />
                    </div>
                </div>
            </div>
        </div>

        {/* Right Column: Logs */}
        <div className="lg:col-span-1 bg-black border border-zinc-800 rounded-xl p-4 font-mono text-xs flex flex-col h-[600px] shadow-2xl">
            <div className="flex items-center justify-between text-zinc-500 border-b border-zinc-900 pb-2 mb-2">
                <div className="flex items-center gap-2">
                    <Terminal className="w-4 h-4" />
                    <span>/var/log/titans.log</span>
                </div>
                <div className="flex gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500/20" />
                    <div className="w-2 h-2 rounded-full bg-yellow-500/20" />
                    <div className="w-2 h-2 rounded-full bg-green-500/20" />
                </div>
            </div>
            <div 
                ref={logsContainerRef}
                className="flex-grow overflow-y-auto space-y-1 text-zinc-400 scrollbar-hide font-mono leading-tight"
            >
                {logs.map((log, i) => (
                    <div key={i} className={`break-all ${log.includes('[PLASTICITY]') ? 'text-yellow-500' : log.includes('[TITANS]') ? 'text-indigo-400' : 'text-zinc-500'}`}>
                        <span className="text-zinc-700 select-none mr-2">
                            {new Date().toLocaleTimeString().split(' ')[0]}
                        </span>
                        {log}
                    </div>
                ))}
            </div>
             <div className="mt-2 pt-2 border-t border-zinc-900 flex items-center justify-between text-zinc-600">
                <span className="flex items-center gap-2">
                    <Zap className="w-3 h-3 text-green-900" />
                    Kernel: 5.15.0-generic
                </span>
                <span className="animate-pulse">_</span>
            </div>
        </div>
    </div>
  );
};

export default DemoView;