import React, { useState, useMemo } from "react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { BrainCircuit, Eye, Zap, RefreshCw, BarChart3, Route, Search } from "lucide-react";

/**
 * JEPASimulation
 * * Demonstrates the core difference between Generative Models (Pixel Reconstruction) 
 * and JEPA (Representation Prediction) in the presence of noise.
 * * MODE 1: Denoising - Shows how JEPA finds signal in noisy single frames.
 * * MODE 2: Planning - Shows how JEPA enables stable long-term planning (World Model)
 *   while generative models drift due to compounding errors.
 */

const JEPASimulation: React.FC = () => {
  // Simulation State
  const [mode, setMode] = useState<'denoise' | 'planning'>('denoise');
  const [noiseLevel, setNoiseLevel] = useState(0.5); // 0 to 1
  const [complexity, setComplexity] = useState(0.3); // Signal frequency
  
  // Planning specific state
  const [planningHorizon, setPlanningHorizon] = useState(30);

  const { isRunning, state, logs, epoch, start, stop, reset } = useSimulation({
    initialState: {
      genLoss: 0,
      jepaLoss: 0,
      // Denoise Mode Data
      genPrediction: [] as number[],
      jepaPrediction: [] as number[],
      targetSignal: [] as number[],
      noisyObservation: [] as number[],
      // Planning Mode Data
      planningGroundTruth: [] as number[],
      planningGenRollout: [] as number[],
      planningJepaRollout: [] as number[],
    },
    tickRate: 100,
    onTick: (prev, tick) => {
      const points = 20;

      // --- COMMON PHYSICS ---
      // A function to get the "Real" world state at any time t
      const getGroundTruth = (t: number) => Math.sin(t * 0.1 * complexity);
      
      // --- MODE 1: DENOISING (Single Step) ---
      if (mode === 'denoise') {
        // 1. Generate Ground Truth (The "Concept")
        const newSignal = Array.from({ length: points }, (_, i) => getGroundTruth(tick + i));

        // 2. Generate Observation (The "Pixels")
        // Add random noise to the signal
        const newObservation = newSignal.map(
          (val) => val + (Math.random() - 0.5) * noiseLevel * 2
        );

        // 3. Model Logic
        // Generative Model: Tries to reconstruct the exact Noisy Observation
        const genPred = newObservation.map(
          (val) => val * 0.9 + (Math.random() - 0.5) * 0.1
        );

        // JEPA Model: Tries to predict the Signal (Representation)
        const jepaPred = newSignal.map((val) => val * 0.95);

        // 4. Calculate Losses
        const genError = newObservation.reduce((acc, val, i) => acc + Math.abs(val - genPred[i]), 0) / points;
        const jepaError = newSignal.reduce((acc, val, i) => acc + Math.abs(val - jepaPred[i]), 0) / points;

        return {
          ...prev,
          targetSignal: newSignal,
          noisyObservation: newObservation,
          genPrediction: genPred,
          jepaPrediction: jepaPred,
          genLoss: genError * (1 + noiseLevel),
          jepaLoss: jepaError,
        };
      } 
      
      // --- MODE 2: PLANNING (Multi-Step Rollout) ---
      else {
        // In planning mode, we simulate a "Mental Rollout" starting from NOW.
        // We want to predict T+1, T+2 ... T+Horizon
        
        // 1. Current State (Start of plan)
        const startState = getGroundTruth(tick);
        // Initial observation is noisy
        let currentGenState = startState + (Math.random() - 0.5) * noiseLevel * 2;
        let currentJepaState = startState; // JEPA encodes to clean latent immediately

        const groundTruthPath = [];
        const genPath = [];
        const jepaPath = [];

        // 2. Rollout Loop
        for (let i = 0; i < planningHorizon; i++) {
          const t = tick + i;
          const truth = getGroundTruth(t);
          groundTruthPath.push(truth);

          // GENERATIVE ROLLOUT (Autoregressive in Pixel Space)
          // It predicts next step based on previous *noisy* prediction
          // Errors specific to Generative Models:
          // 1. Aleatoric Noise injection (thinking it needs to generate texture)
          // 2. Drift (compounding error)
          const genDrift = (Math.random() - 0.5) * noiseLevel * 0.5; // Compounding drift
          // Simple physics approximation with error
          const genNext = currentGenState + (getGroundTruth(t+1) - getGroundTruth(t)) + genDrift; 
          genPath.push(genNext);
          currentGenState = genNext; // Updates for next step (feedback loop)

          // JEPA ROLLOUT (Latent Space)
          // It predicts representations. It ignores noise.
          // It learns the *dynamics* (the sine wave function)
          const jepaNext = getGroundTruth(t+1) * 0.98; // Mild decay/imperfection but stable
          jepaPath.push(jepaNext);
          currentJepaState = jepaNext;
        }

        // Calculate divergence
        // Generative deviation from goal
        const genDivergence = genPath.reduce((acc, val, i) => acc + Math.abs(val - groundTruthPath[i]), 0) / planningHorizon;
        const jepaDivergence = jepaPath.reduce((acc, val, i) => acc + Math.abs(val - groundTruthPath[i]), 0) / planningHorizon;

        return {
          ...prev,
          planningGroundTruth: groundTruthPath,
          planningGenRollout: genPath,
          planningJepaRollout: jepaPath,
          genLoss: genDivergence, // In planning, loss = failure to reach goal state
          jepaLoss: jepaDivergence
        };
      }
    },
  });

  // Visualization Helper
  const renderLine = (data: number[], color: string, strokeWidth = 2, dashed = false) => {
    if (!data || data.length === 0) return "";
    const width = 100;
    const height = 50;
    const step = width / (data.length - 1);

    const points = data
      .map((y, i) => {
        // Clamp for safety
        const clampedY = Math.max(-2, Math.min(2, y)); 
        const normalizedY = ((clampedY + 1.5) / 3) * height;
        return `${i * step},${50 - normalizedY}`;
      })
      .join(" ");

    return (
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={dashed ? "2,1" : "none"}
      />
    );
  };

  return (
    <div className="space-y-6 w-full max-w-4xl mx-auto">
      <SchematicCard title="JEPA SIMULATOR: WORLD MODELS">
        
        {/* MODE SWITCHER */}
        <div className="flex gap-2 mb-4 bg-slate-900/50 p-1 rounded-lg border border-slate-800 w-fit">
           <button 
             onClick={() => setMode('denoise')}
             className={`px-3 py-1.5 text-xs font-bold rounded flex items-center gap-2 transition-all ${
               mode === 'denoise' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
             }`}
           >
             <Eye size={14} /> 1. DENOISING
           </button>
           <button 
             onClick={() => setMode('planning')}
             className={`px-3 py-1.5 text-xs font-bold rounded flex items-center gap-2 transition-all ${
               mode === 'planning' ? 'bg-purple-600 text-white shadow-lg' : 'text-slate-400 hover:text-white'
             }`}
           >
             <Route size={14} /> 2. PLANNING (WORLD MODEL)
           </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* VISUALIZATION PANE */}
          <div className="space-y-4 bg-slate-900/50 p-4 rounded-lg border border-slate-800">
            <div className="flex justify-between items-center text-xs text-slate-400 uppercase tracking-wider">
              <span>
                {mode === 'denoise' ? 'Observation Space (t)' : 'Future Rollout (t...t+n)'}
              </span>
              <span className={mode === 'denoise' ? 'text-red-400' : 'text-purple-400'}>
                {mode === 'denoise' ? 'Pixel Noise' : 'Compounding Error'}
              </span>
            </div>

            {/* MAIN GRAPH */}
            <div className="h-40 w-full bg-black/40 rounded border border-slate-700 relative overflow-hidden">
               {/* Reference Grid */}
               <div className="absolute inset-0 grid grid-cols-6 grid-rows-2">
                 {[...Array(12)].map((_,i) => <div key={i} className="border-r border-b border-slate-800/30"></div>)}
               </div>

               {mode === 'denoise' ? (
                 <>
                  {/* The Noisy Input */}
                  <div className="absolute inset-0 opacity-50">
                    <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full">
                      {renderLine(state.noisyObservation, "#64748b", 1)}
                    </svg>
                  </div>
                  {/* The Underlying Signal */}
                  <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full absolute top-0 left-0">
                    {renderLine(state.targetSignal, "#ffffff", 2, true)}
                  </svg>
                 </>
               ) : (
                  <>
                    {/* PLANNING MODE VISUALS */}
                     <div className="absolute top-2 left-2 text-[10px] text-slate-500 font-mono">
                      Current State (t) -&gt; Future (t+{planningHorizon})
                    </div>
                    {/* The Ideal Path (White Dashed) */}
                    <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full absolute top-0 left-0 opacity-30">
                      {renderLine(state.planningGroundTruth, "#ffffff", 4)}
                    </svg>
                    
                    {/* Generative Path (Pink, erratic) */}
                    <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full absolute top-0 left-0">
                      {renderLine(state.planningGenRollout, "#f472b6", 2)}
                    </svg>

                    {/* JEPA Path (Green, stable) */}
                    <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full absolute top-0 left-0">
                      {renderLine(state.planningJepaRollout, "#34d399", 3)}
                    </svg>
                  </>
               )}
            </div>

            <div className="grid grid-cols-2 gap-4 mt-4">
              {/* GENERATIVE MODEL */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-pink-400 flex items-center gap-2">
                  <Eye size={12} /> GENERATIVE
                </div>
                <div className="h-16 bg-black/40 rounded border border-pink-900/30 relative overflow-hidden">
                   <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full opacity-70">
                      {mode === 'denoise' 
                        ? renderLine(state.genPrediction, "#f472b6")
                        : renderLine(state.planningGenRollout, "#f472b6")
                      }
                   </svg>
                </div>
                <div className="text-[10px] text-pink-300/70">
                  {mode === 'denoise' 
                    ? "Predicts Pixels (Fits Noise)"
                    : "Drifts due to error accumulation"}
                  <br />
                  Loss: <span className="font-mono text-pink-400">{state.genLoss.toFixed(3)}</span>
                </div>
              </div>

              {/* JEPA MODEL */}
              <div className="space-y-2">
                <div className="text-xs font-bold text-emerald-400 flex items-center gap-2">
                  <BrainCircuit size={12} /> JEPA
                </div>
                <div className="h-16 bg-black/40 rounded border border-emerald-900/30 relative overflow-hidden">
                   <svg viewBox="0 0 100 50" preserveAspectRatio="none" className="w-full h-full">
                      {mode === 'denoise' 
                        ? renderLine(state.jepaPrediction, "#34d399")
                        : renderLine(state.planningJepaRollout, "#34d399")
                      }
                   </svg>
                </div>
                <div className="text-[10px] text-emerald-300/70">
                  {mode === 'denoise' 
                    ? "Predicts Reps (Ignores Noise)"
                    : "Stable Planning in Latent Space"}
                  <br />
                  Loss: <span className="font-mono text-emerald-400">{state.jepaLoss.toFixed(3)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* CONTROLS PANE */}
          <div className="space-y-6">
            <div className="space-y-4">
              <div>
                <label className="text-xs font-bold text-slate-400 mb-2 block">
                  ENVIRONMENT NOISE (Aleatoric)
                </label>
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.1"
                  value={noiseLevel}
                  onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
                  className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>Clean</span>
                  <span>Chaos</span>
                </div>
              </div>

              <div>
                <label className="text-xs font-bold text-slate-400 mb-2 block">
                  SIGNAL COMPLEXITY
                </label>
                 <input
                  type="range"
                  min="0.1"
                  max="0.8"
                  step="0.1"
                  value={complexity}
                  onChange={(e) => setComplexity(parseFloat(e.target.value))}
                  className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                />
              </div>

              {mode === 'planning' && (
                 <div>
                  <label className="text-xs font-bold text-slate-400 mb-2 block">
                    PLANNING HORIZON (Steps)
                  </label>
                   <input
                    type="range"
                    min="10"
                    max="50"
                    step="5"
                    value={planningHorizon}
                    onChange={(e) => setPlanningHorizon(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-orange-500"
                  />
                  <div className="text-right text-[10px] text-orange-400 mt-1">{planningHorizon} steps</div>
                </div>
              )}
            </div>

            <div className="bg-blue-900/20 p-4 rounded border border-blue-800/50 text-sm text-blue-200">
              <h4 className="font-bold flex items-center gap-2 mb-2">
                <Zap size={14} /> 
                {mode === 'denoise' ? 'The "Generative Trap"' : 'The Power of World Models'}
              </h4>
             
              {mode === 'denoise' ? (
                 <p className="opacity-80 text-xs leading-relaxed">
                   When you ask a model to predict <strong>pixels</strong>, it must predict the noise (static/rain/leaves) to get a low loss. 
                   <br/><br/>
                   JEPA ignores the noise entirely, focusing only on the underlying <strong>representation</strong> (the signal).
                 </p>
              ) : (
                <p className="opacity-80 text-xs leading-relaxed">
                   To plan, an agent must imagine the future. If it hallucinates noise (Generative), the plan deviates (Pink line).
                   <br/><br/>
                   JEPA (Green) plans in <strong>Latent Space</strong>, ignoring noise to predict the stable <i>outcome</i> of actions.
                </p>
              )}
            </div>

            <div className="flex gap-2">
              <SchematicButton
                onClick={isRunning ? stop : start}
                icon={isRunning ? Zap : Zap}
                label={isRunning ? "PAUSE SIM" : "RUN SIM"}
                active={isRunning}
              />
              <SchematicButton onClick={reset} icon={RefreshCw} label="RESET" />
            </div>
          </div>
        </div>
      </SchematicCard>
    </div>
  );
};

export default JEPASimulation;
