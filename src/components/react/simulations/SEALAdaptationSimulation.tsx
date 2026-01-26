import type React from "react";
import { useState, useRef, useEffect } from "react";
import { Play, Pause, RotateCcw, Zap, Database, GitBranch, TrendingUp } from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface SelfEdit {
  id: number;
  syntheticData: string[];
  learningRate: number;
  epochs: number;
  reward: number;
  selected: boolean;
}

interface SimulationState extends Record<string, unknown> {
  iteration: number;
  numEdits: number;
  topK: number;
  edits: SelfEdit[];
  averageReward: number;
  policyAccuracy: number;
  knowledgeRetention: number;
}

const SEALAdaptationSimulation: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [numEditsPerTask, setNumEditsPerTask] = useState(8);
  const [topK, setTopK] = useState(3);

  const { isRunning, state, history, epoch, start, stop, reset } = useSimulation<SimulationState>({
    initialState: {
      iteration: 0,
      numEdits: numEditsPerTask,
      topK: topK,
      edits: [],
      averageReward: 0,
      policyAccuracy: 32.7, // Starting baseline
      knowledgeRetention: 100,
    },
    onTick: (prev, tick) => {
      // Generate new self-edits every 10 ticks (represents new RL iteration)
      if (tick % 10 === 0) {
        const newIteration = Math.floor(tick / 10);
        
        // Generate candidate self-edits with varying quality
        const edits: SelfEdit[] = Array.from({ length: prev.numEdits }, (_, i) => {
          // Early iterations: more variance, lower average reward
          // Later iterations: policy improves, higher rewards
          const iterationBonus = newIteration * 0.05;
          const baseReward = 0.3 + Math.random() * 0.4 + iterationBonus;
          const noisyReward = Math.max(0, Math.min(1, baseReward + (Math.random() - 0.5) * 0.2));
          
          return {
            id: tick * 100 + i,
            syntheticData: [
              `Q: What is fact ${i}? A: Answer ${i}`,
              `The concept ${i} relates to topic ${i % 3}`,
            ],
            learningRate: 1e-5 + Math.random() * 1e-4,
            epochs: 2 + Math.floor(Math.random() * 4),
            reward: noisyReward,
            selected: false,
          };
        });

        // Sort by reward and select top-k
        const sorted = [...edits].sort((a, b) => b.reward - a.reward);
        sorted.forEach((edit, idx) => {
          edit.selected = idx < prev.topK;
        });

        const selectedEdits = sorted.filter(e => e.selected);
        const avgReward = selectedEdits.reduce((sum, e) => sum + e.reward, 0) / selectedEdits.length;

        // Policy improves based on average reward of selected edits
        // Knowledge retention decreases slightly with each adaptation
        const policyImprovement = (avgReward - 0.5) * 5; // Scale reward to accuracy gain
        const newAccuracy = Math.min(47, prev.policyAccuracy + policyImprovement);
        const forgettingRate = 0.5; // Small knowledge decay per iteration
        const newRetention = Math.max(70, prev.knowledgeRetention - forgettingRate);

        return {
          ...prev,
          iteration: newIteration,
          edits: sorted,
          averageReward: avgReward,
          policyAccuracy: newAccuracy,
          knowledgeRetention: newRetention,
        };
      }

      return prev;
    },
    onLog: (state) => {
      if (state.iteration > 0) {
        return `RL Iteration ${state.iteration}: Avg Reward=${state.averageReward.toFixed(3)}, Policy Acc=${state.policyAccuracy.toFixed(1)}%`;
      }
      return null;
    },
    tickRate: 200,
  });

  // Visualize the SEAL training loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    // Clear canvas
    ctx.fillStyle = "rgba(0, 0, 0, 0.9)";
    ctx.fillRect(0, 0, rect.width, rect.height);

    // Draw SEAL loop visualization
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const radius = Math.min(rect.width, rect.height) * 0.35;

    // Draw circular RL loop
    ctx.strokeStyle = "rgba(59, 130, 246, 0.3)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw 4 stages of SEAL loop
    const stages = [
      { label: "Generate", icon: "📝", angle: 0, color: "#3b82f6" },
      { label: "Apply SFT", icon: "⚙️", angle: Math.PI / 2, color: "#10b981" },
      { label: "Evaluate", icon: "📊", angle: Math.PI, color: "#f59e0b" },
      { label: "Reinforce", icon: "🎯", angle: (3 * Math.PI) / 2, color: "#8b5cf6" },
    ];

    stages.forEach((stage, idx) => {
      const x = centerX + radius * Math.cos(stage.angle - Math.PI / 2);
      const y = centerY + radius * Math.sin(stage.angle - Math.PI / 2);

      // Draw stage circle
      ctx.fillStyle = stage.color;
      ctx.beginPath();
      ctx.arc(x, y, 30, 0, Math.PI * 2);
      ctx.fill();

      // Highlight current stage
      if (isRunning && Math.floor(epoch / 2.5) % 4 === idx) {
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, 35, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Draw label
      ctx.fillStyle = "#fff";
      ctx.font = "12px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText(stage.label, x, y + 55);
    });

    // Draw self-edits as particles
    if (state.edits.length > 0) {
      state.edits.forEach((edit, idx) => {
        const angle = (idx / state.edits.length) * Math.PI * 2;
        const distance = radius * 0.6;
        const x = centerX + distance * Math.cos(angle);
        const y = centerY + distance * Math.sin(angle);

        // Color based on reward
        const hue = edit.reward * 120; // 0 (red) to 120 (green)
        ctx.fillStyle = edit.selected 
          ? `hsla(${hue}, 100%, 60%, 0.9)` 
          : `hsla(${hue}, 50%, 40%, 0.3)`;
        
        const size = edit.selected ? 8 : 5;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();

        // Draw selection glow
        if (edit.selected) {
          ctx.strokeStyle = `hsla(${hue}, 100%, 70%, 0.5)`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, size + 3, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    }

    // Draw iteration counter in center
    ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
    ctx.font = "bold 24px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Round ${state.iteration}`, centerX, centerY - 10);
    
    ctx.font = "12px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(148, 163, 184, 0.8)";
    ctx.fillText(`ReST-EM`, centerX, centerY + 10);

  }, [state, isRunning, epoch]);

  // Performance chart
  const performanceHistory = history.policyAccuracy?.slice(-30) || [];
  const retentionHistory = history.knowledgeRetention?.slice(-30) || [];

  return (
    <div className="w-full space-y-4">
      <SchematicCard title="SEAL: SELF-ADAPTING LLM TRAINING">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Main Visualization */}
          <div className="space-y-4">
            <div className="bg-black/40 rounded-lg p-4 border border-slate-700/50">
              <canvas
                ref={canvasRef}
                className="w-full"
                style={{ height: "400px" }}
              />
            </div>

            {/* Controls */}
            <div className="flex gap-2">
              {!isRunning ? (
                <SchematicButton onClick={start} className="flex-1">
                  <Play size={16} />
                  START RL LOOP
                </SchematicButton>
              ) : (
                <SchematicButton onClick={stop} className="flex-1">
                  <Pause size={16} />
                  PAUSE
                </SchematicButton>
              )}
              <SchematicButton onClick={reset}>
                <RotateCcw size={16} />
                RESET
              </SchematicButton>
            </div>

            {/* Configuration */}
            <div className="bg-slate-900/50 rounded-lg p-4 border border-slate-700/50 space-y-3">
              <div>
                <div className="flex items-center justify-between text-[10px] uppercase font-mono text-slate-400 mb-2">
                  <span>EDITS PER TASK: {numEditsPerTask}</span>
                  <Zap size={12} className="text-blue-400" />
                </div>
                <input
                  type="range"
                  min="4"
                  max="16"
                  value={numEditsPerTask}
                  onChange={(e) => setNumEditsPerTask(Number(e.target.value))}
                  className="w-full"
                  disabled={isRunning}
                  aria-label="Edits per task"
                />
              </div>

              <div>
                <div className="flex items-center justify-between text-[10px] uppercase font-mono text-slate-400 mb-2">
                  <span>TOP-K SELECTION: {topK}</span>
                  <GitBranch size={12} className="text-emerald-400" />
                </div>
                <input
                  type="range"
                  min="1"
                  max="8"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full"
                  disabled={isRunning}
                  aria-label="Top-K selection"
                />
              </div>
            </div>
          </div>

          {/* Metrics Panel */}
          <div className="space-y-4">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-3">
                <div className="text-[10px] uppercase font-mono text-blue-400 mb-1">
                  Policy Accuracy
                </div>
                <div className="text-2xl font-mono text-blue-300">
                  {state.policyAccuracy.toFixed(1)}%
                </div>
                <div className="text-[9px] text-slate-500 mt-1">
                  Target: 47.0%
                </div>
              </div>

              <div className="bg-emerald-900/20 border border-emerald-500/30 rounded-lg p-3">
                <div className="text-[10px] uppercase font-mono text-emerald-400 mb-1">
                  Avg Reward
                </div>
                <div className="text-2xl font-mono text-emerald-300">
                  {state.averageReward.toFixed(3)}
                </div>
                <div className="text-[9px] text-slate-500 mt-1">
                  Top-{topK} edits
                </div>
              </div>

              <div className="bg-amber-900/20 border border-amber-500/30 rounded-lg p-3">
                <div className="text-[10px] uppercase font-mono text-amber-400 mb-1">
                  RL Iteration
                </div>
                <div className="text-2xl font-mono text-amber-300">
                  {state.iteration}
                </div>
                <div className="text-[9px] text-slate-500 mt-1">
                  ReST-EM rounds
                </div>
              </div>

              <div className="bg-rose-900/20 border border-rose-500/30 rounded-lg p-3">
                <div className="text-[10px] uppercase font-mono text-rose-400 mb-1">
                  Knowledge Retention
                </div>
                <div className="text-2xl font-mono text-rose-300">
                  {state.knowledgeRetention.toFixed(0)}%
                </div>
                <div className="text-[9px] text-slate-500 mt-1">
                  Forgetting penalty
                </div>
              </div>
            </div>

            {/* Performance Chart */}
            <div className="bg-black/40 rounded-lg p-4 border border-slate-700/50">
              <div className="text-[10px] uppercase font-mono text-slate-400 mb-3 flex items-center gap-2">
                <TrendingUp size={12} />
                TRAINING PROGRESS
              </div>
              <svg viewBox="0 0 300 150" className="w-full" aria-label="Performance chart">
                <title>Training Progress</title>
                {/* Grid */}
                {[0, 25, 50, 75, 100].map((y) => (
                  <line
                    key={y}
                    x1="0"
                    y1={150 - (y * 1.5)}
                    x2="300"
                    y2={150 - (y * 1.5)}
                    stroke="rgba(71, 85, 105, 0.3)"
                    strokeWidth="1"
                  />
                ))}

                {/* Accuracy line */}
                {performanceHistory.length > 1 && (
                  <polyline
                    points={performanceHistory
                      .map((acc: number, i: number) => {
                        const x = (i / (performanceHistory.length - 1)) * 300;
                        const y = 150 - ((acc / 100) * 150);
                        return `${x},${y}`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="rgb(59, 130, 246)"
                    strokeWidth="2"
                  />
                )}

                {/* Retention line */}
                {retentionHistory.length > 1 && (
                  <polyline
                    points={retentionHistory
                      .map((ret: number, i: number) => {
                        const x = (i / (retentionHistory.length - 1)) * 300;
                        const y = 150 - ((ret / 100) * 150);
                        return `${x},${y}`;
                      })
                      .join(" ")}
                    fill="none"
                    stroke="rgb(251, 113, 133)"
                    strokeWidth="2"
                    strokeDasharray="4,4"
                  />
                )}

                {/* Legend */}
                <text x="10" y="15" fill="rgb(59, 130, 246)" fontSize="10" fontFamily="monospace">
                  — Accuracy
                </text>
                <text x="10" y="30" fill="rgb(251, 113, 133)" fontSize="10" fontFamily="monospace">
                  - - Retention
                </text>
              </svg>
            </div>

            {/* Self-Edit Quality Distribution */}
            <div className="bg-black/40 rounded-lg p-4 border border-slate-700/50">
              <div className="text-[10px] uppercase font-mono text-slate-400 mb-3 flex items-center gap-2">
                <Database size={12} />
                SELF-EDIT REWARDS
              </div>
              <div className="space-y-1">
                {state.edits.slice(0, 8).map((edit, idx) => {
                  const barWidth = edit.reward * 100;
                  const hue = edit.reward * 120;
                  return (
                    <div key={edit.id} className="flex items-center gap-2">
                      <div className="text-[9px] font-mono text-slate-500 w-12">
                        Edit {idx + 1}
                      </div>
                      <div className="flex-1 h-4 bg-slate-800/50 rounded overflow-hidden">
                        <div
                          className="h-full transition-all duration-300"
                          style={{
                            width: `${barWidth}%`,
                            backgroundColor: `hsl(${hue}, ${edit.selected ? 100 : 50}%, ${edit.selected ? 60 : 40}%)`,
                          }}
                        />
                      </div>
                      <div className="text-[9px] font-mono text-slate-400 w-12 text-right">
                        {edit.reward.toFixed(2)}
                      </div>
                      {edit.selected && (
                        <div className="text-emerald-400 text-[9px]">✓</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Explanation */}
            <div className="bg-slate-900/50 rounded-lg p-3 border border-slate-700/50">
              <div className="text-[10px] uppercase font-mono text-slate-400 mb-2">
                🔍 What to watch:
              </div>
              <ul className="text-xs text-slate-300 space-y-1 list-disc list-inside">
                <li><strong>Green particles</strong>: High-reward self-edits selected for RL update</li>
                <li><strong>Policy Accuracy</strong>: Improves as better edits are discovered</li>
                <li><strong>Knowledge Retention</strong>: Decreases due to catastrophic forgetting</li>
                <li><strong>Top-K</strong>: Higher K = more diversity but slower convergence</li>
              </ul>
            </div>
          </div>
        </div>
      </SchematicCard>
    </div>
  );
};

export default SEALAdaptationSimulation;
