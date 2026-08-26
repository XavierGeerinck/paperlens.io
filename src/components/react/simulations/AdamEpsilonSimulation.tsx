import React from "react";
import { Play, RotateCcw, Microscope, Activity } from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

// --- Types ---
interface Point { x: number; y: number }

// We need to track the internal state of the optimizer (Momentum m, Variance v)
// to accurately simulate the trajectory and oscillations.
interface AdamState {
	m: { x: number; y: number };
	v: { x: number; y: number };
	params: Point;
}

interface OptimizerState extends Record<string, unknown> {
	step: number;
	
	// Full state history for visualization
	defaultPath: Point[];   // eps=1e-8
	karpathyPath: Point[];  // eps=1e-10
	
	// Current internal optimizer states
	defaultState: AdamState;
	karpathyState: AdamState;
	
	isConverged: boolean;
}

const AdamEpsilonSimulation: React.FC = () => {
    // -------------------------------------------------------------------------
    // 1. Simulation Logic (Full Adam Implementation)
    // -------------------------------------------------------------------------
	// Loss landscape: x^2 + 1e-9 * y^2
	const grad = (x: number, y: number) => ({ dx: 2 * x, dy: 2 * 1e-9 * y });

	const initialAdamState: AdamState = {
		m: { x: 0, y: 0 },
		v: { x: 0, y: 0 },
		params: { x: -2, y: 1.8 }
	};

	const { isRunning, state, start, stop, reset } = useSimulation<OptimizerState>({
		initialState: {
			step: 0,
			defaultPath: [initialAdamState.params],
			karpathyPath: [initialAdamState.params],
			defaultState: initialAdamState,
			karpathyState: initialAdamState,
			isConverged: false,
		},
		tickRate: 60, // Faster tick rate for smoother animation
		onTick: (prev) => {
			if (prev.step > 300) {
				stop();
				return { ...prev, isConverged: true };
			}

			// FULL ADAM UPDATE STEP
			// theta_{t+1} = theta_t - alpha * m_hat / (sqrt(v_hat) + eps)
			const updateAdam = (s: AdamState, eps: number, t: number): AdamState => {
				const { dx, dy } = grad(s.params.x, s.params.y);
				const alpha = 0.05; // Learning rate
				const beta1 = 0.9;
				const beta2 = 0.999;

				// Update biased first moment estimate (m)
				const m_x = beta1 * s.m.x + (1 - beta1) * dx;
				const m_y = beta1 * s.m.y + (1 - beta1) * dy;

				// Update biased second raw moment estimate (v)
				const v_x = beta2 * s.v.x + (1 - beta2) * (dx * dx);
				const v_y = beta2 * s.v.y + (1 - beta2) * (dy * dy);

				// Compute bias-corrected first moment estimate
				const m_hat_x = m_x / (1 - Math.pow(beta1, t));
				const m_hat_y = m_y / (1 - Math.pow(beta1, t));

				// Compute bias-corrected second raw moment estimate
				const v_hat_x = v_x / (1 - Math.pow(beta2, t));
				const v_hat_y = v_y / (1 - Math.pow(beta2, t));

				// Update parameters
				// The "Epsilon Trap" happens here: if sqrt(v_hat) << eps, the denominator becomes constant 'eps'
				const param_x = s.params.x - alpha * m_hat_x / (Math.sqrt(v_hat_x) + eps);
				const param_y = s.params.y - alpha * m_hat_y / (Math.sqrt(v_hat_y) + eps);

				return {
					m: { x: m_x, y: m_y },
					v: { x: v_x, y: v_y },
					params: { x: param_x, y: param_y }
				};
			};

			// Step starts at 1
			const t = prev.step + 1;
			const nextDefault = updateAdam(prev.defaultState, 1e-8, t);
			const nextKarpathy = updateAdam(prev.karpathyState, 1e-10, t);

			return {
				step: t,
				defaultState: nextDefault,
				karpathyState: nextKarpathy,
				defaultPath: [...prev.defaultPath, nextDefault.params],
				karpathyPath: [...prev.karpathyPath, nextKarpathy.params],
				isConverged: false
			};
		},
	});

    // -------------------------------------------------------------------------
    // 2. Visualization
    // -------------------------------------------------------------------------
	const toScreen = (p: Point) => ({
		x: ((p.x + 2.5) / 5) * 300,
		y: 200 - ((p.y + 0.5) / 2.5) * 200
	});

	const renderPath = (path: Point[], color: string) => {
		if (path.length < 2) return null;
		const d = path.map((p, i) => {
			const s = toScreen(p);
			return `${i === 0 ? 'M' : 'L'} ${s.x} ${s.y}`;
		}).join(' ');
		return <path d={d} fill="none" stroke={color} strokeWidth="2" />;
	};

	const lastDefault = state.defaultState.params;
	const lastKarpathy = state.karpathyState.params;

	// METRIC CALCULATION FOR "DEEP DIVE" PANEL
	// Check the flat dimension (y) gradients magnitude vs epsilon
	const currentDy = Math.abs(2 * 1e-9 * lastDefault.y);
	const ratioDefault = currentDy / 1e-8; // If < 1, Epsilon dominates
	const ratioKarpathy = currentDy / 1e-10; // If > 1, Gradient dominates (Good)

	return (
		<div className="flex flex-col gap-6 p-4 bg-slate-950 rounded-lg">
			
			<div className="flex items-center justify-between">
				<div className="flex items-center gap-2">
					<Activity className="text-indigo-400" size={20} />
					<h3 className="font-bold text-slate-200">Adam's Epsilon Trap</h3>
				</div>
				<span className="font-mono text-xs text-slate-500">STEP {state.step}</span>
			</div>

			{/* Main Comparison View */}
			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
				
				{/* Left: The Trajectory Map */}
				<div className="lg:col-span-2 relative h-[300px] bg-slate-900/50 border border-slate-800 rounded-lg overflow-hidden">
					{/* Grid Background */}
					<svg className="absolute inset-0 w-full h-full opacity-10">
						<defs>
							<pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
								<path d="M 20 0 L 0 0 0 20" fill="none" stroke="white" strokeWidth="0.5"/>
							</pattern>
						</defs>
						<rect width="100%" height="100%" fill="url(#grid)" />
					</svg>

					{/* Labels */}
					<div className="absolute top-3 left-3 bg-slate-950/80 px-2 py-1 rounded border border-slate-800 pointer-events-none">
						<span className="text-[10px] font-mono text-slate-400">LOSS LANDSCAPE (Top-Down)</span>
					</div>

					<div className="absolute inset-0 flex items-center justify-center">
						<svg width="300" height="200" className="overflow-visible">
							{/* Origin Marker */}
							<path d="M 145 100 L 155 100 M 150 95 L 150 105" stroke="#22272f" strokeWidth="1" />
							
							{/* Contours (representing the flat canyon) */}
							<ellipse cx="150" cy="100" rx="20" ry="80" stroke="#22272f" strokeWidth="1" strokeDasharray="2 2" fill="none" />
							<ellipse cx="150" cy="100" rx="40" ry="120" stroke="#161a20" strokeWidth="1" strokeDasharray="2 2" fill="none" />

							{/* Trajectories */}
							{renderPath(state.defaultPath, "#f5555d")}
							{renderPath(state.karpathyPath, "#35d492")}

							{/* Heads */}
							<g transform={`translate(${toScreen(lastDefault).x}, ${toScreen(lastDefault).y})`}>
								<circle r="4" fill="#f5555d" className="animate-pulse" />
								<text y="-8" textAnchor="middle" className="text-[8px] fill-red-400 font-mono font-bold">1e-8</text>
							</g>

							<g transform={`translate(${toScreen(lastKarpathy).x}, ${toScreen(lastKarpathy).y})`}>
								<circle r="4" fill="#35d492" />
								<text y="-8" textAnchor="middle" className="text-[8px] fill-emerald-400 font-mono font-bold">1e-10</text>
							</g>
						</svg>
					</div>
				</div>

				{/* Right: Deep Dive Analysis Panel */}
				<div className="flex flex-col gap-4">
					
					<div className="p-4 bg-slate-900 border border-slate-700 rounded-lg">
						<div className="flex items-center gap-2 mb-3">
							<Microscope size={14} className="text-sky-400" />
							<span className="text-xs font-bold text-sky-400 uppercase">Signal-to-Noise Ratio</span>
						</div>
						
						{/* Bar Chart Comparison */}
						<div className="space-y-4">
							<div>
								<div className="flex justify-between text-[10px] text-slate-400 mb-1">
									<span>Gradient Magnitude</span>
									<span className="font-mono text-slate-200">~{currentDy.toExponential(1)}</span>
								</div>
								
								{/* Threshold Line Visualization */}
								<div className="relative h-6 w-full bg-slate-800 rounded-sm overflow-hidden">
									{/* The Gradient Signal Level */}
									<div className="absolute top-0 bottom-0 left-0 bg-yellow-400/80 w-[50%] border-r-2 border-yellow-200 z-10 transition-all duration-300" 
										style={{ width: '40%' }}>
									</div>
									<div className="absolute top-1 left-2 text-[9px] font-bold text-yellow-900 z-20">SIGNAL (Grad)</div>

									{/* Epsilon Barriers */}
									{/* 1e-10 Barrier (Way to the left) */}
									<div className="absolute top-0 bottom-0 left-[10%] border-l border-dashed border-emerald-500 z-0"></div>
									<div className="absolute -bottom-4 left-[10%] text-[8px] text-emerald-500">-10</div>

									{/* 1e-8 Barrier (Usually dominates signal) */}
									<div className="absolute top-0 bottom-0 left-[60%] border-l-2 border-red-500 z-0 bg-red-500/10 w-full"></div>
									<div className="absolute -bottom-4 left-[60%] text-[8px] text-red-500">-8 (Wall)</div>
								</div>
								
								<div className="mt-6 flex flex-col gap-2">
									<div className={`text-[10px] p-2 rounded border ${ratioDefault < 1 ? 'bg-red-500/10 border-red-500/30 text-red-300' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
										<span className="font-bold">eps=1e-8:</span> {ratioDefault < 1 ? "BLOCKED. Epsilon > Gradient." : "Active"}
									</div>
									<div className={`text-[10px] p-2 rounded border ${ratioKarpathy > 1 ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' : 'bg-slate-800 border-slate-700 text-slate-400'}`}>
										<span className="font-bold">eps=1e-10:</span> {ratioKarpathy > 1 ? "ADAPTIVE. Gradient > Epsilon." : "Blocked"}
									</div>
								</div>
							</div>
						</div>
					</div>

					<div className="bg-slate-900 p-3 rounded-lg border border-slate-800">
						<div className="text-[10px] text-slate-500 mb-1 uppercase">Distance to Goal (Y)</div>
						<div className="flex items-end justify-between h-16 gap-2">
							<div className="w-full bg-red-500/20 rounded-t relative group h-full">
								<div className="absolute bottom-0 w-full bg-red-500 transition-all duration-300" style={{ height: `${(Math.abs(lastDefault.y)/2)*100}%` }}></div>
								<div className="absolute -top-4 w-full text-center text-[10px] text-red-400">{lastDefault.y.toFixed(3)}</div>
							</div>
							<div className="w-full bg-emerald-500/20 rounded-t relative group h-full">
								<div className="absolute bottom-0 w-full bg-emerald-500 transition-all duration-300" style={{ height: `${(Math.abs(lastKarpathy.y)/2)*100}%` }}></div>
								<div className="absolute -top-4 w-full text-center text-[10px] text-emerald-400">{lastKarpathy.y.toFixed(3)}</div>
							</div>
						</div>
					</div>

				</div>
			</div>

			{/* Controls */}
			<div className="flex gap-2 border-t border-zinc-800 pt-4 mt-2">
				<SchematicButton onClick={isRunning ? stop : start} variant="primary" className="flex-1">
					<Play size={14} className={isRunning ? "opacity-50" : ""} />
					{isRunning ? "PAUSE SIMULATION" : "RUN SIMULATION"}
				</SchematicButton>
				<SchematicButton onClick={reset} variant="secondary">
					<RotateCcw size={14} /> RESET
				</SchematicButton>
			</div>
		</div>
	);
};


export default AdamEpsilonSimulation;
