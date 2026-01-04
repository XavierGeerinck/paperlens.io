import React from "react";
import {
	Activity,
	ArrowRight,
	Network,
	AlertTriangle,
	CheckCircle,
	Zap,
	ShieldCheck,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface MHCState {
	layer: number;
	wildSignal: number;
	mhcSignal: number;
	wildStatus: "STABLE" | "EXPLODED" | "VANISHED";
	mhcStatus: "STABLE";
	wildHistory: number[];
	mhcHistory: number[];
}

const DeepSeekMHCSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			layer: 0,
			wildSignal: 1.0,
			mhcSignal: 1.0,
			wildStatus: "STABLE" as "STABLE" | "EXPLODED" | "VANISHED",
			mhcStatus: "STABLE" as "STABLE",
			wildHistory: [1.0],
			mhcHistory: [1.0],
		},
		onTick: (prev) => {
			if (prev.layer >= 100) {
				stop();
				return prev;
			}

			// Wild Mode: Random drift that compounds
			const wildDrift = 1.0 + (Math.random() * 0.4 - 0.18); // Biased slightly upward to explode
			let newWildSignal = prev.wildSignal * wildDrift;
			let newWildStatus = prev.wildStatus;

			if (newWildSignal > 100) {
				newWildStatus = "EXPLODED";
				newWildSignal = 100; // Cap for visualization
			} else if (newWildSignal < 0.01) {
				newWildStatus = "VANISHED";
				newWildSignal = 0;
			}

			// mHC Mode: Doubly Stochastic constraint keeps signal near 1.0
			// Simulate small perturbations that are corrected (Sinkhorn normalization)
			const perturbation = (Math.random() * 0.4 - 0.2); 
			// The signal drifts but is constantly pulled back to 1.0 by the constraints
			const rawSignal = prev.mhcSignal + perturbation;
			const newMhcSignal = rawSignal + (1.0 - rawSignal) * 0.8; // Strong correction

			return {
				...prev,
				layer: prev.layer + 1,
				wildSignal: newWildStatus === "STABLE" ? newWildSignal : prev.wildSignal,
				mhcSignal: newMhcSignal,
				wildStatus: newWildStatus,
				wildHistory: [...prev.wildHistory, newWildSignal].slice(-50),
				mhcHistory: [...prev.mhcHistory, newMhcSignal].slice(-50),
			};
		},
		tickRate: 100,
	});

	const renderMiniGraph = (history: number[], color: string, max: number) => {
		const width = 200;
		const height = 60;
		const points = history
			.map((val, i) => {
				const x = (i / (history.length - 1)) * width;
				const y = height - (Math.min(val, max) / max) * height;
				return `${x},${y}`;
			})
			.join(" ");

		return (
			<svg width={width} height={height} className="overflow-visible">
				<polyline
					points={points}
					fill="none"
					stroke={color}
					strokeWidth="2"
					strokeLinecap="round"
					strokeLinejoin="round"
				/>
			</svg>
		);
	};

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="SIGNAL_SURVIVAL_PROTOCOL">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: STANDARD (WILD) */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6">
						<div className="flex items-center gap-2 text-red-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<AlertTriangle size={16} /> Standard (Wild)
						</div>

						<div className="flex flex-col gap-2">
							<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase">
								<span>Signal Magnitude</span>
								<span className={state.wildStatus !== "STABLE" ? "text-red-500 font-bold" : ""}>
									{state.wildSignal.toFixed(4)}
								</span>
							</div>
							<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
								<div
									className={`h-full transition-all duration-300 ${
										state.wildStatus === "EXPLODED"
											? "bg-red-500 w-full"
											: state.wildStatus === "VANISHED"
												? "bg-blue-500 w-0"
												: "bg-slate-400"
									}`}
									style={{ width: `${Math.min(state.wildSignal * 50, 100)}%` }}
								/>
							</div>
						</div>

						<div className="bg-black/40 p-4 rounded border border-slate-800 flex flex-col items-center justify-center min-h-[100px] relative">
							{renderMiniGraph(state.wildHistory, "#94a3b8", 5)}
							{state.wildStatus !== "STABLE" && (
								<div className="absolute bg-red-950/90 border border-red-500 px-3 py-1 rounded text-[10px] font-bold text-red-400 animate-bounce">
									SIGNAL_{state.wildStatus}
								</div>
							)}
						</div>

						<p className="text-[11px] text-slate-400 leading-relaxed italic">
							Without constraints, small errors in weight initialization compound
							exponentially across layers, leading to NaN or zeroed signals.
						</p>
					</div>

					{/* RIGHT: DeepSeek mHC */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-emerald-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<ShieldCheck size={16} /> DeepSeek mHC
						</div>

						<div className="flex flex-col gap-2">
							<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase">
								<span>Signal Magnitude</span>
								<span className="text-emerald-400 font-bold">
									{state.mhcSignal.toFixed(4)}
								</span>
							</div>
							<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
								<div
									className="h-full bg-emerald-500 transition-all duration-300"
									style={{ width: `${Math.min(state.mhcSignal * 50, 100)}%` }}
								/>
							</div>
						</div>

						<div className="bg-black/40 p-4 rounded border border-slate-800 flex flex-col items-center justify-center min-h-[100px] relative">
							{renderMiniGraph(state.mhcHistory, "#10b981", 5)}
							<div className="absolute top-2 right-2">
								<Zap size={14} className="text-emerald-500 animate-pulse" />
							</div>
						</div>

						<p className="text-[11px] text-slate-400 leading-relaxed">
							mHC uses <span className="text-emerald-400 font-bold">Doubly Stochastic</span> constraints. Every row and column sums to 1.0, ensuring energy conservation.
						</p>
					</div>
				</div>

				{/* Layer Counter */}
				<div className="mt-4 flex items-center justify-center gap-4 bg-slate-950 p-2 rounded border border-slate-800">
					<div className="text-[10px] font-mono text-slate-500 uppercase">Depth Progress</div>
					<div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
						<div 
							className="h-full bg-blue-500 transition-all duration-300" 
							style={{ width: `${state.layer}%` }}
						/>
					</div>
					<div className="text-xs font-mono text-blue-400">L{state.layer}/100</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">What to watch for: Signal Survival</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice how the <span className="text-blue-400 font-bold">Standard</span> signal eventually hits a wall (Explosion or Vanishing). mHC allows the signal to "survive" through hundreds of layers by maintaining a constant energy state.
						</p>
					</div>
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">What to watch for: Doubly Stochastic</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The <span className="text-emerald-400 font-bold">mHC</span> weights are iteratively normalized using the Sinkhorn-Knopp algorithm. This forces the network to act as a conservative system where information is routed but never lost.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_SIGNAL" : "START_SIGNAL"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default DeepSeekMHCSimulation;
