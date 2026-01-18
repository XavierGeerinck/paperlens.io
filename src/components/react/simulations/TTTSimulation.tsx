import { useMemo } from "react";
import type { FC } from "react";
import { Cpu, Zap, Activity, Info } from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

type TTTSimState = {
	// Simulation parameters
	contextLength: number;
	speed: number;

	// Transformer Metrics
	transformerLatency: number;
	transformerLoss: number;
	transformerMemory: number;

	// RNN Metrics
	rnnLatency: number;
	rnnLoss: number;
	rnnMemory: number;

	// TTT Metrics
	tttLatency: number;
	tttLoss: number;
	tttMemory: number;
};

function renderInternalGraph(
	data: number[] | undefined,
	color: string,
	title: string,
	min = 0,
	max = 1,
) {
	const width = 200;
	const height = 50;
	// useSimulation keeps typically 50 items.
	const values = data || [];
	
	if (values.length < 2) return null;

	const points = values
		.map((val, i) => {
			const x = (i / (values.length - 1)) * width;
			const y = height - ((Math.min(max, Math.max(min, val)) - min) / (max - min)) * height;
			return `${x},${y}`;
		})
		.join(" ");

	const currentVal = values[values.length - 1];

	return (
		<div className="flex flex-col">
			<div className="text-[10px] font-mono text-slate-500 uppercase mb-1 flex justify-between">
				<span>{title}</span>
				<span style={{ color }}>{currentVal.toFixed(2)}</span>
			</div>
			<svg
				viewBox={`0 0 ${width} ${height}`}
				className="w-full h-12 overflow-visible bg-black/20 rounded border border-slate-800/50"
				preserveAspectRatio="none"
			>
				<polyline
					points={points}
					fill="none"
					stroke={color}
					strokeWidth="1.5"
					strokeLinecap="round"
					strokeLinejoin="round"
				/>
			</svg>
		</div>
	);
}

const TTTSimulation: FC = () => {
	const { isRunning, state, history, start, stop, reset, update } = useSimulation<TTTSimState>({
		initialState: {
			contextLength: 0,
			speed: 1,
			transformerLatency: 0.1,
			transformerLoss: 0.1,
			transformerMemory: 0.1,
			rnnLatency: 0.1,
			rnnLoss: 0.1,
			rnnMemory: 0.1,
			tttLatency: 0.1,
			tttLoss: 0.1,
			tttMemory: 0.2,
		},
		onTick: (prev) => {
			const newContext = prev.contextLength + (prev.speed * 100); // Process 100 tokens per tick adjusted by speed
			const ctxNormalized = newContext / 10000; // Normalize for simulation math

			// Noise generator
			const noise = () => (Math.random() - 0.5) * 0.05;

			// 1. TRANSFORMER (Full Attention)
			// Latency scales linearly with context (attending to all previous tokens)
			// Loss stays low and constant (perfect recall)
			// Memory scales linearly (KV cache)
			const tfLatency = Math.min(1, 0.1 + ctxNormalized * 0.8 + noise());
			const tfLoss = Math.max(0, 0.1 + noise()); // Low loss
			const tfMemory = Math.min(1, 0.1 + ctxNormalized * 0.9); // Linear memory growth

			// 2. RNN / Mamba
			// Latency is constant (O(1))
			// Loss improves initially then degrades as context overflows fixed state
			// Memory is constant
			const rnnLatency = 0.15 + noise();
			const rnnLoss = Math.min(1, 0.1 + (ctxNormalized > 0.5 ? (ctxNormalized - 0.5) * 1.5 : 0) + noise());
			const rnnMemory = 0.1; // Small constant state

			// 3. TTT-E2E
			// Latency is constant (like RNN) but slightly higher due to update step
			// Loss stays low (like Transformer) due to test-time training
			// Memory is constant (compressed weights)
			const tttLatency = 0.25 + noise(); // Constant but higher base cost than inference-only RNN
			const tttLoss = Math.max(0, 0.1 + noise()); // Matches Transformer performance
			const tttMemory = 0.2; // Constant, larger than RNN state but fixed

			return {
				...prev,
				contextLength: newContext,
				transformerLatency: tfLatency,
				transformerLoss: tfLoss,
				transformerMemory: tfMemory,
				rnnLatency: rnnLatency,
				rnnLoss: rnnLoss,
				rnnMemory: rnnMemory,
				tttLatency: tttLatency,
				tttLoss: tttLoss,
				tttMemory: tttMemory,
			};
		},
		tickRate: 100,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="CONTEXT_SCALING_SIMULATION">
				<div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
					{/* Controls */}
					<div className="space-y-4 lg:col-span-1">
						<div className="flex items-center gap-2 text-indigo-300 font-mono text-xs uppercase border-b border-slate-800 pb-2">
							<Activity size={16} /> Simulation Control
						</div>
						
						<div className="p-3 bg-black/30 border border-slate-800 rounded mb-4">
							<div className="text-[10px] font-mono text-slate-500 uppercase mb-1">
								Context Length
							</div>
							<div className="text-2xl font-mono text-white">
								{state.contextLength.toLocaleString()} tk
							</div>
						</div>

						<label className="block">
							<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
								<span>Simulation Speed</span>
								<span className="text-slate-300">{state.speed}x</span>
							</div>
							<input
								type="range"
								min={1}
								max={10}
								step={1}
								value={state.speed}
								onChange={(e) => update({ speed: Number(e.target.value) })}
								className="w-full accent-indigo-500"
							/>
						</label>

						<div className="flex gap-2 pt-2">
							<SchematicButton onClick={isRunning ? stop : start}>
								{isRunning ? "PAUSE" : "RUN"}
							</SchematicButton>
							<SchematicButton onClick={() => reset()} variant="secondary">
								RESET
							</SchematicButton>
						</div>

						<div className="mt-4 p-3 bg-indigo-900/10 border border-indigo-500/20 rounded-lg">
							<h4 className="text-[10px] font-bold text-indigo-400 uppercase mb-1 flex items-center gap-2">
								<Info size={14} /> Insight
							</h4>
							<p className="text-[11px] text-slate-400 leading-relaxed">
								Watch how <span className="text-emerald-400 font-bold">TTT</span> maintains low loss (like Transformer) while keeping latency flat (like RNNs) as context grows.
							</p>
						</div>
					</div>

					{/* Transformer Column */}
					<div className="bg-slate-800/30 p-4 rounded-lg space-y-4 border border-slate-700/50">
						<div className="flex items-center gap-2 text-rose-300 font-bold font-mono text-sm uppercase border-b border-rose-500/20 pb-2">
							<Cpu size={16} /> Transformer
						</div>
						{renderInternalGraph(history["transformerLatency"], "#f43f5e", "Latency (Time/Token)")}
						{renderInternalGraph(history["transformerLoss"], "#fb7185", "Loss (Perplexity)")}
						{renderInternalGraph(history["transformerMemory"], "#fda4af", "Memory (KV Cache)")}
					</div>

					{/* RNN Column */}
					<div className="bg-slate-800/30 p-4 rounded-lg space-y-4 border border-slate-700/50">
						<div className="flex items-center gap-2 text-amber-300 font-bold font-mono text-sm uppercase border-b border-amber-500/20 pb-2">
							<Activity size={16} /> RNN / Mamba
						</div>
						{renderInternalGraph(history["rnnLatency"], "#d97706", "Latency (Time/Token)")}
						{renderInternalGraph(history["rnnLoss"], "#f59e0b", "Loss (Perplexity)")}
						{renderInternalGraph(history["rnnMemory"], "#fbbf24", "Memory (Hidden State)")}
					</div>

					{/* TTT Column */}
					<div className="bg-emerald-900/20 p-4 rounded-lg space-y-4 border border-emerald-500/30">
						<div className="flex items-center gap-2 text-emerald-300 font-bold font-mono text-sm uppercase border-b border-emerald-500/20 pb-2">
							<Zap size={16} /> TTT-E2E
						</div>
						{renderInternalGraph(history["tttLatency"], "#059669", "Latency (Time/Token)")}
						{renderInternalGraph(history["tttLoss"], "#10b981", "Loss (Perplexity)")}
						{renderInternalGraph(history["tttMemory"], "#34d399", "Memory (Weights)")}
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default TTTSimulation;
