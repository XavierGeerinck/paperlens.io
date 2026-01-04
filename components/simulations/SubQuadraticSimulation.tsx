import React from "react";
import { Activity, Layers, Database, Zap } from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const SubQuadraticSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			contextLength: 1024,
		},
		onTick: (prev) => {
			if (prev.contextLength < 131072) { // 128k limit
				return { ...prev, contextLength: prev.contextLength * 1.5 };
			}
			return prev;
		},
		tickRate: 300,
	});

	// Realistic Constants (approx 7B model parameters)
	const LAYERS = 32;
	const HIDDEN_DIM = 4096;
	const BYTES_PER_PARAM = 2; // FP16
	const STATE_DIM = 16; // Mamba default state dimension

	// 1. Memory Calculation (GB)
	// Transformer: KV Cache = 2 * Layers * Dim * SeqLen * Bytes
	const transformerMemGB = (2 * LAYERS * HIDDEN_DIM * state.contextLength * BYTES_PER_PARAM) / 1e9;
	// SSM: Recurrent State = Layers * Dim * StateDim * Bytes
	const ssmMemGB = (LAYERS * HIDDEN_DIM * STATE_DIM * BYTES_PER_PARAM) / 1e9;

	// 2. Compute Cost (Relative Operations per Token)
	// Transformer: Attention requires scanning all previous tokens (Linear per step -> Quadratic total)
	const transformerOps = state.contextLength; 
	// SSM: Fixed state update (Constant per step -> Linear total)
	const ssmOps = STATE_DIM * 100; // Constant baseline

	const maxMem = 80; // 80GB A100 VRAM limit
	const maxOps = 131072; // Max context for visualization

	const getBarHeight = (val: number, max: number) => Math.min(100, (val / max) * 100);

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="SCALING_LAW_COMPARISON (7B MODEL)">
				<div className="flex justify-center mb-6">
					<div className="bg-slate-800 px-6 py-3 rounded-lg border border-slate-700 text-center">
						<div className="text-xs text-slate-500 uppercase font-bold mb-1">Sequence Length</div>
						<div className="text-3xl font-mono text-white">
							{Math.round(state.contextLength).toLocaleString()} <span className="text-sm text-slate-500">tokens</span>
						</div>
					</div>
				</div>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: TRANSFORMER */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-90">
						<div className="flex items-center gap-2 text-red-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Layers size={16} /> Transformer
						</div>

						<div className="space-y-6">
							{/* Memory Metric */}
							<div className="space-y-2">
								<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
									<span className="flex items-center gap-1"><Database size={10}/> KV Cache (VRAM)</span>
									<span className={transformerMemGB > maxMem ? "text-red-500 font-bold" : "text-slate-300"}>
										{transformerMemGB.toFixed(2)} GB {transformerMemGB > maxMem && "(OOM)"}
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div 
										className={`h-full transition-all duration-300 ${transformerMemGB > maxMem ? "bg-red-600" : "bg-red-500"}`}
										style={{ width: `${getBarHeight(transformerMemGB, maxMem)}%` }} 
									/>
								</div>
							</div>

							{/* Compute Metric */}
							<div className="space-y-2">
								<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
									<span className="flex items-center gap-1"><Zap size={10}/> Compute / Token</span>
									<span className="text-slate-300">O(N) Scaling</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div 
										className="h-full bg-red-500 transition-all duration-300"
										style={{ width: `${getBarHeight(transformerOps, maxOps)}%` }} 
									/>
								</div>
							</div>
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed mt-4">
							<strong className="text-red-400">Linear Memory Growth:</strong> Storing Key-Value pairs for 128k tokens requires ~67GB VRAM, nearly filling an A100 GPU purely with cache.
						</div>
					</div>

					{/* RIGHT: SSM */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-emerald-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Activity size={16} /> Hybrid SSM (Mamba)
						</div>

						<div className="space-y-6">
							{/* Memory Metric */}
							<div className="space-y-2">
								<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
									<span className="flex items-center gap-1"><Database size={10}/> Recurrent State</span>
									<span className="text-emerald-400 font-bold">
										{ssmMemGB.toFixed(4)} GB
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden relative">
									<div 
										className="h-full bg-emerald-500 transition-all duration-300"
										style={{ width: `${getBarHeight(ssmMemGB, maxMem)}%` }} 
									/>
									{/* Pulse animation to show activity despite constant size */}
									{isRunning && (
										<div className="absolute inset-0 bg-white/20 animate-pulse" style={{ width: `${getBarHeight(ssmMemGB, maxMem)}%` }} />
									)}
								</div>
							</div>

							{/* Compute Metric */}
							<div className="space-y-2">
								<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500">
									<span className="flex items-center gap-1"><Zap size={10}/> Compute / Token</span>
									<span className="text-emerald-400">O(1) Constant</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden relative">
									<div 
										className="h-full bg-emerald-500 transition-all duration-300"
										style={{ width: "2%" }} 
									/>
									{/* Activity indicator */}
									{isRunning && (
										<div className="absolute top-0 left-0 h-full w-1 bg-white animate-[ping_1s_ease-in-out_infinite]" />
									)}
								</div>
							</div>
						</div>

						<div className="p-3 bg-emerald-900/10 rounded border border-emerald-500/20 text-[11px] text-slate-400 leading-relaxed mt-4">
							<strong className="text-emerald-400">Constant Memory:</strong> The SSM state is fixed at ~4MB regardless of sequence length, allowing theoretically infinite context on consumer hardware.
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-red-900/10 border border-red-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-red-400 uppercase mb-1">What to watch for: The Memory Wall</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice how the <span className="text-red-400 font-bold">KV Cache</span> (Top Bar) fills up the GPU memory as sequence length increases. This is the primary bottleneck for long-context Transformers.
						</p>
					</div>
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">What to watch for: Constant State</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The SSM state remains tiny (~4MB). This means you can process a 1M token book with the same RAM usage as a short sentence.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_SCALING" : "START_SCALING_TEST"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default SubQuadraticSimulation;
