import React from "react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { Database, ArrowDown, Layers } from "lucide-react";

const MLASimulation: React.FC = () => {
	const { state, start, stop, reset, isRunning } = useSimulation({
		initialState: {
			tokens: 0,
		},
		onTick: (prev) => ({
			tokens: prev.tokens + 1,
		}),
		tickRate: 100,
	});

	// Derived stats
	const cacheSizeMHA = state.tokens * 128; // Standard MHA growth
	const cacheSizeMLA = state.tokens * 4; // MLA growth (approx 32x compression)

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="KV_CACHE_COMPRESSION_ENGINE">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT COLUMN: STANDARD MHA */}
					<div className="flex flex-col items-center gap-6 border-r border-slate-800 pr-4">
						<div className="text-sm font-bold text-red-400 uppercase tracking-wider">
							Standard MHA
						</div>

						{/* 1. Input Token */}
						<div className="flex flex-col items-center gap-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								Input (h_t)
							</div>
							<div className="w-20 h-6 bg-slate-800 border border-slate-600 rounded flex items-center justify-center text-[10px] font-mono">
								[ 4096 ]
							</div>
							<ArrowDown className="text-slate-600 w-4 h-4" />
						</div>

						{/* 2. Projection Layer */}
						<div className="flex flex-col items-center gap-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								Full Projection
							</div>
							<div className="w-24 h-24 bg-red-900/20 border-2 border-red-500/50 rounded flex flex-col items-center justify-center gap-2 p-2">
								<div className="grid grid-cols-2 gap-1 w-full h-full opacity-50">
									<div className="bg-red-500/40 rounded"></div>
									<div className="bg-red-500/40 rounded"></div>
									<div className="bg-red-500/40 rounded"></div>
									<div className="bg-red-500/40 rounded"></div>
								</div>
							</div>
						</div>

						<ArrowDown className="text-slate-600 w-4 h-4" />

						{/* 3. KV Cache Storage */}
						<div className="flex flex-col items-center gap-2 w-full">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								VRAM Usage
							</div>
							<div className="w-32 h-40 border-2 border-red-500 bg-red-950/30 rounded flex flex-col justify-end relative overflow-hidden">
								{/* Fill Level */}
								<div
									className="w-full bg-red-500 transition-all duration-200"
									style={{
										height: `${Math.min((cacheSizeMHA / 2000) * 100, 100)}%`,
									}}
								/>
								<div className="absolute inset-0 flex items-center justify-center flex-col gap-1 z-10">
									<Database size={20} className="text-red-200" />
									<div className="text-xs font-bold text-white drop-shadow-md">
										{cacheSizeMHA} MB
									</div>
								</div>
							</div>
							<div className="text-xs text-red-400 font-mono">
								OOM Risk: HIGH
							</div>
						</div>
					</div>

					{/* RIGHT COLUMN: DEEPSEEK MLA */}
					<div className="flex flex-col items-center gap-6 pl-4">
						<div className="text-sm font-bold text-emerald-400 uppercase tracking-wider">
							DeepSeek MLA
						</div>

						{/* 1. Input Token */}
						<div className="flex flex-col items-center gap-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								Input (h_t)
							</div>
							<div className="w-20 h-6 bg-slate-800 border border-slate-600 rounded flex items-center justify-center text-[10px] font-mono">
								[ 4096 ]
							</div>
							<ArrowDown className="text-slate-600 w-4 h-4" />
						</div>

						{/* 2. Projection Layer */}
						<div className="flex flex-col items-center gap-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								Low-Rank Down
							</div>
							<div className="w-24 h-24 bg-emerald-900/20 border-2 border-emerald-500/50 rounded flex flex-col items-center justify-center gap-2 p-2 relative">
								<div className="w-6 h-full bg-emerald-500/40 rounded animate-pulse"></div>
								<div className="absolute -right-16 top-1/2 -translate-y-1/2 text-[9px] text-slate-500 w-14 text-center leading-tight">
									Latent c_KV (Compressed)
								</div>
							</div>
						</div>

						<ArrowDown className="text-slate-600 w-4 h-4" />

						{/* 3. KV Cache Storage */}
						<div className="flex flex-col items-center gap-2 w-full">
							<div className="text-[10px] font-mono text-slate-500 uppercase">
								VRAM Usage
							</div>
							<div className="w-32 h-40 border-2 border-emerald-500 bg-emerald-950/30 rounded flex flex-col justify-end relative overflow-hidden">
								{/* Fill Level */}
								<div
									className="w-full bg-emerald-500 transition-all duration-200"
									style={{
										height: `${Math.min((cacheSizeMLA / 2000) * 100, 100)}%`,
									}}
								/>
								<div className="absolute inset-0 flex items-center justify-center flex-col gap-1 z-10">
									<Database size={20} className="text-emerald-200" />
									<div className="text-xs font-bold text-white drop-shadow-md">
										{cacheSizeMLA} MB
									</div>
								</div>
							</div>
							<div className="text-xs text-emerald-400 font-mono">
								Efficiency: 32x
							</div>
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">
							What to watch for: Latent Compression
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice the{" "}
							<span className="text-emerald-400 font-bold">Low-Rank Down</span>{" "}
							projection. Instead of storing full-dimensional KV vectors, MLA
							compresses them into a tiny latent vector. This reduces the KV
							cache footprint by up to{" "}
							<span className="text-emerald-400 font-bold">93%</span>.
						</p>
					</div>
					<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">
							What to watch for: Weight Absorption
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The Up-Projection matrix ({"$W_{UKV}$"}) is mathematically
							absorbed into the Query/Output projections. This means the model{" "}
							<span className="text-blue-400 font-bold">never expands</span> the
							compressed cache back to full size in memory during computation.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "PAUSE_GENERATION" : "START_INFERENCE"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default MLASimulation;
