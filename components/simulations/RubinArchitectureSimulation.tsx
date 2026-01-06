import React from "react";
import { Cpu, Database, Layers, Server } from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const RubinArchitectureSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			bwBlackwell: 0,
			bwRubin: 0,
		},
		onTick: (prev) => {
			const targetBlackwell = 8.0; // TB/s
			const targetRubin = 22.0; // TB/s (HBM4)

			return {
				...prev,
				bwBlackwell: Math.min(prev.bwBlackwell + 0.4, targetBlackwell),
				bwRubin: Math.min(prev.bwRubin + 1.1, targetRubin), // Faster ramp to show higher ceiling
			};
		},
		tickRate: 100,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="ARCHITECTURAL_EVOLUTION_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-6">
					{/* LEFT: BLACKWELL (Previous Gen) */}
					<div className="flex flex-col items-center gap-6 border-r border-slate-800 pr-4 opacity-80">
						<div className="text-sm font-bold text-slate-400 uppercase tracking-wider">
							Blackwell (B200)
						</div>

						{/* Chip Visual */}
						<div className="relative w-32 h-32 bg-slate-800 rounded-lg border-2 border-slate-600 flex flex-col items-center justify-center p-2">
							<div className="absolute top-2 left-2 text-[9px] font-mono text-slate-500">
								2x Reticle
							</div>
							<div className="grid grid-cols-2 gap-1 w-full h-full">
								<div className="bg-slate-700 rounded flex items-center justify-center">
									<Layers size={16} className="text-slate-500" />
								</div>
								<div className="bg-slate-700 rounded flex items-center justify-center">
									<Layers size={16} className="text-slate-500" />
								</div>
							</div>
							{/* HBM3e Stacks */}
							<div className="absolute -bottom-3 flex gap-1">
								{[1, 2, 3, 4].map((i) => (
									<div
										key={i}
										className="w-4 h-8 bg-slate-900 border border-slate-700 rounded-sm overflow-hidden flex flex-col justify-end"
									>
										<div
											className="w-full bg-blue-500 transition-all duration-200"
											style={{ height: `${(state.bwBlackwell / 8.0) * 100}%` }}
										/>
									</div>
								))}
							</div>
						</div>

						{/* Metrics */}
						<div className="flex flex-col items-center mt-4">
							<div className="text-xs font-mono text-slate-500 uppercase">
								HBM3e Bandwidth
							</div>
							<div className="text-2xl font-mono text-blue-400">
								{state.bwBlackwell.toFixed(1)}{" "}
								<span className="text-sm text-slate-500">TB/s</span>
							</div>
						</div>
					</div>

					{/* RIGHT: RUBIN (Next Gen) */}
					<div className="flex flex-col items-center gap-6 pl-4">
						<div className="text-sm font-bold text-emerald-400 uppercase tracking-wider">
							Rubin (R100)
						</div>

						{/* Chip Visual */}
						<div className="relative w-40 h-40 bg-slate-800 rounded-lg border-2 border-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.1)] flex flex-col items-center justify-center p-2">
							<div className="absolute top-2 left-2 text-[9px] font-mono text-emerald-500">
								4x Reticle
							</div>
							<div className="grid grid-cols-2 gap-1 w-full h-full">
								<div className="bg-emerald-900/30 rounded flex items-center justify-center border border-emerald-500/30">
									<Layers size={20} className="text-emerald-500" />
								</div>
								<div className="bg-emerald-900/30 rounded flex items-center justify-center border border-emerald-500/30">
									<Layers size={20} className="text-emerald-500" />
								</div>
								<div className="bg-emerald-900/30 rounded flex items-center justify-center border border-emerald-500/30">
									<Layers size={20} className="text-emerald-500" />
								</div>
								<div className="bg-emerald-900/30 rounded flex items-center justify-center border border-emerald-500/30">
									<Layers size={20} className="text-emerald-500" />
								</div>
							</div>
							{/* HBM4 Stacks */}
							<div className="absolute -bottom-3 flex gap-1">
								{[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
									<div
										key={i}
										className="w-3 h-10 bg-slate-900 border border-slate-700 rounded-sm overflow-hidden flex flex-col justify-end"
									>
										<div
											className="w-full bg-yellow-500 transition-all duration-200"
											style={{ height: `${(state.bwRubin / 13.0) * 100}%` }}
										/>
									</div>
								))}
							</div>
						</div>

						{/* Metrics */}
						<div className="flex flex-col items-center mt-4">
							<div className="text-xs font-mono text-slate-500 uppercase">
								HBM4 Bandwidth
							</div>
							<div className="text-3xl font-mono text-yellow-400 font-bold">
								{state.bwRubin.toFixed(1)}{" "}
								<span className="text-sm text-slate-500">TB/s</span>
							</div>
							<div className="text-[10px] text-emerald-500 mt-1 font-bold">
								+175% Throughput
							</div>
						</div>
					</div>
				</div>

				<div className="grid grid-cols-3 gap-4 mt-8 border-t border-slate-800 pt-6">
					<div className="p-3 bg-slate-800/50 rounded border border-slate-700">
						<div className="flex items-center gap-2 text-slate-400 mb-1">
							<Cpu size={14} />{" "}
							<span className="text-xs font-bold">CPU ARCH</span>
						</div>
						<div className="flex justify-between items-end">
							<div className="text-xs text-slate-500">Grace</div>
							<div className="text-sm font-bold text-emerald-400">Vera</div>
						</div>
					</div>
					<div className="p-3 bg-slate-800/50 rounded border border-slate-700">
						<div className="flex items-center gap-2 text-slate-400 mb-1">
							<Database size={14} />{" "}
							<span className="text-xs font-bold">MEMORY</span>
						</div>
						<div className="flex justify-between items-end">
							<div className="text-xs text-slate-500">HBM3e</div>
							<div className="text-sm font-bold text-yellow-400">HBM4</div>
						</div>
					</div>
					<div className="p-3 bg-slate-800/50 rounded border border-slate-700">
						<div className="flex items-center gap-2 text-slate-400 mb-1">
							<Server size={14} />{" "}
							<span className="text-xs font-bold">PROCESS</span>
						</div>
						<div className="flex justify-between items-end">
							<div className="text-xs text-slate-500">4NP</div>
							<div className="text-sm font-bold text-emerald-400">3nm</div>
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">
							What to watch for: Bandwidth
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice the HBM4 stacks on the Rubin chip. By moving to HBM4,
							NVIDIA achieves a{" "}
							<span className="text-emerald-400 font-bold">2.8x increase</span>{" "}
							in memory bandwidth (22 TB/s vs 8 TB/s), which is critical for
							feeding the massive FLOPs required for trillion-parameter models.
						</p>
					</div>
					<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">
							What to watch for: Compute Density
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The Rubin chip features a{" "}
							<span className="text-blue-400 font-bold">4x Reticle</span> design
							on a 3nm process. This allows for significantly more transistors
							and specialized "Vera" CPU cores to be packed into the same
							Superchip footprint compared to Blackwell.
						</p>
					</div>
				</div>

				<div className="mt-4 p-3 bg-purple-900/10 border border-purple-500/20 rounded-lg">
					<h4 className="text-[10px] font-bold text-purple-400 uppercase mb-1">
						New Component: Rubin CPX
					</h4>
					<p className="text-[11px] text-slate-400 leading-relaxed">
						Not shown in this single-chip view is the new{" "}
						<span className="text-purple-400 font-bold">Rubin CPX</span>, a
						specialized accelerator with 128GB GDDR7 dedicated to the "prefill"
						phase of inference, decoupling context processing from token
						generation.
					</p>
				</div>

				<div className="flex gap-4 mt-6">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "PAUSE_BENCHMARK" : "RUN_COMPARISON"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default RubinArchitectureSimulation;
