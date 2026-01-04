import React, { useMemo } from "react";
import { Activity, Layers, ArrowRight } from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const SubQuadraticSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			contextLength: 1024,
		},
		onTick: (prev) => {
			if (prev.contextLength < 128000) {
				return { ...prev, contextLength: prev.contextLength * 1.2 }; // Slower growth to see the divergence
			}
			return prev;
		},
		tickRate: 200,
	});

	const contextK = state.contextLength / 1000;

	// Transformer Metrics (Quadratic)
	const transformerMetrics = {
		memory: Math.pow(contextK, 2) * 0.1, // Scaling factor for visualization
		compute: Math.pow(contextK, 2) * 0.15,
		color: "#ef4444",
	};

	// SSM Metrics (Linear)
	const ssmMetrics = {
		memory: contextK * 0.2,
		compute: contextK * 0.1,
		color: "#10b981",
	};

	// Normalize for bars (max out at some value to keep UI sane)
	const maxVal = 200; // Arbitrary scale unit

	const getHeight = (val: number) => Math.min(100, (val / maxVal) * 100);

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="SCALING_LAW_COMPARISON">
				{/* Context Counter */}
				<div className="flex justify-center mb-6">
					<div className="bg-slate-800 px-6 py-3 rounded-lg border border-slate-700 text-center">
						<div className="text-xs text-slate-500 uppercase font-bold mb-1">
							Sequence Length
						</div>
						<div className="text-3xl font-mono text-white">
							{Math.round(state.contextLength).toLocaleString()}{" "}
							<span className="text-sm text-slate-500">tokens</span>
						</div>
					</div>
				</div>

				<div className="grid grid-cols-2 gap-8">
					{/* Transformer Column */}
					<div className="space-y-4 border-r border-slate-800 pr-4">
						<div className="flex items-center gap-2 text-red-400 font-bold font-mono">
							<Layers size={18} /> TRANSFORMER
						</div>
						<div className="text-xs text-slate-500">
							Complexity: <span className="text-red-400">O(N²)</span>
						</div>

						{/* Visualizing the Matrix Explosion */}
						<div className="aspect-square w-full bg-slate-950 rounded border border-slate-800 relative overflow-hidden p-2">
							<div className="absolute inset-0 flex items-center justify-center opacity-20">
								<div className="w-full h-full grid grid-cols-12 grid-rows-12 gap-0.5">
									{Array.from({ length: 144 }).map((_, i) => (
										<div
											key={i}
											className="bg-red-500 transition-opacity duration-300"
											style={{
												opacity:
													i / 144 < transformerMetrics.compute / maxVal
														? 0.8
														: 0.1,
											}}
										/>
									))}
								</div>
							</div>
							<div className="absolute bottom-2 left-2 text-xs font-mono text-red-400">
								Attn Matrix: {Math.round(contextK)}k x {Math.round(contextK)}k
							</div>
						</div>

						{/* Metrics Bars */}
						<div className="space-y-3">
							<div>
								<div className="flex justify-between text-xs mb-1">
									<span>VRAM (KV Cache)</span>
									<span className="text-red-400">
										{transformerMetrics.memory.toFixed(1)} GB
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div
										className="h-full bg-red-500 transition-all duration-200"
										style={{
											width: `${getHeight(transformerMetrics.memory)}%`,
										}}
									/>
								</div>
							</div>
							<div>
								<div className="flex justify-between text-xs mb-1">
									<span>Compute (FLOPs)</span>
									<span className="text-red-400">
										{transformerMetrics.compute.toFixed(1)} TFLOPs
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div
										className="h-full bg-red-500 transition-all duration-200"
										style={{
											width: `${getHeight(transformerMetrics.compute)}%`,
										}}
									/>
								</div>
							</div>
						</div>
					</div>

					{/* SSM Column */}
					<div className="space-y-4 pl-4">
						<div className="flex items-center gap-2 text-emerald-400 font-bold font-mono">
							<Activity size={18} /> HYBRID SSM (MAMBA)
						</div>
						<div className="text-xs text-slate-500">
							Complexity: <span className="text-emerald-400">O(N)</span>
						</div>

						{/* Visualizing Linear Scan */}
						<div className="aspect-square w-full bg-slate-950 rounded border border-slate-800 relative overflow-hidden p-2 flex items-center">
							<div className="w-full h-12 flex gap-1 items-center overflow-hidden">
								{Array.from({ length: 20 }).map((_, i) => (
									<div
										key={i}
										className="h-8 w-4 bg-emerald-500 rounded-sm transition-all duration-300"
										style={{
											opacity:
												0.5 + Math.sin(i + state.contextLength / 1000) * 0.5,
										}}
									/>
								))}
								<ArrowRight className="text-emerald-500 animate-pulse" />
							</div>
							<div className="absolute bottom-2 left-2 text-xs font-mono text-emerald-400">
								State Size: Constant (16MB)
							</div>
						</div>

						{/* Metrics Bars */}
						<div className="space-y-3">
							<div>
								<div className="flex justify-between text-xs mb-1">
									<span>VRAM (State)</span>
									<span className="text-emerald-400">
										{ssmMetrics.memory.toFixed(1)} GB
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div
										className="h-full bg-emerald-500 transition-all duration-200"
										style={{ width: `${getHeight(ssmMetrics.memory)}%` }}
									/>
								</div>
							</div>
							<div>
								<div className="flex justify-between text-xs mb-1">
									<span>Compute (FLOPs)</span>
									<span className="text-emerald-400">
										{ssmMetrics.compute.toFixed(1)} TFLOPs
									</span>
								</div>
								<div className="h-2 bg-slate-800 rounded-full overflow-hidden">
									<div
										className="h-full bg-emerald-500 transition-all duration-200"
										style={{ width: `${getHeight(ssmMetrics.compute)}%` }}
									/>
								</div>
							</div>
						</div>
					</div>
				</div>

				<div className="flex gap-4 mt-6">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "PAUSE_SIMULATION" : "START_SCALING_TEST"}
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
