import React from "react";
import type { FC } from "react";
import {
	BrainCircuit,
	Cpu,
	Database,
	Play,
	Pause,
	RotateCcw,
	Gauge,
	Activity,
} from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

type Model1State = {
	// Metrics
	contextToken: number;
	entropy: number;
	routingToMoE: boolean;
	cacheSavings: number;
	latency: number;
	memoryUsage: number;
	totalTokensProcessed: number;
	moeProcessedCount: number;
	engramProcessedCount: number;

	// Real-time data
	currentValueVector: number[];
	positionalSignal: number[];
	combinedVector: number[];
};

function clamp(value: number, min: number, max: number) {
	return Math.max(min, Math.min(max, value));
}

const renderSparkline = (
	history: number[] | undefined,
	color: string,
	height = 40,
) => {
	const values = history && history.length > 1 ? history : [0, 0];
	const width = 200;
	const paddingX = 2;
	const paddingY = 2;

	const min = Math.min(...values, 0);
	const max = Math.max(...values, 1);

	const points = values
		.map((val, i) => {
			const x = paddingX + (i / (values.length - 1)) * (width - 2 * paddingX);
			const y =
				height -
				paddingY -
				((clamp(val, min, max) - min) / (max - min)) * (height - 2 * paddingY);
			return `${x},${y}`;
		})
		.join(" ");

	return (
		<svg
			viewBox={`0 0 ${width} ${height}`}
			className="w-full h-full overflow-visible"
			preserveAspectRatio="none"
		>
			<title>Metric Sparkline</title>
			<polyline
				points={points}
				fill="none"
				stroke={color}
				strokeWidth="1.5"
				strokeLinecap="round"
				strokeLinejoin="round"
			/>
		</svg>
	);
};

const VectorViz: FC<{ label: string; values: number[]; color: string }> = ({
	label,
	values,
	color,
}) => (
	<div className="flex flex-col gap-1">
		<div className="text-[10px] font-mono text-zinc-500 uppercase flex justify-between">
			<span>{label}</span>
			<span className="text-zinc-600">[{values.length}d]</span>
		</div>
		<div className="flex gap-[2px] h-8 items-end">
			{values.map((v, i) => (
				<div
					key={`${label}-${i}`}
					className="flex-1 rounded-sm transition-all duration-300"
					style={{
						height: `${Math.abs(v) * 100}%`,
						backgroundColor: v > 0 ? color : `${color}88`,
						opacity: 0.3 + Math.abs(v) * 0.7,
					}}
				/>
			))}
		</div>
	</div>
);

const DeepSeekModel1Simulation: FC = () => {
	const { isRunning, state, logs, history, start, stop, reset } =
		useSimulation<Model1State>({
			initialState: {
				contextToken: 0,
				entropy: 0.5,
				routingToMoE: true,
				cacheSavings: 0,
				latency: 20,
				memoryUsage: 100,
				totalTokensProcessed: 0,
				moeProcessedCount: 0,
				engramProcessedCount: 0,
				currentValueVector: Array.from({ length: 32 }, () => Math.random()),
				positionalSignal: Array.from({ length: 32 }, () => Math.sin(0)),
				combinedVector: Array.from({ length: 32 }, () => Math.random()),
			},
			tickRate: 400,
			onTick: (prev) => {
				const newToken = prev.contextToken + 1;

				// Complexity/Entropy simulation
				// High entropy (new information) goes to MoE
				// Low entropy (facts/repetition) goes to Engram
				const entropy =
					0.3 + Math.random() * 0.7 * (Math.sin(newToken / 10) * 0.5 + 0.5);
				const routingToMoE = entropy > 0.65;

				const moeCount = routingToMoE
					? prev.moeProcessedCount + 1
					: prev.moeProcessedCount;
				const engramCount = routingToMoE
					? prev.engramProcessedCount
					: prev.engramProcessedCount + 1;

				// Calculate savings: Engram is 80% cheaper
				const currentSavings = (engramCount / (moeCount + engramCount)) * 30; // Max 30% aggregate as claim

				// VVPA Signal
				const freq = 1 / Math.pow(10000, 0); // Simplified
				const posSignal = Array.from({ length: 32 }, (_, i) =>
					Math.sin(newToken * freq * (i / 16)),
				);

				const semanticValue = Array.from({ length: 32 }, () => Math.random());
				const combined = semanticValue.map(
					(v, i) => v * (0.8 + 0.2 * posSignal[i]),
				);

				return {
					contextToken: newToken,
					entropy,
					routingToMoE,
					moeProcessedCount: moeCount,
					engramProcessedCount: engramCount,
					totalTokensProcessed: moeCount + engramCount,
					cacheSavings: currentSavings,
					latency: routingToMoE
						? 45 + Math.random() * 10
						: 8 + Math.random() * 4,
					memoryUsage: 100 - currentSavings,
					currentValueVector: semanticValue,
					positionalSignal: posSignal,
					combinedVector: combined,
				};
			},
			onLog: (state) => {
				const path = state.routingToMoE ? "MoE (Compute)" : "Engram (Memory)";
				return `Token ${state.contextToken}: Routing to ${path} (Entropy: ${state.entropy.toFixed(2)})`;
			},
		});

	return (
		<div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-4 bg-zinc-950 rounded-xl overflow-hidden">
			{/* Sidebar: Controls & High-level Status */}
			<div className="lg:col-span-4 space-y-4">
				<SchematicCard title="MODEL1 CONTROLLER">
					<div className="space-y-6">
						<div className="flex justify-between items-center bg-zinc-900/50 p-3 rounded border border-zinc-800">
							<div>
								<div className="text-[10px] font-mono text-zinc-500 uppercase">
									Tokens Processed
								</div>
								<div className="text-2xl font-mono text-white leading-none mt-1">
									{state.totalTokensProcessed}
								</div>
							</div>
							<div className="flex gap-2 text-zinc-400">
								<SchematicButton
									onClick={isRunning ? stop : start}
									variant={isRunning ? "secondary" : "primary"}
								>
									{isRunning ? (
										<Pause className="w-4 h-4" />
									) : (
										<Play className="w-4 h-4" />
									)}
								</SchematicButton>
								<SchematicButton onClick={reset} variant="secondary">
									<RotateCcw className="w-4 h-4" />
								</SchematicButton>
							</div>
						</div>

						<div className="space-y-4">
							<div className="space-y-1">
								<div className="flex justify-between text-[11px] font-mono uppercase">
									<span className="text-zinc-400">Memory Efficiency</span>
									<span className="text-emerald-400">
										+{state.cacheSavings.toFixed(1)}%
									</span>
								</div>
								<div className="h-2 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
									<div
										className="h-full bg-emerald-500 transition-all duration-500 ease-out"
										style={{ width: `${(state.cacheSavings / 30) * 100}%` }}
									/>
								</div>
							</div>

							<div className="space-y-1">
								<div className="flex justify-between text-[11px] font-mono uppercase">
									<span className="text-zinc-400">Inference Latency</span>
									<span className="text-amber-400">
										{state.latency.toFixed(1)}ms
									</span>
								</div>
								<div className="h-10 bg-zinc-900/50 rounded border border-zinc-800 p-1">
									{renderSparkline(history.latency, "#f8c05a", 32)}
								</div>
							</div>
						</div>

						<div className="pt-4 border-t border-zinc-900">
							<div className="text-[10px] font-mono text-zinc-500 uppercase mb-3">
								Live Log
							</div>
							<div className="h-32 overflow-y-auto font-mono text-[10px] space-y-1 px-2 custom-scrollbar">
								{logs
									.slice()
									.reverse()
									.map((log, i) => (
										<div
											key={`log-${logs.length - i}`}
											className={i === 0 ? "text-indigo-400" : "text-zinc-600"}
										>
											{">"} {log}
										</div>
									))}
							</div>
						</div>
					</div>
				</SchematicCard>
			</div>

			{/* Main Panel: Architecture Visualization */}
			<div className="lg:col-span-8 space-y-4">
				<SchematicCard title="ENTROPY ROUTING ENGINE">
					<div className="relative h-64 bg-zinc-900/30 rounded border border-zinc-800 flex items-center justify-around overflow-hidden">
						{/* Background Grid */}
						<div className="absolute inset-0 bg-grid opacity-10 pointer-events-none" />

						{/* Input Side */}
						<div className="z-10 flex flex-col items-center">
							<div
								className={`p-4 rounded-xl border-2 transition-all duration-300 ${isRunning ? "border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.3)] animate-pulse" : "border-zinc-700"}`}
							>
								<Activity className="w-8 h-8 text-indigo-400" />
							</div>
							<span className="mt-2 text-[10px] font-mono text-zinc-400">
								INPUT STREAM
							</span>
						</div>

						{/* Routing Logic */}
						<div className="flex flex-col items-center gap-2">
							<div
								className={`text-[10px] font-mono px-2 py-1 rounded bg-zinc-800 transition-colors ${state.routingToMoE ? "border-none text-zinc-500" : "border border-emerald-500 text-emerald-400 font-bold"}`}
							>
								{state.entropy.toFixed(3)} ENTROPY
							</div>
							<div className="flex items-center gap-1">
								<div
									className={`w-8 h-[2px] transition-colors ${state.routingToMoE ? "bg-amber-500" : "bg-emerald-500"}`}
								/>
								<div
									className={`p-3 rounded-lg border bg-zinc-900 transition-colors ${state.routingToMoE ? "border-amber-500" : "border-emerald-500"}`}
								>
									<Gauge
										className={`w-5 h-5 ${state.routingToMoE ? "text-amber-500" : "text-emerald-500"}`}
									/>
								</div>
								<div
									className={`w-12 h-[2px] transition-colors ${state.routingToMoE ? "bg-amber-500" : "bg-emerald-500"}`}
								/>
							</div>
							<span className="text-[10px] font-mono text-zinc-400">
								DYNAMIC ROUTER
							</span>
						</div>

						{/* Output Targets */}
						<div className="flex flex-col gap-8">
							<div
								className={`flex items-center gap-3 p-3 rounded border transition-all duration-300 ${state.routingToMoE ? "border-amber-500 bg-amber-500/10 scale-105" : "border-zinc-800 opacity-40"}`}
							>
								<Cpu className="w-6 h-6 text-amber-500" />
								<div>
									<div className="text-[11px] font-mono font-bold text-amber-200">
										DEEP MoE
									</div>
									<div className="text-[9px] font-mono text-amber-500/70">
										REASONING STACK
									</div>
								</div>
							</div>

							<div
								className={`flex items-center gap-3 p-3 rounded border transition-all duration-300 ${!state.routingToMoE ? "border-emerald-500 bg-emerald-500/10 scale-105" : "border-zinc-800 opacity-40"}`}
							>
								<Database className="w-6 h-6 text-emerald-500" />
								<div>
									<div className="text-[11px] font-mono font-bold text-emerald-200">
										ENGRAM
									</div>
									<div className="text-[9px] font-mono text-emerald-500/70">
										FACT LOOKUP
									</div>
								</div>
							</div>
						</div>

						{/* Path visualization lines */}
						<svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20">
							<title>Routing Path</title>
							<path
								d="M 120,128 L 300,128"
								stroke="currentColor"
								fill="none"
								className={
									state.routingToMoE ? "text-amber-500" : "text-emerald-500"
								}
							/>
						</svg>
					</div>
				</SchematicCard>

				<SchematicCard title="VALUE VECTOR POSITION AWARENESS (VVPA)">
					<div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-6 bg-zinc-950 rounded border border-zinc-900">
						<VectorViz
							label="Semantic Value (v_t)"
							values={state.currentValueVector}
							color="#8b6cf2"
						/>
						<VectorViz
							label="Positional Signal (R_m)"
							values={state.positionalSignal}
							color="#e13540"
						/>
						<VectorViz
							label="Combined VVPA Output"
							values={state.combinedVector}
							color="#35d492"
						/>
					</div>
					<div className="mt-4 p-3 bg-zinc-900/50 rounded border border-zinc-800">
						<div className="flex items-start gap-3">
							<BrainCircuit className="w-5 h-5 text-indigo-400 shrink-0 mt-1" />
							<p className="text-[11px] font-mono text-zinc-400 leading-relaxed">
								<span className="text-white font-bold">Concept:</span> VVPA
								injects a high-frequency positional signal directly into the
								Value stream. This allows the model to differentiate tokens that
								have identical semantics but appear in different positions in
								long contexts without expensive attention re-calculation.
							</p>
						</div>
					</div>
				</SchematicCard>
			</div>
		</div>
	);
};

export default DeepSeekModel1Simulation;
