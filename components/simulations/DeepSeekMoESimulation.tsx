import React from "react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { Brain } from "lucide-react";

const EXPERT_COUNT = 8; // Reduced for visual clarity

const DeepSeekMoESimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			experts: Array(EXPERT_COUNT)
				.fill(0)
				.map((_, i) => ({
					id: i,
					load: 0,
					bias: 0,
					specialization: ["Code", "Math", "Logic", "Creative"][i % 4],
				})),
			tokensProcessed: 0,
			activeIndices: [] as number[],
			currentTask: "Idle",
		},
		onTick: (prev) => {
			// 1. Generate a random task
			const tasks = [
				"Generate Python Code",
				"Solve Calculus",
				"Write Poem",
				"Debug SQL",
			];
			const taskIdx = Math.floor(Math.random() * tasks.length);
			const currentTask = tasks[taskIdx];

			// 2. Router Logic (Simulated)
			const scores = prev.experts.map((e) => {
				let affinity = Math.random();
				if (currentTask.includes("Code") && e.specialization === "Code")
					affinity += 0.5;
				if (currentTask.includes("Calculus") && e.specialization === "Math")
					affinity += 0.5;

				return affinity + e.bias;
			});

			const sortedIdx = scores
				.map((s, i) => [s, i])
				.sort((a, b) => b[0] - a[0]);
			const winners = [sortedIdx[0][1], sortedIdx[1][1]];

			// 3. Update Loads & Biases
			const newExperts = prev.experts.map((e, i) => {
				let newLoad = e.load * 0.9; // Decay
				if (winners.includes(i)) newLoad += 10; // Spike load

				const targetLoad = 10;
				const loadDiff = targetLoad - newLoad;
				const newBias = e.bias + loadDiff * 0.005;

				return { ...e, load: newLoad, bias: newBias };
			});

			return {
				...prev,
				experts: newExperts,
				tokensProcessed: prev.tokensProcessed + 1,
				activeIndices: winners,
				currentTask,
			};
		},
		tickRate: 800,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="MOE_ARCHITECTURE_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-12 py-8">
					{/* LEFT: STANDARD MoE */}
					<div className="flex flex-col items-center gap-6 border-r border-slate-800 pr-6 opacity-70">
						<div className="text-sm font-bold text-slate-400 uppercase tracking-wider">
							Standard MoE (Mixtral)
						</div>

						<div className="w-full space-y-4">
							<div className="p-3 rounded border border-slate-700 bg-slate-800/50 text-[10px] text-slate-400 text-center">
								No Shared Expert: Basic knowledge is redundant across all
								experts.
							</div>

							<div className="grid grid-cols-4 gap-2">
								{Array(8)
									.fill(0)
									.map((_, i) => (
										<div
											key={`std-exp-${i}`}
											className="flex flex-col items-center gap-1"
										>
											<div className="w-full h-10 bg-slate-800 border border-slate-700 rounded relative overflow-hidden">
												<div
													className="absolute bottom-0 left-0 w-full bg-slate-500 transition-all duration-500"
													style={{
														height: isRunning
															? `${20 + Math.random() * 60}%`
															: "0%",
													}}
												/>
											</div>
											<div className="text-[8px] text-slate-600">Exp {i}</div>
										</div>
									))}
							</div>
						</div>
					</div>

					{/* RIGHT: DEEPSEEK MoE */}
					<div className="flex flex-col items-center gap-6 pl-2">
						<div className="text-sm font-bold text-emerald-400 uppercase tracking-wider">
							DeepSeek MoE (V3)
						</div>

						<div className="w-full space-y-6">
							{/* Shared Expert */}
							<div className="p-3 rounded border border-blue-500/30 bg-blue-900/10 relative">
								<div className="flex justify-between items-center mb-2">
									<div className="flex items-center gap-2 text-blue-400 font-bold text-[10px] uppercase">
										<Brain size={12} /> Shared Expert
									</div>
									<div className="text-[9px] font-mono text-blue-300">
										Always Active
									</div>
								</div>
								<div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
									<div
										className={`h-full bg-blue-500 transition-all duration-500 ${isRunning ? "w-full animate-pulse" : "w-0"}`}
									></div>
								</div>
								<div className="mt-1 text-[9px] text-slate-500">
									Handles common knowledge (grammar, syntax).
								</div>
							</div>

							{/* Routed Experts */}
							<div className="grid grid-cols-4 gap-2">
								{state.experts.map((exp, i) => (
									<div
										key={exp.id}
										className={`p-1.5 rounded border transition-all duration-300 flex flex-col items-center gap-1 ${
											state.activeIndices.includes(i)
												? "bg-emerald-500/20 border-emerald-500 scale-105"
												: "bg-slate-800 border-slate-700 opacity-40"
										}`}
									>
										<div className="w-full h-10 bg-slate-900 rounded relative overflow-hidden">
											<div
												className="absolute bottom-0 left-0 w-full bg-emerald-500 transition-all duration-300"
												style={{ height: `${Math.min(exp.load * 2, 100)}%` }}
											/>
										</div>
										<div className="text-[7px] font-mono text-slate-500">
											B:{exp.bias.toFixed(1)}
										</div>
									</div>
								))}
							</div>
						</div>
					</div>
				</div>

				<div className="flex gap-4 mt-4 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "PAUSE_ROUTING" : "START_COMPARISON"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>

			<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
					<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">
						What to watch for: Shared Experts
					</h4>
					<p className="text-[11px] text-slate-400 leading-relaxed">
						Unlike traditional MoE, DeepSeek-V3 isolates "common knowledge" into
						a shared expert that is{" "}
						<span className="text-blue-400 font-bold">always active</span>. This
						prevents the specialized experts from wasting capacity on redundant
						tasks like basic grammar.
					</p>
				</div>
				<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
					<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">
						What to watch for: Bias Balancing
					</h4>
					<p className="text-[11px] text-slate-400 leading-relaxed">
						Watch the{" "}
						<span className="font-mono text-emerald-400 font-bold">
							B (Bias)
						</span>{" "}
						values. Instead of using a complex loss function that hurts
						performance, the router dynamically adjusts a bias term to ensure no
						expert is left behind (or overloaded).
					</p>
				</div>
			</div>
		</div>
	);
};

export default DeepSeekMoESimulation;
