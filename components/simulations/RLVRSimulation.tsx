import React from "react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import {
	CheckCircle,
	XCircle,
	BrainCircuit,
	Terminal,
	AlertTriangle,
} from "lucide-react";

interface ThoughtStep {
	id: number;
	text: string;
	type: "thought" | "error" | "correction" | "verified";
}

const RLVRSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			steps: [] as ThoughtStep[],
			standardOutput: "" as string,
			status: "idle" as
				| "idle"
				| "thinking"
				| "verifying"
				| "success"
				| "failed",
			reward: 0,
		},
		onTick: (prev) => {
			const stepCount = prev.steps.length;

			// Simulation Script: A math problem solving attempt
			// Problem: Solve 2x + 5 = 15

			let newStep: ThoughtStep | null = null;
			let newStatus = prev.status;
			let newReward = prev.reward;
			let newStandardOutput = prev.standardOutput;

			if (stepCount === 0) {
				newStep = {
					id: 1,
					text: "Problem: Solve 2x + 5 = 15",
					type: "thought",
				};
				newStatus = "thinking";
				newStandardOutput = "Thinking...";
			} else if (stepCount === 1) {
				newStep = {
					id: 2,
					text: "Subtract 5 from both sides: 2x = 10",
					type: "thought",
				};
				newStandardOutput = "The answer is x = 4."; // Standard model hallucination/error
			} else if (stepCount === 2) {
				newStep = { id: 3, text: "Divide by 2: x = 4", type: "error" }; // Intentional error
			} else if (stepCount === 3) {
				newStep = {
					id: 4,
					text: "Wait, let me double check...",
					type: "correction",
				}; // The "Aha" moment
			} else if (stepCount === 4) {
				newStep = { id: 5, text: "10 / 2 is 5, not 4.", type: "correction" };
			} else if (stepCount === 5) {
				newStep = { id: 6, text: "Final Answer: x = 5", type: "thought" };
				newStatus = "verifying";
			} else if (stepCount === 6) {
				// Verification Step
				newStep = {
					id: 7,
					text: "VERIFIER: Check 2(5) + 5 = 15... TRUE",
					type: "verified",
				};
				newStatus = "success";
				newReward = 1.0;
				stop(); // End simulation
			}

			if (newStep) {
				return {
					...prev,
					steps: [...prev.steps, newStep],
					status: newStatus,
					reward: newReward,
					standardOutput: newStandardOutput,
				};
			}
			return prev;
		},
		tickRate: 1200,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="REASONING_EVOLUTION_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: STANDARD MODEL (SFT) */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-80">
						<div className="flex items-center gap-2 text-slate-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Terminal size={16} /> Standard Model (SFT)
						</div>

						<div className="p-4 bg-slate-950 rounded border border-slate-800 min-h-[100px] flex flex-col gap-2">
							<div className="text-[10px] text-slate-500 uppercase">
								Prompt: Solve 2x + 5 = 15
							</div>
							<div
								className={`text-sm font-mono ${state.standardOutput.includes("x = 4") ? "text-red-400" : "text-slate-300"}`}
							>
								{state.standardOutput}
							</div>
							{state.standardOutput.includes("x = 4") && (
								<div className="mt-auto flex items-center gap-2 text-[10px] text-red-500 font-bold uppercase">
									<XCircle size={12} /> Incorrect (No Reasoning)
								</div>
							)}
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
							Standard models often "leap" to an answer based on pattern
							matching, leading to errors in multi-step logic.
						</div>
					</div>

					{/* RIGHT: DEEPSEEK-R1 (RLVR) */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-purple-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<BrainCircuit size={16} /> DeepSeek-R1 (RLVR)
						</div>

						<div className="space-y-3 min-h-[300px]">
							{state.steps.map((step, i) => (
								<div
									key={step.id}
									className={`p-3 rounded border text-sm font-mono transition-all duration-500 animate-in fade-in slide-in-from-left-4 ${
										step.type === "error"
											? "bg-red-900/20 border-red-500/50 text-red-300"
											: step.type === "correction"
												? "bg-amber-900/20 border-amber-500/50 text-amber-300"
												: step.type === "verified"
													? "bg-green-900/20 border-green-500/50 text-green-300 font-bold"
													: "bg-slate-800 border-slate-700 text-slate-300"
									}`}
								>
									<div className="flex items-start gap-3">
										<div className="mt-1">
											{step.type === "error" && <XCircle size={14} />}
											{step.type === "correction" && (
												<AlertTriangle size={14} />
											)}
											{step.type === "verified" && <CheckCircle size={14} />}
											{step.type === "thought" && (
												<span className="text-slate-500 text-xs">#{i + 1}</span>
											)}
										</div>
										<div>
											{step.type === "correction" && (
												<span className="text-xs uppercase font-bold text-amber-500 block mb-1">
													Self-Correction
												</span>
											)}
											{step.text}
										</div>
									</div>
								</div>
							))}
							{state.status === "thinking" && (
								<div className="flex items-center gap-2 text-slate-500 text-xs animate-pulse px-4">
									<span className="w-2 h-2 bg-slate-500 rounded-full" />{" "}
									Thinking...
								</div>
							)}
						</div>

						{/* Verifier Status */}
						<div
							className={`mt-auto p-6 rounded border-2 border-dashed flex flex-col items-center justify-center gap-2 transition-all duration-500 ${
								state.status === "success"
									? "border-green-500 bg-green-500/10"
									: "border-slate-700 bg-slate-900"
							}`}
						>
							{state.status === "success" ? (
								<>
									<CheckCircle size={48} className="text-green-500" />
									<div className="text-green-400 font-bold">REWARD: +1.0</div>
								</>
							) : (
								<>
									<Terminal size={48} className="text-slate-700" />
									<div className="text-slate-600 font-bold text-xs uppercase">
										Awaiting Verification
									</div>
								</>
							)}
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-amber-900/10 border border-amber-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-amber-400 uppercase mb-1">
							What to watch for: Self-Correction
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice the{" "}
							<span className="text-amber-400 font-bold">"Aha" moment</span> at
							step #4. Unlike standard models that commit to their first guess,
							RLVR-trained models learn to backtrack and fix errors when they
							detect a logical inconsistency.
						</p>
					</div>
					<div className="p-3 bg-green-900/10 border border-green-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-green-400 uppercase mb-1">
							What to watch for: Verifiable Rewards
						</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The model is only rewarded when the{" "}
							<span className="text-green-400 font-bold">Verifier</span>{" "}
							confirms the final answer is correct. This objective feedback loop
							eliminates the "sycophancy" common in human-labeled datasets.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_REASONING" : "START_REASONING"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default RLVRSimulation;
