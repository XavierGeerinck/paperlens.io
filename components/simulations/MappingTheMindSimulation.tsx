import React, { useState, useEffect } from "react";
import {
	Play,
	RotateCcw,
	BrainCircuit,
	Search,
	Zap,
	Info,
	ArrowRight,
	Network,
	Microscope,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const INPUT_STREAM = [
	{
		id: "geo",
		label: "Golden Gate Bridge",
		category: "Geography",
		color: "#3b82f6",
	},
	{
		id: "code",
		label: "def train_model():",
		category: "Code",
		color: "#10b981",
	},
	{ id: "math", label: "E = mc²", category: "Physics", color: "#f59e0b" },
	{
		id: "geo",
		label: "Alcatraz Island",
		category: "Geography",
		color: "#3b82f6",
	},
	{ id: "code", label: "import torch", category: "Code", color: "#10b981" },
	{ id: "math", label: "F = ma", category: "Physics", color: "#f59e0b" },
];

const MappingTheMindSimulation: React.FC = () => {
	const [stepIndex, setStepIndex] = useState(0);

	const { isRunning, start, stop, reset, state } = useSimulation({
		initialState: {
			neuronActivation: 0,
			saeFeatures: [0, 0, 0], // [Geo, Code, Physics]
			currentInput: null as (typeof INPUT_STREAM)[0] | null,
		},
		onTick: (prev) => {
			const input = INPUT_STREAM[stepIndex % INPUT_STREAM.length];

			// The polysemantic neuron fires for EVERYTHING (superposition)
			const neuronAct = 0.8 + Math.random() * 0.2;

			// The SAE disentangles it into specific features
			const newFeatures = [0, 0, 0];
			if (input.id === "geo") newFeatures[0] = 0.9 + Math.random() * 0.1;
			if (input.id === "code") newFeatures[1] = 0.9 + Math.random() * 0.1;
			if (input.id === "math") newFeatures[2] = 0.9 + Math.random() * 0.1;

			setStepIndex((s) => s + 1);

			return {
				neuronActivation: neuronAct,
				saeFeatures: newFeatures,
				currentInput: input,
			};
		},
		tickRate: 1500, // Slow enough to read
	});

	// Reset helper
	useEffect(() => {
		if (!isRunning) {
			setStepIndex(0);
		}
	}, [isRunning]);

	const currentInput = state.currentInput || INPUT_STREAM[0];

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="SAE_DISENTANGLEMENT_PROCESS">
				{/* Visualization Pipeline */}
				<div className="flex flex-col md:flex-row items-center justify-between gap-4 py-8 px-2">
					{/* 1. Input Stream */}
					<div className="flex flex-col items-center gap-2 w-full md:w-1/4">
						<div className="text-xs font-mono text-slate-500 uppercase mb-2">
							1. Model Input
						</div>
						<div
							className="h-16 w-full rounded border border-slate-700 bg-slate-800 flex items-center justify-center text-center px-2 transition-all duration-300"
							style={{
								borderColor: isRunning ? currentInput.color : "#334155",
							}}
						>
							<span
								className="font-mono font-bold"
								style={{ color: isRunning ? currentInput.color : "#94a3b8" }}
							>
								{isRunning ? `"${currentInput.label}"` : "Waiting..."}
							</span>
						</div>
					</div>

					<ArrowRight className="text-slate-600 hidden md:block" />

					{/* 2. Polysemantic Neuron (The Black Box) */}
					<div className="flex flex-col items-center gap-2 w-full md:w-1/4">
						<div className="text-xs font-mono text-slate-500 uppercase mb-2">
							2. Neuron #4096
						</div>
						<div className="relative">
							<div
								className="w-24 h-24 rounded-full border-4 flex items-center justify-center transition-all duration-300 shadow-[0_0_30px_rgba(255,255,255,0.1)]"
								style={{
									backgroundColor: isRunning
										? `rgba(255, 255, 255, ${state.neuronActivation})`
										: "#1e293b",
									borderColor: isRunning ? "#fff" : "#334155",
									transform: isRunning ? "scale(1.1)" : "scale(1)",
								}}
							>
								<BrainCircuit
									size={32}
									className={isRunning ? "text-black" : "text-slate-600"}
								/>
							</div>
							{isRunning && (
								<div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-xs font-mono text-white bg-slate-800 px-2 py-1 rounded whitespace-nowrap">
									Activation: {state.neuronActivation.toFixed(2)}
								</div>
							)}
						</div>
						<div className="text-xs text-slate-400 text-center mt-8 max-w-[150px]">
							Fires for <span className="text-white font-bold">ALL</span> inputs
							(Superposition)
						</div>
					</div>

					<ArrowRight className="text-slate-600 hidden md:block" />

					{/* 3. SAE Features (The Rosetta Stone) */}
					<div className="flex flex-col items-center gap-2 w-full md:w-1/3">
						<div className="text-xs font-mono text-slate-500 uppercase mb-2">
							3. Sparse Features (SAE)
						</div>
						<div className="grid grid-cols-1 gap-3 w-full">
							{/* Feature 1: Geography */}
							<div className="flex items-center gap-3 p-2 rounded bg-slate-950 border border-slate-800">
								<div
									className="w-3 h-3 rounded-full transition-all duration-300"
									style={{
										backgroundColor:
											state.saeFeatures[0] > 0.5 ? "#3b82f6" : "#1e293b",
										boxShadow:
											state.saeFeatures[0] > 0.5 ? "0 0 10px #3b82f6" : "none",
									}}
								/>
								<div className="flex-1">
									<div className="flex justify-between text-xs mb-1">
										<span className="text-blue-400 font-bold">
											Feature A: Geography
										</span>
										<span className="font-mono text-slate-500">
											{state.saeFeatures[0].toFixed(2)}
										</span>
									</div>
									<div className="h-1 bg-slate-800 rounded-full overflow-hidden">
										<div
											className="h-full bg-blue-500 transition-all duration-300"
											style={{ width: `${state.saeFeatures[0] * 100}%` }}
										/>
									</div>
								</div>
							</div>

							{/* Feature 2: Code */}
							<div className="flex items-center gap-3 p-2 rounded bg-slate-950 border border-slate-800">
								<div
									className="w-3 h-3 rounded-full transition-all duration-300"
									style={{
										backgroundColor:
											state.saeFeatures[1] > 0.5 ? "#10b981" : "#1e293b",
										boxShadow:
											state.saeFeatures[1] > 0.5 ? "0 0 10px #10b981" : "none",
									}}
								/>
								<div className="flex-1">
									<div className="flex justify-between text-xs mb-1">
										<span className="text-emerald-400 font-bold">
											Feature B: Python Code
										</span>
										<span className="font-mono text-slate-500">
											{state.saeFeatures[1].toFixed(2)}
										</span>
									</div>
									<div className="h-1 bg-slate-800 rounded-full overflow-hidden">
										<div
											className="h-full bg-emerald-500 transition-all duration-300"
											style={{ width: `${state.saeFeatures[1] * 100}%` }}
										/>
									</div>
								</div>
							</div>

							{/* Feature 3: Physics */}
							<div className="flex items-center gap-3 p-2 rounded bg-slate-950 border border-slate-800">
								<div
									className="w-3 h-3 rounded-full transition-all duration-300"
									style={{
										backgroundColor:
											state.saeFeatures[2] > 0.5 ? "#f59e0b" : "#1e293b",
										boxShadow:
											state.saeFeatures[2] > 0.5 ? "0 0 10px #f59e0b" : "none",
									}}
								/>
								<div className="flex-1">
									<div className="flex justify-between text-xs mb-1">
										<span className="text-amber-400 font-bold">
											Feature C: Physics
										</span>
										<span className="font-mono text-slate-500">
											{state.saeFeatures[2].toFixed(2)}
										</span>
									</div>
									<div className="h-1 bg-slate-800 rounded-full overflow-hidden">
										<div
											className="h-full bg-amber-500 transition-all duration-300"
											style={{ width: `${state.saeFeatures[2] * 100}%` }}
										/>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "PAUSE_ANALYSIS" : "START_DECODING"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>

			<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
				<div className="p-3 bg-slate-800/50 rounded border border-slate-700 flex gap-3 items-start">
					<Network className="text-purple-400 shrink-0 mt-1" size={20} />
					<div>
						<div className="text-sm font-bold text-slate-200 mb-1">
							Polysemanticity
						</div>
						<div className="text-xs text-slate-400">
							Notice how the central neuron fires for{" "}
							<span className="text-white">every</span> input? It's doing "too
							much work", representing multiple unrelated concepts in
							superposition.
						</div>
					</div>
				</div>
				<div className="p-3 bg-slate-800/50 rounded border border-slate-700 flex gap-3 items-start">
					<Microscope className="text-blue-400 shrink-0 mt-1" size={20} />
					<div>
						<div className="text-sm font-bold text-slate-200 mb-1">
							Sparse Resolution
						</div>
						<div className="text-xs text-slate-400">
							The SAE expands the signal (often 16x-64x) and applies a sparsity
							filter. This forces the signal to resolve into distinct,
							human-readable features.
						</div>
					</div>
				</div>
			</div>
		</div>
	);
};

export default MappingTheMindSimulation;
