import React, { useState, useEffect } from "react";
import {
	BrainCircuit,
	Network,
	Microscope,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const INPUT_STREAM = [
	{ id: "geo", label: "Golden Gate Bridge", category: "Geography", color: "#3b82f6" },
	{ id: "code", label: "def train_model():", category: "Code", color: "#10b981" },
	{ id: "math", label: "E = mc²", category: "Physics", color: "#f59e0b" },
	{ id: "geo", label: "Alcatraz Island", category: "Geography", color: "#3b82f6" },
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
			const neuronAct = 0.8 + Math.random() * 0.2;
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
		tickRate: 1200,
	});

	useEffect(() => {
		if (!isRunning) setStepIndex(0);
	}, [isRunning]);

	const currentInput = state.currentInput || INPUT_STREAM[0];

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="SAE_DISENTANGLEMENT_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: RAW NEURON (POLYSEMANTIC) */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-80">
						<div className="flex items-center gap-2 text-slate-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Network size={16} /> Raw Neuron #4096
						</div>

						<div className="p-4 bg-slate-950 rounded border border-slate-800 min-h-[160px] flex flex-col items-center justify-center gap-4">
							<div className="text-[10px] text-slate-500 uppercase">Activation Level</div>
							<div 
								className="w-16 h-16 rounded-full border-2 flex items-center justify-center transition-all duration-300"
								style={{
									backgroundColor: isRunning ? `rgba(255, 255, 255, ${state.neuronActivation})` : "transparent",
									borderColor: isRunning ? "#fff" : "#334155",
									boxShadow: isRunning ? "0 0 20px rgba(255,255,255,0.2)" : "none"
								}}
							>
								<BrainCircuit size={24} className={isRunning ? "text-black" : "text-slate-600"} />
							</div>
							<div className="text-xs font-mono text-slate-400">
								{isRunning ? `Input: "${currentInput.label}"` : "Waiting..."}
							</div>
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
							Raw neurons are <span className="text-slate-300 font-bold">Polysemantic</span>. This single neuron fires for Geography, Code, and Physics because concepts are stored in superposition.
						</div>
					</div>

					{/* RIGHT: SAE FEATURES (MONOSEMANTIC) */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-blue-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Microscope size={16} /> SAE Features
						</div>

						<div className="p-4 bg-slate-950 rounded border border-blue-500/30 min-h-[160px] flex flex-col gap-3">
							<div className="text-[10px] text-blue-400 uppercase mb-1">Disentangled Concepts</div>
							{[
								{ label: "Geography", val: state.saeFeatures[0], color: "bg-blue-500" },
								{ label: "Python Code", val: state.saeFeatures[1], color: "bg-emerald-500" },
								{ label: "Physics", val: state.saeFeatures[2], color: "bg-amber-500" },
							].map((f, i) => (
								<div key={i} className="space-y-1">
									<div className="flex justify-between text-[10px] font-mono">
										<span className="text-slate-400">{f.label}</span>
										<span className="text-slate-500">{f.val.toFixed(2)}</span>
									</div>
									<div className="h-1 bg-slate-800 rounded-full overflow-hidden">
										<div 
											className={`h-full ${f.color} transition-all duration-300`}
											style={{ width: `${f.val * 100}%` }}
										/>
									</div>
								</div>
							))}
						</div>

						<div className="p-3 bg-blue-900/10 rounded border border-blue-500/20 text-[11px] text-slate-400 leading-relaxed">
							The SAE acts as a <span className="text-blue-400 font-bold">Rosetta Stone</span>, mapping the messy raw activations into clean, human-interpretable features.
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">What to watch for: Superposition</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Notice how the <span className="text-emerald-400 font-bold">Raw Neuron</span> fires for every input. It's impossible to tell what the model is "thinking" just by looking at raw activations.
						</p>
					</div>
					<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">What to watch for: Sparsity</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							In the SAE view, only <span className="text-blue-400 font-bold">one feature</span> fires at a time. This sparsity is what makes the model's internal logic interpretable to humans.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_DECODING" : "START_DECODING"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default MappingTheMindSimulation;
