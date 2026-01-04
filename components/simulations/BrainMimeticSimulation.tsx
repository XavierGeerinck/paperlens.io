import React from "react";
import {
	BrainCircuit,
	Database,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface BrainState {
	token: number;
	staticMemory: string[];
	plasticMemory: string[];
	surprise: number;
	loss: number;
}

const BrainMimeticSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			token: 0,
			staticMemory: [] as string[],
			plasticMemory: [] as string[],
			surprise: 0,
			loss: 2.5,
		},
		onTick: (prev) => {
			const tokens = [
				"The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
				"Wait", "the", "dog", "is", "actually", "a", "robot", "!!!"
			];

			if (prev.token >= tokens.length) {
				stop();
				return prev;
			}

			const currentToken = tokens[prev.token];
			const isSurprising = ["robot", "!!!"].includes(currentToken);
			const surpriseVal = isSurprising ? 0.9 : 0.1 + Math.random() * 0.1;

			return {
				...prev,
				token: prev.token + 1,
				staticMemory: [...prev.staticMemory, currentToken].slice(-5),
				plasticMemory: isSurprising 
					? [...prev.plasticMemory, currentToken].slice(-5)
					: prev.plasticMemory,
				surprise: surpriseVal,
				loss: Math.max(0.1, prev.loss * 0.9 - (isSurprising ? 0.5 : 0)),
			};
		},
		tickRate: 800,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="TEST_TIME_PLASTICITY_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: STATIC (TRANSFORMER) */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-80">
						<div className="flex items-center gap-2 text-slate-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Database size={16} /> Static (Transformer)
						</div>

						<div className="p-4 bg-slate-950 rounded border border-slate-800 min-h-[120px]">
							<div className="text-[10px] text-slate-500 uppercase mb-2">KV Cache (Fixed Window)</div>
							<div className="flex flex-wrap gap-2">
								{state.staticMemory.map((t, i) => (
									<span key={i} className="px-2 py-1 bg-slate-800 rounded text-xs font-mono text-slate-400">
										{t}
									</span>
								))}
								{state.staticMemory.length === 0 && <span className="text-slate-700 italic text-xs">Empty...</span>}
							</div>
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
							Standard Transformers use a <span className="text-slate-300 font-bold">Read-Only</span> KV cache. They can only "see" what fits in the window and cannot learn new patterns during inference.
						</div>
					</div>

					{/* RIGHT: PLASTIC (TITANS) */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-purple-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<BrainCircuit size={16} /> Plastic (Titans)
						</div>

						<div className="p-4 bg-slate-950 rounded border border-purple-500/30 min-h-[120px] relative overflow-hidden">
							<div className="text-[10px] text-purple-400 uppercase mb-2">Neural Memory (Weights)</div>
							<div className="flex flex-wrap gap-2">
								{state.plasticMemory.map((t, i) => (
									<span key={i} className="px-2 py-1 bg-purple-900/30 border border-purple-500/50 rounded text-xs font-mono text-purple-200 animate-pulse">
										{t}
									</span>
								))}
								{state.plasticMemory.length === 0 && <span className="text-slate-700 italic text-xs">Awaiting Surprise...</span>}
							</div>
							{state.surprise > 0.5 && (
								<div className="absolute inset-0 bg-purple-500/5 animate-pulse pointer-events-none" />
							)}
						</div>

						<div className="grid grid-cols-2 gap-4">
							<div className="p-2 bg-slate-950 rounded border border-slate-800">
								<div className="text-[9px] text-slate-500 uppercase">Surprise</div>
								<div className={`text-lg font-mono font-bold ${state.surprise > 0.5 ? "text-amber-400" : "text-slate-400"}`}>
									{(state.surprise * 100).toFixed(1)}%
								</div>
							</div>
							<div className="p-2 bg-slate-950 rounded border border-slate-800">
								<div className="text-[9px] text-slate-500 uppercase">Loss</div>
								<div className="text-lg font-mono font-bold text-emerald-400">
									{state.loss.toFixed(3)}
								</div>
							</div>
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-purple-900/10 border border-purple-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-purple-400 uppercase mb-1">What to watch for: Neural Memory</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Unlike the static cache, the <span className="text-purple-400 font-bold">Neural Memory</span> updates its internal weights during inference. It "learns" the robot dog fact physically, allowing for infinite context recall.
						</p>
					</div>
					<div className="p-3 bg-amber-900/10 border border-amber-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-amber-400 uppercase mb-1">What to watch for: Surprise Gating</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Updates only happen when <span className="text-amber-400 font-bold">Surprise</span> is high. This mimics biological plasticity, where the brain only rewires itself when encountering novel or contradictory information.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_INFERENCE" : "START_INFERENCE"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default BrainMimeticSimulation;
