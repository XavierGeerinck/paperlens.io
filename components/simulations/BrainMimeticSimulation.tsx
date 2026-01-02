import React, { useRef, useEffect } from "react";
import { Play, RotateCcw, Database, BrainCircuit, Pause } from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

// --- Brain Mimetic Logic ---
interface BrainMimeticState {
	loss: number;
	surprise: number;
	memory: number;
	alpha: number;
	eta: number;
	theta: number;
}

const BRAIN_MIMETIC_INIT: BrainMimeticState = {
	loss: 2.5,
	surprise: 0.8,
	memory: 0,
	alpha: 0.1,
	eta: 0.9,
	theta: 0.01,
};

const brainMimeticTick = (
	prev: BrainMimeticState,
	tick: number,
): Partial<BrainMimeticState> => {
	const newLoss = Math.max(
		0.1,
		prev.loss * 0.95 + (Math.random() * 0.1 - 0.05),
	);
	const newSurprise = Math.max(0, prev.surprise * 0.9 + Math.random() * 0.2);
	return {
		loss: newLoss,
		surprise: newSurprise,
		memory: Math.min(100, prev.memory + 0.8),
		alpha: 0.1 + Math.sin(tick / 10) * 0.05,
		eta: 0.9 - newSurprise * 0.1,
		theta: 0.01 + newSurprise * 0.05,
	};
};

const brainMimeticLog = (state: BrainMimeticState) => {
	if (Math.random() > 0.6)
		return `[TITANS] Surprise: ${state.surprise.toFixed(4)} | Loss: ${state.loss.toFixed(4)}`;
	return null;
};

const renderGraphLine = (
	data: number[] | undefined,
	max: number,
	color: string,
) => {
	if (!data || data.length < 2) return null;
	const width = 100;
	const height = 40;
	const step = width / (Math.max(data.length, 50) - 1);

	const points = data
		.map((val, i) => {
			const x = i * step;
			const normalizedY = Math.min(val, max) / max;
			const y = height - normalizedY * height;
			return `${x},${y}`;
		})
		.join(" ");

	return (
		<polyline
			points={points}
			fill="none"
			stroke={color}
			strokeWidth="1.5"
			vectorEffect="non-scaling-stroke"
			strokeLinecap="round"
			strokeLinejoin="round"
		/>
	);
};

const BrainMimeticSimulation: React.FC = () => {
	const logsContainerRef = useRef<HTMLDivElement>(null);
	const { isRunning, state, logs, history, epoch, start, stop, reset } =
		useSimulation<BrainMimeticState>({
			initialState: BRAIN_MIMETIC_INIT,
			onTick: brainMimeticTick,
			onLog: brainMimeticLog,
			tickRate: 200,
		});

	useEffect(() => {
		if (logsContainerRef.current) {
			logsContainerRef.current.scrollTop =
				logsContainerRef.current.scrollHeight;
		}
	}, [logs]);

	return (
		<div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-500">
			<div className="lg:col-span-2 space-y-6">
				<SchematicCard title="TRAINING_ENVIRONMENT">
					<div className="flex items-center justify-between mb-8">
						<div className="flex items-center gap-2 text-sky-400 font-mono text-sm">
							<BrainCircuit className="w-4 h-4" />
							<span>BRAIN_MIMETIC_MODEL_V1</span>
						</div>
						<div className="flex gap-2">
							{!isRunning ? (
								<SchematicButton onClick={start}>
									<Play className="w-3 h-3 inline mr-2" /> INITIATE
								</SchematicButton>
							) : (
								<SchematicButton onClick={stop}>
									<Pause className="w-3 h-3 inline mr-2" /> HALT
								</SchematicButton>
							)}
							<button
								onClick={reset}
								className="p-3 border border-slate-700 text-slate-400 hover:text-white transition-colors"
							>
								<RotateCcw className="w-4 h-4" />
							</button>
						</div>
					</div>

					<div className="grid grid-cols-2 gap-4">
						{/* Graph 1: Loss */}
						<div className="bg-slate-950 border border-slate-800 p-4 relative">
							<div className="text-[10px] font-mono text-slate-500 uppercase mb-2 flex justify-between">
								<span>Loss (MSE)</span>
								<span className="text-slate-200">{state.loss.toFixed(4)}</span>
							</div>
							<div className="h-24 w-full">
								<svg
									viewBox="0 0 100 40"
									preserveAspectRatio="none"
									className="w-full h-full overflow-visible"
								>
									{renderGraphLine(history["loss"], 3, "#f43f5e")}
								</svg>
							</div>
						</div>

						{/* Graph 2: Surprise */}
						<div className="bg-slate-950 border border-slate-800 p-4 relative">
							<div className="text-[10px] font-mono text-slate-500 uppercase mb-2 flex justify-between">
								<span>Surprise (∇L)</span>
								<span className="text-slate-200">
									{state.surprise.toFixed(4)}
								</span>
							</div>
							<div className="h-24 w-full">
								<svg
									viewBox="0 0 100 40"
									preserveAspectRatio="none"
									className="w-full h-full overflow-visible"
								>
									{renderGraphLine(history["surprise"], 1.5, "#22d3ee")}
								</svg>
							</div>
						</div>
					</div>

					<div className="mt-6 flex items-center gap-4 text-xs font-mono text-slate-500 border-t border-slate-800 pt-4">
						<div className="flex items-center gap-2 min-w-[140px]">
							<Database className="w-3 h-3 text-sky-500" />
							MEM_LOAD: {state.memory.toFixed(1)}%
						</div>
						<div className="flex-grow h-1 bg-slate-800">
							<div
								className="h-full bg-sky-500 transition-all duration-200"
								style={{ width: `${state.memory}%` }}
							/>
						</div>
						<div>EPOCH {epoch.toString().padStart(4, "0")}</div>
					</div>
				</SchematicCard>

				{/* Code & Vars */}
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<SchematicCard title="KERNEL_LOGIC" className="font-mono text-[10px]">
						<div className="text-slate-600 mb-2">
							def forward(self, x, state):
						</div>
						<div className="text-emerald-500 pl-2">M, S = state</div>
						<div className="text-slate-400 pl-2">surprise = grad(loss, M)</div>
						<div className="text-sky-400 pl-2 mt-1"># Plasticity Update</div>
						<div className="text-slate-300 pl-2">
							M = (1 - {state.alpha.toFixed(2)}) * M
						</div>
						<div className="text-slate-300 pl-6">
							+ {state.eta.toFixed(2)} * S
						</div>
						<div className="text-slate-300 pl-6">
							- {state.theta.toFixed(3)} * surprise
						</div>
					</SchematicCard>

					<SchematicCard title="PARAMETERS">
						<div className="space-y-1">
							<div className="flex justify-between text-xs font-mono border-b border-slate-800 pb-1">
								<span className="text-blue-400">ALPHA (Forget)</span>
								<span className="text-slate-300">{state.alpha.toFixed(4)}</span>
							</div>
							<div className="flex justify-between text-xs font-mono border-b border-slate-800 pb-1">
								<span className="text-yellow-400">ETA (Momentum)</span>
								<span className="text-slate-300">{state.eta.toFixed(4)}</span>
							</div>
							<div className="flex justify-between text-xs font-mono border-b border-slate-800 pb-1">
								<span className="text-red-400">THETA (Learn)</span>
								<span className="text-slate-300">{state.theta.toFixed(4)}</span>
							</div>
						</div>
					</SchematicCard>
				</div>
			</div>

			{/* Logs */}
			<div className="lg:col-span-1">
				<SchematicCard title="SYS_LOGS" className="h-[600px] flex flex-col">
					<div
						ref={logsContainerRef}
						className="flex-grow overflow-y-auto space-y-1 text-slate-500 font-mono text-[10px] scrollbar-hide"
					>
						{logs.map((log, i) => (
							<div
								key={i}
								className="break-all border-l-2 border-slate-800 pl-2"
							>
								<span className="text-slate-700 mr-2">
									{new Date().toLocaleTimeString().split(" ")[0]}
								</span>
								{log}
							</div>
						))}
					</div>
				</SchematicCard>
			</div>
		</div>
	);
};

export default BrainMimeticSimulation;
