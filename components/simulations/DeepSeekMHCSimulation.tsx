import React, { useState, useEffect, useRef } from "react";
import {
	Play,
	RotateCcw,
	Pause,
	AlertTriangle,
	CheckCircle,
	Activity,
	ArrowRight,
	ArrowDown,
	GitCommit,
	Network,
	Split,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import {
	SchematicCard,
	SchematicButton,
	DataLabel,
	TechBadge,
} from "../SketchElements";

// --- DeepSeek mHC Logic ---

interface DeepSeekMHCState {
	mode: "WILD" | "mHC";
	layer: number;
	signal: number[]; // 4 lanes
	matrix: number[][]; // 4x4 weights
	laneHistory: number[][]; // History of 4 lanes for graphing
	status: "STABLE" | "EXPLODED" | "VANISHED";
}

const HISTORY_LENGTH = 60;

const DEEPSEEK_INIT: DeepSeekMHCState = {
	mode: "WILD",
	layer: 0,
	signal: [1.0, 1.2, 0.8, 1.1], // Slightly varied initial signals
	matrix: Array(4)
		.fill(0)
		.map(() => Array(4).fill(0.25)),
	laneHistory: Array(HISTORY_LENGTH).fill([1.0, 1.0, 1.0, 1.0]), // Init with stable values
	status: "STABLE",
};

const generateRandomMatrix = (mode: "WILD" | "mHC") => {
	// In Wild mode, we use higher variance and potentially larger sums
	// In mHC, the raw values don't matter as much before normalization, but we keep them positive
	const scale = mode === "WILD" ? 2.0 : 1.0;
	return Array(4)
		.fill(0)
		.map(() =>
			Array(4)
				.fill(0)
				.map(() => Math.random() * scale),
		);
};

const sinkhornKnopp = (mat: number[][], iterations = 10) => {
	let m = mat.map((row) => [...row]); // Deep copy

	for (let iter = 0; iter < iterations; iter++) {
		// Normalize Rows
		for (let i = 0; i < 4; i++) {
			const sum = m[i].reduce((a, b) => a + b, 0);
			if (sum > 0) m[i] = m[i].map((val) => val / sum);
		}
		// Normalize Cols
		for (let j = 0; j < 4; j++) {
			let sum = 0;
			for (let i = 0; i < 4; i++) sum += m[i][j];
			if (sum > 0) {
				for (let i = 0; i < 4; i++) m[i][j] = m[i][j] / sum;
			}
		}
	}
	return m;
};

const multiplyMatrixVector = (mat: number[][], vec: number[]) => {
	return mat.map((row) => row.reduce((sum, val, i) => sum + val * vec[i], 0));
};

const deepSeekTick = (
	prev: DeepSeekMHCState,
	tick: number,
): Partial<DeepSeekMHCState> => {
	if (prev.status !== "STABLE") return {};

	// 1. Generate Weights
	let matrix = generateRandomMatrix(prev.mode);

	// 2. Apply Constraints
	if (prev.mode === "mHC") {
		matrix = sinkhornKnopp(matrix);
	} else {
		// Wild mode: Normalize rows loosely but introduce drift noise
		const noise = 1.0 + (Math.random() * 0.4 - 0.2); // 0.8 to 1.2 multiplier
		matrix = matrix.map((row) => {
			const sum = row.reduce((a, b) => a + b, 0);
			return row.map((v) => (v / sum) * noise);
		});
	}

	// 3. Propagate Signal
	const newSignal = multiplyMatrixVector(matrix, prev.signal);

	// 4. Update History
	const newHistory = [...prev.laneHistory.slice(1), newSignal];

	// 5. Analyze Health
	const maxVal = Math.max(...newSignal);

	let status: DeepSeekMHCState["status"] = "STABLE";
	if (maxVal > 1000) status = "EXPLODED"; // Higher threshold for visual drama
	if (maxVal < 0.001) status = "VANISHED";

	return {
		layer: prev.layer + 1,
		signal: newSignal,
		matrix: matrix,
		laneHistory: newHistory,
		status: status,
	};
};

const AlgorithmFlowchart: React.FC<{ mode: "WILD" | "mHC" }> = ({ mode }) => {
	return (
		<div className="flex flex-col items-center gap-2 p-4 bg-slate-900/30 border border-slate-800 rounded-sm font-mono text-[9px] text-slate-400 select-none">
			{/* Input Node */}
			<div className="px-3 py-1.5 bg-slate-800 border border-slate-600 rounded flex items-center gap-2 text-slate-300">
				<GitCommit className="w-3 h-3 text-sky-400" />
				<span>Input Signal X</span>
			</div>

			<ArrowDown className="w-3 h-3 text-slate-600" />

			{/* Logic Split */}
			<div className="flex gap-2 w-full">
				<div
					className={`flex-1 flex flex-col items-center gap-2 p-2 border border-dashed rounded transition-colors ${mode === "WILD" ? "border-red-500/50 bg-red-900/10" : "border-slate-800 opacity-30"}`}
				>
					<div className="text-center font-bold text-red-400">
						Standard Init
					</div>
					<div className="text-[8px] text-center leading-tight">
						Random Weights
						<br />W ~ N(0, σ)
					</div>
				</div>

				<div
					className={`flex-1 flex flex-col items-center gap-2 p-2 border border-dashed rounded transition-colors ${mode === "mHC" ? "border-emerald-500/50 bg-emerald-900/10" : "border-slate-800 opacity-30"}`}
				>
					<div className="text-center font-bold text-emerald-400">
						mHC Constraint
					</div>
					<div className="text-[8px] text-center leading-tight">
						Sinkhorn-Knopp
						<br />
						Doubly Stochastic
					</div>
				</div>
			</div>

			<ArrowDown className="w-3 h-3 text-slate-600" />

			{/* Matrix Mult */}
			<div className="px-3 py-1.5 bg-slate-800 border border-slate-600 rounded flex items-center gap-2 text-slate-300 relative group">
				<Network className="w-3 h-3 text-indigo-400" />
				<span>Y = Wx</span>
				{mode === "mHC" && (
					<div className="absolute -right-2 -top-2 w-2 h-2 bg-emerald-500 rounded-full animate-ping" />
				)}
			</div>

			<ArrowDown className="w-3 h-3 text-slate-600" />

			{/* Result */}
			<div
				className={`px-3 py-1.5 border rounded flex items-center gap-2 transition-colors ${mode === "mHC" ? "border-emerald-500/50 text-emerald-300" : "border-red-500/50 text-red-300"}`}
			>
				<span>
					{mode === "mHC" ? "Stable Propagation" : "Drift & Collapse"}
				</span>
			</div>
		</div>
	);
};

const NetworkTopology: React.FC<{
	prevSignal: number[];
	currSignal: number[];
	matrix: number[][];
	mode: string;
}> = ({ prevSignal, currSignal, matrix, mode }) => {
	const height = 200;
	const width = 400;
	const nodeRadius = 8;
	const colors = ["#f472b6", "#38bdf8", "#a78bfa", "#34d399"]; // Pink, Blue, Purple, Emerald

	// Helper to log scale radius
	const getRadius = (val: number) =>
		Math.max(2, Math.min(25, Math.log2(val + 1) * 6 + 2));

	return (
		<div className="w-full h-[200px] bg-slate-950 border border-slate-800 relative overflow-hidden flex items-center justify-center">
			<div className="absolute top-2 left-2 text-[10px] font-mono text-zinc-600 uppercase tracking-widest">
				Topology View
			</div>
			<svg
				width="100%"
				height="100%"
				viewBox={`0 0 ${width} ${height}`}
				preserveAspectRatio="xMidYMid meet"
			>
				{/* Links */}
				{matrix.map((row, rowIdx) => {
					// rowIdx is Output index (y)
					return row.map((weight, colIdx) => {
						// colIdx is Input index (x)
						const startX = 60;
						const startY = 40 + colIdx * 40;
						const endX = width - 60;
						const endY = 40 + rowIdx * 40;

						// Bezier
						const path = `M ${startX} ${startY} C ${startX + 100} ${startY}, ${endX - 100} ${endY}, ${endX} ${endY}`;

						return (
							<path
								key={`link-${colIdx}-${rowIdx}`}
								d={path}
								fill="none"
								stroke={colors[colIdx]}
								strokeWidth={Math.max(0.5, weight * 4)}
								opacity={Math.max(0.1, Math.min(0.8, weight))}
								className="transition-all duration-300"
							/>
						);
					});
				})}

				{/* Input Nodes (Previous Signal) */}
				{prevSignal.map((val, i) => {
					const r = getRadius(val);
					const y = 40 + i * 40;
					return (
						<g key={`in-${i}`} className="transition-all duration-300">
							<circle
								cx={60}
								cy={y}
								r={r}
								fill={colors[i]}
								stroke="#1e293b"
								strokeWidth="2"
								fillOpacity={0.8}
							/>
							<text
								x={20}
								y={y + 3}
								textAnchor="end"
								className="text-[10px] fill-slate-500 font-mono"
							>
								{val.toFixed(2)}
							</text>
						</g>
					);
				})}

				{/* Output Nodes (Current Signal) */}
				{currSignal.map((val, i) => {
					const r = getRadius(val);
					const y = 40 + i * 40;
					return (
						<g key={`out-${i}`} className="transition-all duration-300">
							<circle
								cx={width - 60}
								cy={y}
								r={r}
								fill={colors[i]}
								stroke="#1e293b"
								strokeWidth="2"
								fillOpacity={0.8}
							/>
							<text
								x={width - 20}
								y={y + 3}
								textAnchor="start"
								className="text-[10px] fill-slate-500 font-mono"
							>
								{val.toFixed(2)}
							</text>
						</g>
					);
				})}

				{/* Labels */}
				<text
					x={60}
					y={20}
					textAnchor="middle"
					className="text-[9px] fill-slate-500 font-mono uppercase"
				>
					Layer {mode === "mHC" ? "N" : "N"}
				</text>
				<text
					x={width - 60}
					y={20}
					textAnchor="middle"
					className="text-[9px] fill-slate-500 font-mono uppercase"
				>
					Layer {mode === "mHC" ? "N+1" : "N+1"}
				</text>
			</svg>
		</div>
	);
};

const DeepSeekMHCSimulation: React.FC = () => {
	const [dsMode, setDsMode] = useState<"WILD" | "mHC">("mHC"); // Default to mHC for better first impression

	// Inject mode into tick
	const effectiveTick = (prev: DeepSeekMHCState, tick: number) => {
		return deepSeekTick({ ...prev, mode: dsMode }, tick);
	};

	const sim = useSimulation<DeepSeekMHCState>({
		initialState: DEEPSEEK_INIT,
		onTick: effectiveTick,
		tickRate: 100, // Faster for better flow
	});

	// Reset when switching modes
	useEffect(() => {
		sim.reset();
	}, [dsMode]);

	const prevSignal = sim.state.laneHistory[
		sim.state.laneHistory.length - 2
	] || [1, 1, 1, 1];

	// Render multi-line graph
	const renderMultiLineGraph = () => {
		const width = 100;
		const height = 100;
		const data = sim.state.laneHistory;
		const step = width / (HISTORY_LENGTH - 1);

		// Colors for 4 lanes
		const colors = ["#f472b6", "#38bdf8", "#a78bfa", "#34d399"]; // Pink, Blue, Purple, Emerald

		const getY = (val: number) => {
			if (val <= 0) return 95;
			// Log scale: Center 1.0 at 50%
			// Pad 5% top and bottom to ensure stroke is visible
			const logVal = Math.log2(val);
			const y = 50 - logVal * 15;
			return Math.max(5, Math.min(95, y));
		};

		return (
			<svg
				viewBox="0 0 100 100"
				preserveAspectRatio="none"
				className="w-full h-full"
			>
				{/* Zones */}
				<rect x="0" y="0" width="100" height="20" fill="rgba(239,68,68,0.05)" />
				<rect
					x="0"
					y="80"
					width="100"
					height="20"
					fill="rgba(239,68,68,0.05)"
				/>

				{/* Reference Line (Signal = 1.0) */}
				<line
					x1="0"
					y1="50"
					x2="100"
					y2="50"
					stroke="#334155"
					strokeWidth="0.5"
					strokeDasharray="2 2"
				/>

				{/* Lanes */}
				{[0, 1, 2, 3].map((laneIdx) => {
					const points = data
						.map((vec, t) => {
							const x = t * step;
							const val = vec[laneIdx];
							const y = getY(val);
							return `${x},${y}`;
						})
						.join(" ");

					return (
						<polyline
							key={laneIdx}
							points={points}
							fill="none"
							stroke={colors[laneIdx]}
							strokeWidth="1.5"
							vectorEffect="non-scaling-stroke"
							strokeLinecap="round"
							strokeLinejoin="round"
							opacity={sim.state.signal[laneIdx] === 0 ? 0.2 : 0.9}
						/>
					);
				})}
			</svg>
		);
	};

	return (
		<div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-in fade-in duration-500 font-mono">
			{/* Left Column: Controls & Matrix (4 cols) */}
			<div className="lg:col-span-4 space-y-6">
				<SchematicCard title="CONTROLLER">
					<div className="flex flex-col gap-4">
						<div className="flex bg-slate-900 p-1 rounded border border-slate-800">
							<button
								onClick={() => setDsMode("WILD")}
								className={`flex-1 py-2 text-xs font-bold transition-all ${dsMode === "WILD" ? "bg-red-500/20 text-red-500 shadow-[0_0_10px_rgba(239,68,68,0.2)]" : "text-slate-500 hover:text-slate-300"}`}
							>
								WILD MODE
							</button>
							<button
								onClick={() => setDsMode("mHC")}
								className={`flex-1 py-2 text-xs font-bold transition-all ${dsMode === "mHC" ? "bg-emerald-500/20 text-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.2)]" : "text-slate-500 hover:text-slate-300"}`}
							>
								mHC PROTOCOL
							</button>
						</div>

						<div className="flex gap-2">
							{!sim.isRunning && sim.state.status === "STABLE" ? (
								<SchematicButton onClick={sim.start}>
									<Play className="w-3 h-3 inline mr-2" /> SIMULATE
								</SchematicButton>
							) : (
								<SchematicButton onClick={sim.stop}>
									<Pause className="w-3 h-3 inline mr-2" /> PAUSE
								</SchematicButton>
							)}
							<button
								onClick={sim.reset}
								className="p-3 border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
							>
								<RotateCcw className="w-4 h-4" />
							</button>
						</div>

						<div className="pt-4 border-t border-slate-800">
							<DataLabel
								label="DEPTH (LAYERS)"
								value={sim.state.layer.toString().padStart(4, "0")}
							/>
							<div className="mt-4">
								<span className="text-[9px] font-mono text-zinc-600 uppercase tracking-widest mb-0.5 block">
									SYSTEM STATUS
								</span>
								<div
									className={`flex items-center gap-2 font-bold ${
										sim.state.status === "STABLE"
											? "text-emerald-500"
											: sim.state.status === "EXPLODED"
												? "text-red-500"
												: "text-slate-500"
									}`}
								>
									{sim.state.status === "STABLE" && (
										<CheckCircle className="w-4 h-4" />
									)}
									{sim.state.status !== "STABLE" && (
										<AlertTriangle className="w-4 h-4" />
									)}
									{sim.state.status}
								</div>
							</div>
						</div>
					</div>
				</SchematicCard>

				{/* Algorithm Visualizer (Mermaid-style) */}
				<SchematicCard title="PROTOCOL_LOGIC">
					<AlgorithmFlowchart mode={dsMode} />
				</SchematicCard>

				<SchematicCard title="WEIGHT_MATRIX (ROUTER)">
					<div className="aspect-square relative grid grid-cols-4 gap-1 p-1 bg-slate-900 border border-slate-800">
						{sim.state.matrix.map((row, r) =>
							row.map((val, c) => (
								<div key={`${r}-${c}`} className="relative group">
									<div
										className={`absolute inset-0 transition-opacity duration-300 ${dsMode === "mHC" ? "bg-emerald-500" : "bg-rose-500"}`}
										style={{ opacity: Math.min(val, 1.5) * 0.6 }}
									/>
									{/* Connection Line Visual - Only show stronger connections */}
									{val > 0.5 && (
										<div className="absolute inset-0 flex items-center justify-center opacity-30">
											<ArrowRight className="w-3 h-3 text-white transform -rotate-45" />
										</div>
									)}
								</div>
							)),
						)}
					</div>
					<div className="mt-2 text-[9px] text-slate-500 text-center">
						{dsMode === "mHC"
							? "Matrix constrained to Doubly Stochastic manifold"
							: "Matrix unconstrained (Random Init)"}
					</div>
				</SchematicCard>
			</div>

			{/* Right Column: Signal Visualization (8 cols) */}
			<div className="lg:col-span-8 space-y-6">
				{/* Main Graph */}
				<SchematicCard title="SIGNAL_PROPAGATION_MONITOR">
					<div className="h-64 flex flex-col w-full bg-slate-950 border border-slate-800 relative overflow-hidden">
						{/* Background Grid */}
						<div className="absolute inset-0 grid grid-cols-6 grid-rows-4 pointer-events-none">
							{[...Array(6)].map((_, i) => (
								<div
									key={`v-${i}`}
									className="border-r border-slate-800/30 h-full"
								/>
							))}
							{[...Array(4)].map((_, i) => (
								<div
									key={`h-${i}`}
									className="border-b border-slate-800/30 w-full"
								/>
							))}
						</div>

						{/* The Graph */}
						{renderMultiLineGraph()}

						{/* Overlays */}
						{sim.state.status === "EXPLODED" && (
							<div className="absolute inset-0 bg-red-500/20 flex items-center justify-center backdrop-blur-sm">
								<div className="bg-red-950/90 border border-red-500 p-4 text-red-400 font-mono text-center">
									<AlertTriangle className="w-8 h-8 mx-auto mb-2" />
									<div>SIGNAL EXPLOSION DETECTED</div>
									<div className="text-xs opacity-70 mt-1">
										Values exceeded compute limits
									</div>
								</div>
							</div>
						)}
						{sim.state.status === "VANISHED" && (
							<div className="absolute inset-0 bg-slate-900/50 flex items-center justify-center backdrop-blur-sm">
								<div className="bg-slate-950 border border-slate-700 p-4 text-slate-400 font-mono text-center">
									<Activity className="w-8 h-8 mx-auto mb-2 opacity-50" />
									<div>SIGNAL COLLAPSE</div>
									<div className="text-xs opacity-70 mt-1">
										Gradient flow ceased
									</div>
								</div>
							</div>
						)}
					</div>

					<div className="flex justify-between items-center mt-3 text-[10px] font-mono text-slate-500 uppercase">
						<span>Input Layer</span>
						<span>Current Layer ({sim.state.layer})</span>
					</div>
				</SchematicCard>

				{/* Layer Topology / Network Flow */}
				<SchematicCard title="LAYER_TRANSITION_TOPOLOGY">
					<NetworkTopology
						prevSignal={prevSignal}
						currSignal={sim.state.signal}
						matrix={sim.state.matrix}
						mode={dsMode}
					/>
				</SchematicCard>

				<div className="bg-slate-900/50 border border-slate-800 p-4 text-xs text-slate-400 font-mono leading-relaxed">
					<strong className="text-slate-300 block mb-1">
						Why Lanes Matter:
					</strong>
					In Deep Neural Networks, information needs to travel through hundreds
					of layers. In <span className="text-red-400">Wild Mode</span>, matrix
					multiplications amplify (explode) or diminish (vanish) signals
					randomly. The graph uses a <strong>Logarithmic Scale</strong> to show
					this drift. In <span className="text-emerald-400">mHC Mode</span>, the
					Doubly Stochastic constraint ensures every lane maintains its energy
					centered at 1.0.
				</div>
			</div>
		</div>
	);
};

export default DeepSeekMHCSimulation;
