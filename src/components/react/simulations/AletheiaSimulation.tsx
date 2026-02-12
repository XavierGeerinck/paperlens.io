import React, { useMemo } from "react";
import {
	BrainCircuit,
	ShieldCheck,
	RotateCcw,
	Search,
	FileText,
	CheckCircle2,
	AlertTriangle,
	Play,
	Pause,
	RefreshCw,
} from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface AletheiaState {
	phase: "IDLE" | "GENERATING" | "VERIFYING" | "REVISING" | "ACCEPTED" | "FAILED";
	iteration: number;
	maxIterations: number;
	
	// Simulation Parameters (Sliders)
	computeScale: number;      // 0-100
	toolUse: number;           // 0-1
	verifierStrictness: number;// 0-1

	// Document State
	draftQuality: number;          // 0-1 (How good the math is)
	hallucinationRisk: number;     // 0-1 (Citation errors)
	citationCount: number;         // 0-20
	
	// Visual signals
	activeNode: string | null;     // 'generator', 'verifier', 'reviser'
	searchActive: boolean;
	lastLog: string;
}

const AletheiaSimulation: React.FC = () => {
	const { isRunning, state, start, stop, reset, update } = useSimulation<AletheiaState>({
		initialState: {
			phase: "IDLE",
			iteration: 0,
			maxIterations: 8,
			computeScale: 60,
			toolUse: 0.7,
			verifierStrictness: 0.7,
			draftQuality: 0.1,
			hallucinationRisk: 0.0,
			citationCount: 0,
			activeNode: null,
			searchActive: false,
			lastLog: "Ready to initialize research loop.",
		},
		tickRate: 800, // Slower tick rate for phase visualization
		onTick: (prev) => {
			if (prev.phase === "ACCEPTED" || prev.phase === "FAILED" || prev.phase === "IDLE") {
				return prev;
			}

			const next = { ...prev };
			
			// PHASE TRANSITIONS
			switch (prev.phase) {
				case "GENERATING":
					next.phase = "VERIFYING";
					next.activeNode = "verifier";
					next.lastLog = "Verifier checking proofs and citations...";
					
					// Generation logic: quality goes up with compute
					const genBoost = (prev.computeScale / 100) * 0.15 + 0.05;
					next.draftQuality = Math.min(0.95, prev.draftQuality + genBoost);
					
					// Hullucination risk increases with complexity if no tools used later
					next.hallucinationRisk = Math.min(0.9, prev.hallucinationRisk + 0.1);
					break;

				case "VERIFYING":
					// Search tool activation check
					const needsSearch = Math.random() < prev.toolUse;
					next.searchActive = needsSearch;
					
					// Verify Result Calculation
					const riskReduction = needsSearch ? 0.3 : 0.05;
					next.hallucinationRisk = Math.max(0.05, prev.hallucinationRisk - riskReduction);
					
					// Did we pass?
					const qualityThreshold = 0.85; // High bar for research
					const riskThreshold = 0.15;    // Low tolerance for hallucinations
					const strictnessFactor = prev.verifierStrictness; // 0-1

					const qualityPass = prev.draftQuality > (qualityThreshold * strictnessFactor);
					const riskPass = prev.hallucinationRisk < (riskThreshold / strictnessFactor);

					if (qualityPass && riskPass) {
						next.phase = "ACCEPTED";
						next.activeNode = null;
						next.searchActive = false;
						next.lastLog = "Solution Verified & Accepted!";
						stop();
					} else {
						if (prev.iteration >= prev.maxIterations) {
							next.phase = "FAILED";
							next.activeNode = null;
							next.searchActive = false;
							next.lastLog = "Max iterations reached. Solution rejected.";
							stop();
						} else {
							next.phase = "REVISING";
							next.activeNode = "reviser";
							next.lastLog = `Issues found. Sending to Reviser (Iter ${prev.iteration + 1})`;
						}
					}
					break;

				case "REVISING":
					next.phase = "GENERATING";
					next.activeNode = "generator";
					next.iteration += 1;
					next.lastLog = "Regenerating proof segments...";
					next.citationCount = Math.min(20, prev.citationCount + 2);
					break;
			}

			return next;
		},
	});

	// --- Visual Helpers ---
	const getDraftColor = () => {
		if (state.phase === "ACCEPTED") return "#10b981"; // Emerald
		if (state.phase === "FAILED") return "#ef4444";   // Red
		// Interpolate between red (poor) and blue (good)
		return state.draftQuality > 0.7 ? "#60a5fa" : "#f472b6"; 
	};

	const startSimulation = () => {
		update({
			phase: "GENERATING",
			activeNode: "generator",
			draftQuality: 0.2, // Start with rough draft
			hallucinationRisk: 0.5,
			iteration: 1,
			lastLog: "Generator creating initial hypothesis...",
		});
		start();
	};

	const Node = ({ 
		id, 
		icon: Icon, 
		label, 
		color, 
		x, 
		y, 
		active 
	}: { id: string, icon: any, label: string, color: string, x: number, y: number, active: boolean }) => (
		<div 
			className="absolute transform -translate-x-1/2 -translate-y-1/2 flex flex-col items-center gap-2 transition-all duration-500"
			style={{ left: x, top: y, padding: '12px' }}
		>
			<div 
				className={`w-12 h-12 rounded-xl flex items-center justify-center border transition-all duration-300 ${
					active 
						? `bg-${color}-500/20 border-${color}-400 shadow-[0_0_15px_rgba(var(--${color}-rgb),0.5)] scale-110` 
						: "bg-slate-900 border-slate-700 opacity-60"
				}`}
				style={active ? { borderColor: color, boxShadow: `0 0 20px ${color}40`, backgroundColor: `${color}20` } : {}}
			>
				<Icon size={20} style={{ color: active ? color : "#64748b" }} />
			</div>
			<span className={`text-[10px] font-mono uppercase tracking-wider ${active ? "text-white font-bold" : "text-slate-500"}`}>
				{label}
			</span>
		</div>
	);

	// SVG Connector Paths
	const Connector = ({ start, end, active }: { start: [number, number], end: [number, number], active: boolean }) => {
		const midX = (start[0] + end[0]) / 2;
		const midY = (start[1] + end[1]) / 2 - 20; // Curve up slightly
		const path = `M ${start[0]} ${start[1]} Q ${midX} ${midY} ${end[0]} ${end[1]}`;
		
		return (
			<path
				d={path}
				fill="none"
				stroke={active ? "#94a3b8" : "#334155"}
				strokeWidth={active ? 2 : 1}
				strokeDasharray={active ? "4 4" : "none"}
				className={active ? "animate-[dash_1s_linear_infinite]" : ""}
			>
				{active && <animate attributeName="stroke-dashoffset" from="8" to="0" dur="1s" repeatCount="indefinite" />}
			</path>
		);
	};

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-950 text-slate-100 rounded-xl border border-slate-800">
			<SchematicCard title="ALETHEIA_RESEARCH_LOOP">
				<div className="flex flex-col lg:flex-row gap-6 h-[450px]">
					
					{/* LEFT: CONTROLS */}
					<div className="w-full lg:w-1/3 flex flex-col gap-6 p-4 bg-slate-900/50 rounded-lg border border-slate-800">
						<div className="space-y-4">
							<div className="flex items-center gap-2 text-fuchsia-400 font-mono text-xs uppercase border-b border-slate-800 pb-2">
								<BrainCircuit size={14} /> Agent Configuration
							</div>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-400 uppercase mb-1">
									<span>Compute Scale</span>
									<span className="text-fuchsia-400">{state.computeScale}</span>
								</div>
								<input
									type="range"
									min={10} max={100}
									value={state.computeScale}
									disabled={isRunning}
									onChange={(e) => update({ computeScale: Number(e.target.value) })}
									className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-400 uppercase mb-1">
									<span>Tool Verification</span>
									<span className="text-emerald-400">{(state.toolUse * 100).toFixed(0)}%</span>
								</div>
								<input
									type="range"
									min={0} max={1} step={0.1}
									value={state.toolUse}
									disabled={isRunning}
									onChange={(e) => update({ toolUse: Number(e.target.value) })}
									className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-400 uppercase mb-1">
									<span>Verifier Strictness</span>
									<span className="text-blue-400">{(state.verifierStrictness * 100).toFixed(0)}%</span>
								</div>
								<input
									type="range"
									min={0.1} max={1} step={0.1}
									value={state.verifierStrictness}
									disabled={isRunning}
									onChange={(e) => update({ verifierStrictness: Number(e.target.value) })}
									className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
								/>
							</label>
						</div>

						{/* LOOP METRICS */}
						<div className="mt-auto space-y-4">
							<div className="flex items-center gap-2 text-emerald-400 font-mono text-xs uppercase border-b border-slate-800 pb-2">
								<CheckCircle2 size={14} /> Document Status
							</div>
							
							<div className="grid grid-cols-2 gap-4">
								<div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center">
									<div className="text-[10px] text-slate-500 font-mono mb-1">QUALITY</div>
									<div className="text-xl font-bold font-mono" style={{ color: getDraftColor() }}>
										{(state.draftQuality * 100).toFixed(0)}
									</div>
								</div>
								<div className="bg-slate-950 p-3 rounded border border-slate-800 flex flex-col items-center">
									<div className="text-[10px] text-slate-500 font-mono mb-1">HALLUCINATION</div>
									<div className={`text-xl font-bold font-mono ${state.hallucinationRisk > 0.3 ? "text-orange-400" : "text-emerald-400"}`}>
										{(state.hallucinationRisk * 100).toFixed(0)}%
									</div>
								</div>
							</div>
							
							<div className="flex justify-between items-center text-[10px] font-mono text-slate-500 bg-slate-950 p-2 rounded">
								<span>ITERATION</span>
								<span className="text-slate-300">{state.iteration} / {state.maxIterations}</span>
							</div>
						</div>
					</div>

					{/* RIGHT: ANIMATED CANVAS */}
					<div className="w-full lg:w-2/3 relative bg-black/40 rounded-lg border border-slate-800 overflow-hidden">
						{/* Grid Background */}
						<div className="absolute inset-0 bg-[linear-gradient(rgba(30,41,59,0.3)_1px,transparent_1px),linear-gradient(90deg,rgba(30,41,59,0.3)_1px,transparent_1px)] bg-[size:20px_20px]"></div>

						{/* Connectors SVG Layer */}
						<svg className="absolute inset-0 w-full h-full pointer-events-none">
							<Connector start={[150, 100]} end={[400, 100]} active={state.phase === "VERIFYING"} />
							<Connector start={[400, 100]} end={[275, 320]} active={state.phase === "REVISING"} />
							<Connector start={[275, 320]} end={[150, 100]} active={state.phase === "GENERATING"} />
							
							{/* Tool Connection */}
							{state.searchActive && (
								<line x1="400" y1="140" x2="450" y2="200" stroke="#f59e0b" strokeWidth="2" strokeDasharray="4 4" className="animate-pulse" />
							)}
						</svg>

						{/* Nodes */}
						<Node 
							id="gen" 
							x={150} y={100} 
							icon={BrainCircuit} 
							label="Generator" 
							color="#d946ef" 
							active={state.activeNode === "generator"} 
						/>
						<Node 
							id="ver" 
							x={400} y={100} 
							icon={ShieldCheck} 
							label="Verifier" 
							color="#3b82f6" 
							active={state.activeNode === "verifier"} 
						/>
						<Node 
							id="rev" 
							x={275} y={320} 
							icon={RotateCcw} 
							label="Reviser" 
							color="#f97316" 
							active={state.activeNode === "reviser"} 
						/>

						{/* Document / Draft Visualization in Center */}
						<div 
							className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-slate-900 border border-slate-700 p-4 rounded shadow-2xl transition-all duration-300"
							style={{ 
								width: 140, 
								height: 180,
								borderColor: getDraftColor(),
								boxShadow: `0 0 20px ${getDraftColor()}20`
							}}
						>
							<div className="flex items-center gap-2 mb-2 border-b border-slate-800 pb-1">
								<FileText size={12} className="text-slate-400" />
								<span className="text-[8px] font-mono text-slate-500 uppercase">DRAFT_V{state.iteration}.TEX</span>
							</div>
							{/* Fake text lines */}
							<div className="space-y-1.5">
								{[...Array(6)].map((_, i) => (
									<div 
										key={i} 
										className="h-1 rounded-full transition-all duration-500"
										style={{ 
											width: `${30 + Math.random() * 70}%`,
											backgroundColor: state.draftQuality > (i * 0.15) ? getDraftColor() : "#334155",
											opacity: state.hallucinationRisk > 0.4 && i > 3 ? 0.3 : 1
										}}
									/>
								))}
							</div>
							
							{state.activeNode === "verifier" && (
								<div className="absolute -right-12 top-0 bg-blue-900/80 text-blue-200 text-[10px] px-2 py-1 rounded border border-blue-500/30 animate-pulse">
									Verifying...
								</div>
							)}
						</div>

						{/* Web Tool Icon */}
						<div 
							className={`absolute top-[200px] right-[50px] flex flex-col items-center gap-2 transition-opacity duration-300 ${state.searchActive ? "opacity-100" : "opacity-30"}`}
						>
							<div className="w-10 h-10 rounded-full bg-amber-900/30 border border-amber-500/50 flex items-center justify-center">
								<Search size={16} className="text-amber-400" />
							</div>
							<span className="text-[10px] font-mono text-amber-500 uppercase">Google Search</span>
						</div>

						{/* Result Overlay */}
						{(state.phase === "ACCEPTED" || state.phase === "FAILED") && (
							<div className="absolute inset-0 bg-slate-950/80 flex items-center justify-center z-10 backdrop-blur-sm">
								<div className={`p-6 rounded-xl border flex flex-col items-center gap-3 ${state.phase === "ACCEPTED" ? "bg-emerald-900/20 border-emerald-500" : "bg-red-900/20 border-red-500"}`}>
									{state.phase === "ACCEPTED" ? <CheckCircle2 size={40} className="text-emerald-400" /> : <AlertTriangle size={40} className="text-red-400" />}
									<h3 className={`font-mono text-lg font-bold ${state.phase === "ACCEPTED" ? "text-emerald-400" : "text-red-400"}`}>
										{state.phase === "ACCEPTED" ? "PUBLICATION READY" : "REJECTED"}
									</h3>
									<p className="text-xs text-slate-300 max-w-[200px] text-center">{state.lastLog}</p>
									<button 
										onClick={reset}
										className="mt-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded text-xs text-slate-200 font-mono transition-colors"
									>
										START_NEW_DRAFT
									</button>
								</div>
							</div>
						)}

					</div>
				</div>

				{/* Bottom Bar: Logs & Controls */}
				<div className="mt-6 pt-4 border-t border-slate-800 flex justify-between items-center">
					<div className="font-mono text-xs text-slate-400 flex items-center gap-2">
						<span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
						{state.lastLog}
					</div>
					
					<div className="flex gap-2">
						{!isRunning && state.phase !== "ACCEPTED" && state.phase !== "FAILED" ? (
							<SchematicButton onClick={startSimulation} variant="primary">
								<div className="flex items-center gap-2">
									<Play size={14} /> INITIALIZE_AGENT
								</div>
							</SchematicButton>
						) : (state.phase === "ACCEPTED" || state.phase === "FAILED") ? (
							<SchematicButton onClick={reset} variant="secondary">
								<div className="flex items-center gap-2">
									<RefreshCw size={14} /> RESET
								</div>
							</SchematicButton>
						) : (
							<SchematicButton onClick={stop} variant="secondary">
								<div className="flex items-center gap-2">
									<Pause size={14} /> PAUSE
								</div>
							</SchematicButton>
						)}
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default AletheiaSimulation;
