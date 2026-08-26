import React, { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard } from "../SketchElements";
import {
	Play,
	Pause,
	RefreshCw,
	Zap,
	BrainCircuit,
	Target,
	XCircle,
	Layers,
	Move,
	Rewind,
} from "lucide-react";

const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 400;
const GRID_RES = 20;
const AGENT_RADIUS = 12;
const GOAL_RADIUS = 20;

interface Point {
	x: number;
	y: number;
}
interface Obstacle extends Point {
	id: string;
	radius: number;
}
interface Trajectory {
	id: number;
	points: Point[];
	cost: number;
	selected?: boolean;
}

const distance = (p1: Point, p2: Point) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

const generatePolicyField = (goal: Point, obstacles: Obstacle[]) => {
	const cols = Math.ceil(CANVAS_WIDTH / GRID_RES);
	const rows = Math.ceil(CANVAS_HEIGHT / GRID_RES);
	const field: Point[][] = [];

	for (let r = 0; r < rows; r++) {
		const row: Point[] = [];
		for (let c = 0; c < cols; c++) {
			const x = c * GRID_RES + GRID_RES / 2;
			const y = r * GRID_RES + GRID_RES / 2;

			// Vector to goal
			let dx = goal.x - x;
			let dy = goal.y - y;
			let dist = Math.hypot(dx, dy);

			if (dist > 0) {
				dx /= dist;
				dy /= dist;
			}

			// Obstacle repulsion (Simple Potential Field)
			for (const obs of obstacles) {
				const odx = x - obs.x;
				const ody = y - obs.y;
				const odist = Math.hypot(odx, ody);
				const repulsionRadius = obs.radius + 60;

				if (odist < repulsionRadius) {
					const force = (repulsionRadius - odist) / repulsionRadius;
					dx += (odx / odist) * force * 5; // Strong repulsion
					dy += (ody / odist) * force * 5;
				}
			}

			// Normalize
			const mag = Math.hypot(dx, dy);
			if (mag > 0) {
				row.push({ x: dx / mag, y: dy / mag });
			} else {
				row.push({ x: 0, y: 0 });
			}
		}
		field.push(row);
	}
	return field;
};

const simulateMPCTrajectory = (
	id: number,
	start: Point,
	goal: Point,
	obstacles: Obstacle[],
	horizon: number,
	angleOffset: number,
	stepSize: number,
): Trajectory => {
	const points: Point[] = [start];
	let curr = { ...start };
	let totalCost = 0;

	// Base direction to goal
	const dx = goal.x - start.x;
	const dy = goal.y - start.y;
	const baseAngle = Math.atan2(dy, dx);
	const planAngle = baseAngle + angleOffset;

	for (let t = 0; t < horizon; t++) {
		curr = {
			x: curr.x + Math.cos(planAngle) * stepSize,
			y: curr.y + Math.sin(planAngle) * stepSize,
		};
		points.push(curr);

		// Cost: Distance to Goal
		totalCost += distance(curr, goal);

		// Cost: Obstacle collision
		for (const obs of obstacles) {
			const d = distance(curr, obs);
			if (d < obs.radius + AGENT_RADIUS) {
				totalCost += 10000;
			} else if (d < obs.radius + AGENT_RADIUS + 30) {
				totalCost += 5000 / d; // Soft buffer
			}
		}
	}

	return { id, points, cost: totalCost };
};

// Inner Component that holds the simulation hook
const ControlTheoreticSimulationInner: React.FC<{
	mode: "REFLEXIVE" | "MPC";
	initialObstacles: Obstacle[];
	onDragStart: (id: string, e: React.MouseEvent) => void;
	onDragMove: (e: React.MouseEvent) => void;
	onDragEnd: () => void;
	dragId: string | null;
	needsRetrain: boolean;
	onTrain: () => void;
	policyField: { x: number; y: number }[][];
}> = ({
	mode,
	initialObstacles,
	onDragStart,
	onDragMove,
	onDragEnd,
	dragId,
	needsRetrain,
	onTrain,
	policyField,
}) => {
	const policyFieldRef = useRef(policyField);
	useEffect(() => {
		policyFieldRef.current = policyField;
	}, [policyField]);

	const { isRunning, start, stop, reset, state } = useSimulation({
		initialState: {
			agent: { x: 50, y: CANVAS_HEIGHT / 2 },
			goal: { x: CANVAS_WIDTH - 50, y: CANVAS_HEIGHT / 2 },
			obstacles: initialObstacles,
			trajectories: [] as Trajectory[],
			pathHistory: [] as Point[],
			cost: 0,
			reward: 0,
			cumulativeReward: 0,
			steps: 0,
			mode: mode,
			status: "IDLE" as "IDLE" | "PLANNING" | "MOVING" | "CRASHED" | "SUCCESS",
		},
		tickRate: 80,
		onTick: (prev) => {
			if (prev.status === "CRASHED" || prev.status === "SUCCESS") {
				stop();
				return prev;
			}

			// 1. Goal Check
			if (distance(prev.agent, prev.goal) < GOAL_RADIUS) {
				return {
					...prev,
					status: "SUCCESS" as const,
					reward: 100,
					cumulativeReward: prev.cumulativeReward + 100,
				};
			}

			// 2. Collision Check (REALITY)
			for (const obs of prev.obstacles) {
				if (distance(prev.agent, obs) < obs.radius + AGENT_RADIUS) {
					return {
						...prev,
						status: "CRASHED" as const,
						reward: -100,
						cumulativeReward: prev.cumulativeReward - 100,
					};
				}
			}

			let nextAgent = { ...prev.agent };
			let trajectories: Trajectory[] = [];
			const stepSize = 12;
			let currentReward = 0;

			if (prev.mode === "REFLEXIVE") {
				// --- RL MODE ---
				// Lookup Policy from Ref
				const c = Math.min(
					Math.max(Math.floor(prev.agent.x / GRID_RES), 0),
					Math.ceil(CANVAS_WIDTH / GRID_RES) - 1,
				);
				const r = Math.min(
					Math.max(Math.floor(prev.agent.y / GRID_RES), 0),
					Math.ceil(CANVAS_HEIGHT / GRID_RES) - 1,
				);

				const vector = policyFieldRef.current[r]?.[c] || { x: 1, y: 0 };

				nextAgent.x += vector.x * stepSize;
				nextAgent.y += vector.y * stepSize;

				trajectories = [];
				currentReward =
					distance(prev.agent, prev.goal) - distance(nextAgent, prev.goal);
			} else {
				// --- MPC MODE ---
				const numCandidates = 25;
				const horizon = 12;
				let minCost = Infinity;

				for (let i = 0; i < numCandidates; i++) {
					const angleOffset = (Math.random() - 0.5) * Math.PI * 1.5;
					const traj = simulateMPCTrajectory(
						i,
						prev.agent,
						prev.goal,
						prev.obstacles,
						horizon,
						angleOffset,
						stepSize,
					);

					if (traj.cost < minCost) {
						minCost = traj.cost;
					}
					trajectories.push(traj);
				}

				trajectories.sort((a, b) => a.cost - b.cost);
				if (trajectories.length > 0) {
					trajectories[0].selected = true;
					if (trajectories[0].points.length > 1) {
						nextAgent = trajectories[0].points[1];
					}
				}
			}

			return {
				...prev,
				agent: nextAgent,
				pathHistory: [...prev.pathHistory, prev.agent],
				trajectories,
				steps: prev.steps + 1,
				mode: prev.mode, // Keep mode form state
				reward: currentReward,
				cumulativeReward: prev.cumulativeReward + currentReward,
			};
		},
	});

	return (
		<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
			{/* LEFT: THE INTERNAL MODEL (BRAIN) */}
			<SchematicCard
				title="INTERNAL STATE (SYSTEM VIEWER)"
				className="lg:col-span-1 min-h-[300px] relative overflow-hidden"
			>
				{mode === "REFLEXIVE" ? (
					<div className="h-full flex flex-col">
						<div className="flex justify-between items-center mb-4">
							<span className="text-xs font-mono text-zinc-500">
								POLICY VECTOR FIELD
							</span>
							{needsRetrain && (
								<span className="text-xs text-red-500 font-bold animate-pulse">
									MISMATCH DETECTED
								</span>
							)}
						</div>

						{/* Visual Rep of Policy Grid */}
						<div className="flex-1 relative bg-zinc-900 rounded border border-zinc-800 overflow-hidden mb-4">
							{/* Mini Grid Visualization */}
							<div
								className="absolute inset-0 grid"
								style={{
									gridTemplateColumns: `repeat(${policyField[0]?.length || 1}, 1fr)`,
								}}
							>
								{policyField.flat().map((v, i) => (
									<div
										key={i}
										className="flex items-center justify-center opacity-30"
									>
										<div
											style={{
												transform: `rotate(${Math.atan2(v.y, v.x)}rad)`,
											}}
											className="w-1 h-0.5 bg-yellow-500"
										/>
									</div>
								))}
							</div>
						</div>

						{/* RL Rewards Display */}
						<div className="bg-zinc-900 rounded border border-zinc-800 p-3 font-mono text-xs">
							<div className="flex justify-between mb-1">
								<span className="text-zinc-500">STEP REWARD:</span>
								<span
									className={
										state?.reward > 0 ? "text-green-500" : "text-red-500"
									}
								>
									{state?.reward.toFixed(2)}
								</span>
							</div>
							<div className="flex justify-between">
								<span className="text-zinc-500">CUMULATIVE:</span>
								<span
									className={
										state?.cumulativeReward > 0
											? "text-green-500"
											: "text-zinc-300"
									}
								>
									{state?.cumulativeReward.toFixed(1)}
								</span>
							</div>
						</div>

						<button
							type="button"
							onClick={onTrain}
							className={`mt-4 w-full py-2 font-mono text-xs uppercase border rounded flex items-center justify-center gap-2 ${needsRetrain ? "border-red-500 text-red-500 hover:bg-red-900/20" : "border-zinc-700 text-zinc-500 cursor-not-allowed"}`}
							disabled={!needsRetrain}
						>
							<RefreshCw
								size={14}
								className={!needsRetrain ? "" : "animate-spin"}
							/>
							{needsRetrain ? "Retrain Policy (Offline)" : "Policy Optimized"}
						</button>
					</div>
				) : (
					<div className="h-full flex flex-col">
						<div className="flex justify-between items-center mb-4">
							<span className="text-xs font-mono text-zinc-500">
								ONLINE TRAJECTORY OPTIMIZATION
							</span>
							<span className="text-xs text-cyan-500 font-bold">ACTIVE</span>
						</div>
						<div className="flex-1 bg-zinc-900 rounded border border-zinc-800 p-4 font-mono text-xs text-green-400 overflow-y-auto">
							<div className="flex justify-between mb-2 pb-2 border-b border-zinc-800 text-zinc-500">
								<span>RANK</span>
								<span>ID</span>
								<span>COST</span>
							</div>
							{state?.trajectories?.slice(0, 8).map((t, i) => (
								<div
									key={i}
									className={`flex justify-between mb-1 ${t.selected ? "text-cyan-400 font-bold" : "opacity-60"}`}
								>
									<span>#{i + 1}</span>
									<span>T{t.id}</span>
									<span>{t.cost.toFixed(0)}</span>
								</div>
							))}
							<div className="mt-2 text-zinc-500 italic">
								...Evaluating {state?.trajectories?.length} futures...
							</div>
						</div>
					</div>
				)}
			</SchematicCard>

			{/* RIGHT: THE WORLD (ENVIRONMENT) */}
			<div className="lg:col-span-2">
				<SchematicCard
					title="REALITY (SIMULATION)"
					className="relative h-[400px]"
				>
					<div
						role="button"
						tabIndex={0}
						className={`absolute inset-0 bg-zinc-950 overflow-hidden ${dragId ? "cursor-grabbing" : ""}`}
						onMouseMove={onDragMove}
						onMouseUp={onDragEnd}
						onMouseLeave={onDragEnd}
						aria-label="Simulation Area"
					>
						{/* Grid Lines */}
						<div
							className="absolute inset-0"
							style={{
								backgroundImage:
									"linear-gradient(#222 1px, transparent 1px), linear-gradient(90deg, #222 1px, transparent 1px)",
								backgroundSize: `${GRID_RES}px ${GRID_RES}px`,
								opacity: 0.2,
							}}
						/>

						{/* Policy Overlay (Only for RL Mode to show the 'Mental Map') */}
						{mode === "REFLEXIVE" && (
							<div className="absolute inset-0 pointer-events-none opacity-20">
								{policyField.map((row, r) =>
									row.map((vec, c) => (
										<div
											key={`${r}-${c}`}
											className="absolute w-1 h-3 bg-yellow-500"
											style={{
												left: c * GRID_RES + GRID_RES / 2,
												top: r * GRID_RES + GRID_RES / 2,
												transform: `rotate(${Math.atan2(vec.y, vec.x) + Math.PI / 2}rad)`,
												transformOrigin: "center",
											}}
										/>
									)),
								)}
							</div>
						)}

						{/* Trajectories (MPC Thoughts) */}
						{mode === "MPC" &&
							state?.trajectories?.map((traj, i) => (
								<svg
									key={i}
									className="absolute inset-0 pointer-events-none overflow-visible"
								>
									<title>Trajectory {i}</title>
									<polyline
										points={traj.points.map((p) => `${p.x},${p.y}`).join(" ")}
										fill="none"
										stroke={traj.selected ? "#5cc8ff" : "#5cc8ff"}
										strokeWidth={traj.selected ? 2 : 1}
										strokeOpacity={traj.selected ? 1 : 0.1}
									/>
								</svg>
							))}

						{/* Obstacles (Draggable) */}
						{state?.obstacles.map((obs) => (
							<button
								key={obs.id}
								type="button"
								onMouseDown={(e) => onDragStart(obs.id, e)}
								className={`absolute rounded-full flex items-center justify-center transition-transform hover:scale-105 active:scale-95 ${dragId === obs.id ? "cursor-grabbing border-white" : "cursor-grab border-red-500/50"}`}
								style={{
									left: obs.x - obs.radius,
									top: obs.y - obs.radius,
									width: obs.radius * 2,
									height: obs.radius * 2,
									background:
										"radial-gradient(circle, rgba(127,29,29,0.4) 0%, rgba(69,10,10,0.8) 100%)",
									borderWidth: 2,
								}}
								aria-label="Obstacle"
							>
								<div className="text-red-500/50 text-xs font-mono select-none">
									OBS
								</div>
							</button>
						))}

						{/* Goal */}
						<div
							className="absolute rounded-full border border-green-500 flex items-center justify-center animate-pulse"
							style={{
								left: CANVAS_WIDTH - 50 - GOAL_RADIUS,
								top: CANVAS_HEIGHT / 2 - GOAL_RADIUS,
								width: GOAL_RADIUS * 2,
								height: GOAL_RADIUS * 2,
								background: "rgba(34, 197, 94, 0.2)",
							}}
						>
							<Target size={16} className="text-green-500" />
						</div>

						{/* Agent */}
						{state?.agent && (
							<div
								className={`absolute rounded-full z-10 shadow-[0_0_15px_currentColor] transition-all duration-75 ${mode === "REFLEXIVE" ? "bg-yellow-500 text-yellow-500" : "bg-cyan-500 text-cyan-500"}`}
								style={{
									left: state.agent.x - AGENT_RADIUS,
									top: state.agent.y - AGENT_RADIUS,
									width: AGENT_RADIUS * 2,
									height: AGENT_RADIUS * 2,
								}}
							/>
						)}

						{/* Crash Marker */}
						{state?.status === "CRASHED" && (
							<div
								className="absolute text-red-500"
								style={{ left: state.agent.x - 20, top: state.agent.y - 20 }}
							>
								<XCircle size={40} />
							</div>
						)}
					</div>

					{/* Floating Controls */}
					<div className="absolute bottom-4 left-4 flex gap-2">
						<button
							type="button"
							onClick={() => {
								if (isRunning) stop();
								else if (
									state?.status === "CRASHED" ||
									state?.status === "SUCCESS"
								)
									reset();
								else start();
							}}
							className="p-3 rounded-full bg-zinc-800 text-white hover:bg-zinc-700 shadow-lg border border-zinc-600"
						>
							{isRunning ? (
								<Pause />
							) : state?.status === "CRASHED" || state?.status === "SUCCESS" ? (
								<Rewind />
							) : (
								<Play />
							)}
						</button>
						{isRunning && (
							<div className="px-3 py-1 bg-black/50 rounded backdrop-blur text-xs font-mono text-zinc-300 flex items-center">
								STEPS: {state?.steps}
							</div>
						)}
					</div>

					<div className="absolute top-4 right-4 bg-black/50 backdrop-blur px-3 py-2 rounded border border-zinc-800 text-xs font-mono text-zinc-400 pointer-events-none">
						{dragId
							? "DRAGGING OBSTACLE..."
							: "DRAG OBSTACLES TO TEST ROBUSTNESS"}
					</div>
				</SchematicCard>
			</div>
		</div>
	);
};

const ControlTheoreticSimulation: React.FC = () => {
	const [activeMode, setActiveMode] = useState<"REFLEXIVE" | "MPC">("MPC");
	const [obstacles, setObstacles] = useState<Obstacle[]>([
		{ id: "obs1", x: CANVAS_WIDTH / 2, y: CANVAS_HEIGHT / 2, radius: 45 },
		{ id: "obs2", x: CANVAS_WIDTH * 0.7, y: CANVAS_HEIGHT * 0.3, radius: 30 },
	]);
	const [dragId, setDragId] = useState<string | null>(null);

	// RL State Management
	const [policyField, setPolicyField] = useState<{ x: number; y: number }[][]>(
		[],
	);
	const [needsRetrain, setNeedsRetrain] = useState(false);
	const [simKey, setSimKey] = useState(0); // To force reset

	const trainPolicy = useCallback(() => {
		const field = generatePolicyField(
			{ x: CANVAS_WIDTH - 50, y: CANVAS_HEIGHT / 2 },
			obstacles,
		);
		setPolicyField(field);
		setNeedsRetrain(false);
	}, [obstacles]);

	// Initial Train
	useEffect(() => {
		// Run initial training manually to avoid dependency loop or complexity
		const field = generatePolicyField(
			{ x: CANVAS_WIDTH - 50, y: CANVAS_HEIGHT / 2 },
			obstacles,
		);
		setPolicyField(field);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const handleDragStart = (id: string, e: React.MouseEvent) => {
		setDragId(id);
		e.stopPropagation();
	};

	const handleDragMove = (e: React.MouseEvent) => {
		if (!dragId) return;
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		const x = Math.max(0, Math.min(CANVAS_WIDTH, e.clientX - rect.left));
		const y = Math.max(0, Math.min(CANVAS_HEIGHT, e.clientY - rect.top));

		setObstacles((prev) =>
			prev.map((o) => (o.id === dragId ? { ...o, x, y } : o)),
		);

		// If we move obstacles, the static RL policy is now mismatched
		if (!needsRetrain) setNeedsRetrain(true);
	};

	const handleDragEnd = () => {
		setDragId(null);
		// Restart simulation with new obstacles
		setSimKey((k) => k + 1);
	};

	const handleTrain = () => {
		trainPolicy();
		setSimKey((k) => k + 1); // Reset so agent starts fresh with new policy
	};

	const handleModeChange = (mode: "REFLEXIVE" | "MPC") => {
		setActiveMode(mode);
		setSimKey((k) => k + 1);
	};

	return (
		<div className="flex flex-col gap-6">
			{/* 1. TOP CONTROL PANEL */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
				<SchematicCard title="ARCHITECTURE SELECTOR" className="h-full">
					<div className="flex gap-2 h-full items-center">
						<button
							type="button"
							onClick={() => handleModeChange("REFLEXIVE")}
							className={`flex-1 h-20 rounded-lg border-2 flex flex-col items-center justify-center gap-2 transition-all ${activeMode === "REFLEXIVE" ? "border-yellow-500 bg-yellow-900/20 text-yellow-100" : "border-zinc-800 bg-zinc-900 text-zinc-500 hover:border-zinc-600"}`}
						>
							<Zap size={24} />
							<span className="font-mono font-bold">RL (REFLEXIVE)</span>
						</button>
						<button
							type="button"
							onClick={() => handleModeChange("MPC")}
							className={`flex-1 h-20 rounded-lg border-2 flex flex-col items-center justify-center gap-2 transition-all ${activeMode === "MPC" ? "border-cyan-500 bg-cyan-900/20 text-cyan-100" : "border-zinc-800 bg-zinc-900 text-zinc-500 hover:border-zinc-600"}`}
						>
							<BrainCircuit size={24} />
							<span className="font-mono font-bold">MPC (PLANNING)</span>
						</button>
					</div>
				</SchematicCard>

				<SchematicCard title="LEARNING CONTEXT" className="h-full">
					<div className="text-sm space-y-2 text-zinc-400">
						{activeMode === "REFLEXIVE" ? (
							<div className="space-y-3">
								<p className="text-yellow-400 font-bold flex items-center gap-2">
									<Layers size={14} /> AMORTIZED POLICY (System 1)
								</p>
								<div>
									<p className="text-zinc-300 font-bold text-xs mb-1">
										ARCHITECTURE
									</p>
									<div className="text-zinc-400">
										<ReactMarkdown
											remarkPlugins={[remarkMath]}
											rehypePlugins={[rehypeKatex]}
										>
											{
												'State $(x,y) \\to$ Policy Lookup $\\to$ Action. The "brain" is a cached vector field trained offline.'
											}
										</ReactMarkdown>
									</div>
								</div>
								<div>
									<p className="text-zinc-300 font-bold text-xs mb-1">
										REWARD SIGNAL
									</p>
									<div className="text-zinc-400">
										<ReactMarkdown
											remarkPlugins={[remarkMath]}
											rehypePlugins={[rehypeKatex]}
										>
											{
												"The policy optimizes: $R = (Progress) - (Step Cost) - (Collision)$. It learns to take shortcuts while avoiding static obstacles."
											}
										</ReactMarkdown>
									</div>
								</div>
								<p className="text-xs italic border-l-2 border-yellow-500/30 pl-2">
									Limitation: If obstacles move, the cached plan is wrong until
									you "Retrain".
								</p>
							</div>
						) : (
							<div className="space-y-3">
								<p className="text-cyan-400 font-bold flex items-center gap-2">
									<Move size={14} /> MODEL PREDICTIVE CONTROL (System 2)
								</p>
								<div>
									<p className="text-zinc-300 font-bold text-xs mb-1">
										ARCHITECTURE
									</p>
									<div className="text-zinc-400">
										<ReactMarkdown
											remarkPlugins={[remarkMath]}
											rehypePlugins={[rehypeKatex]}
										>
											{
												"State $\\to$ Forward Model $\\to$ Trajectory Search. We simulate 25 possible futures every single tick."
											}
										</ReactMarkdown>
									</div>
								</div>
								<div>
									<p className="text-zinc-300 font-bold text-xs mb-1">
										COST FUNCTION
									</p>
									<div className="text-zinc-400">
										<ReactMarkdown
											remarkPlugins={[remarkMath]}
											rehypePlugins={[rehypeKatex]}
										>
											{
												"We minimize: $J = (Dist to Goal) + (Collision Risk)$. It actively fears dynamic threats."
											}
										</ReactMarkdown>
									</div>
								</div>
								<p className="text-xs italic border-l-2 border-cyan-500/30 pl-2">
									Advantage: Handles new obstacles instantly without retraining.
								</p>
							</div>
						)}
					</div>
				</SchematicCard>
			</div>

			<ControlTheoreticSimulationInner
				key={simKey}
				mode={activeMode}
				initialObstacles={obstacles}
				onDragStart={handleDragStart}
				onDragMove={handleDragMove}
				onDragEnd={handleDragEnd}
				dragId={dragId}
				needsRetrain={needsRetrain}
				onTrain={handleTrain}
				policyField={policyField}
			/>
		</div>
	);
};

export default ControlTheoreticSimulation;
