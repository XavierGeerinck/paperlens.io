import React, { useState, useMemo } from "react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { Play, Pause, RotateCcw, Zap } from "lucide-react";

const DeepDeltaSimulation: React.FC = () => {
	const [numLayers] = useState(8);
	const [mode, setMode] = useState<"identity" | "projection" | "reflection">(
		"projection",
	);
	const [showTrajectory, setShowTrajectory] = useState(true);

	// Target we want to reach (oscillating pattern)
	const target = [0.8, -0.6];

	const initialState = {
		currentLayer: 0,
		resnetStates: [[0.2, 0.3]] as number[][],
		ddlStates: [[0.2, 0.3]] as number[][],
		isComplete: false,
	};

	const { state, isRunning, start, stop, reset } = useSimulation({
		initialState,
		tickRate: 400,
		onTick: (prev) => {
			if (prev.currentLayer >= numLayers - 1) {
				return { ...prev, isComplete: true };
			}

			const currentX = prev.ddlStates[prev.currentLayer];
			const currentResnetX = prev.resnetStates[prev.currentLayer];

			// Compute next layer
			const nextLayer = prev.currentLayer + 1;
			const angle = (nextLayer * Math.PI) / 4;

			// Direction vector (rotates each layer)
			const k = [Math.cos(angle), Math.sin(angle)];

			// ResNet: x_{l+1} = x_l + f(x_l) - just additive
			const resnetDelta = [
				(target[0] - currentResnetX[0]) * 0.3,
				(target[1] - currentResnetX[1]) * 0.3,
			];
			const nextResnet = [
				currentResnetX[0] + resnetDelta[0],
				currentResnetX[1] + resnetDelta[1],
			];

			// DDL: Uses Delta Operator with different beta based on mode
			const beta = mode === "identity" ? 0 : mode === "projection" ? 1 : 2;
			const proj = k[0] * currentX[0] + k[1] * currentX[1];
			const v = target; // Target value we want to write
			const nextDDL = [
				currentX[0] + beta * k[0] * (v[0] - proj),
				currentX[1] + beta * k[1] * (v[1] - proj),
			];

			return {
				...prev,
				currentLayer: nextLayer,
				resnetStates: [...prev.resnetStates, nextResnet],
				ddlStates: [...prev.ddlStates, nextDDL],
			};
		},
	});

	const getBeta = () => {
		switch (mode) {
			case "identity":
				return 0;
			case "projection":
				return 1;
			case "reflection":
				return 2;
		}
	};

	const getGeometryDescription = () => {
		switch (mode) {
			case "identity":
				return "β=0: Layer acts as identity, preserving input exactly";
			case "projection":
				return "β=1: Projects onto hyperplane, then writes new value";
			case "reflection":
				return "β=2: Householder reflection across hyperplane";
		}
	};

	// Calculate errors
	const resnetError = useMemo(() => {
		if (state.resnetStates.length === 0) return 0;
		const last = state.resnetStates[state.resnetStates.length - 1];
		return Math.sqrt(
			Math.pow(last[0] - target[0], 2) + Math.pow(last[1] - target[1], 2),
		);
	}, [state.resnetStates, target]);

	const ddlError = useMemo(() => {
		if (state.ddlStates.length === 0) return 0;
		const last = state.ddlStates[state.ddlStates.length - 1];
		return Math.sqrt(
			Math.pow(last[0] - target[0], 2) + Math.pow(last[1] - target[1], 2),
		);
	}, [state.ddlStates, target]);

	return (
		<div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-4 bg-slate-950 text-slate-200 font-sans">
			{/* LEFT: Side-by-Side Trajectory Comparison */}
			<SchematicCard title="TRAJECTORY_COMPARISON">
				<div className="space-y-4">
					<div className="grid grid-cols-2 gap-3">
						{/* ResNet Trajectory */}
						<div className="space-y-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
								Standard ResNet
							</div>
							<div className="relative h-48 bg-black/40 border border-slate-700/50 rounded">
								<svg
									viewBox="-1.2 -1.2 2.4 2.4"
									className="w-full h-full transform scale-y-[-1]"
								>
									{/* Grid */}
									<line
										x1="-1.2"
										y1="0"
										x2="1.2"
										y2="0"
										className="stroke-slate-800/30 stroke-[0.01]"
									/>
									<line
										x1="0"
										y1="-1.2"
										x2="0"
										y2="1.2"
										className="stroke-slate-800/30 stroke-[0.01]"
									/>

									{/* Target */}
									<circle
										cx={target[0]}
										cy={target[1]}
										r="0.08"
										className="fill-amber-500/30 stroke-amber-500 stroke-[0.02]"
									/>

									{/* Trajectory */}
									{showTrajectory &&
										state.resnetStates.map((pos, i) => {
											if (i === 0) return null;
											const prev = state.resnetStates[i - 1];
											return (
												<line
													key={i}
													x1={prev[0]}
													y1={prev[1]}
													x2={pos[0]}
													y2={pos[1]}
													className="stroke-blue-400/50 stroke-[0.02]"
												/>
											);
										})}

									{/* Current position */}
									{state.resnetStates.length > 0 && (
										<circle
											cx={state.resnetStates[state.resnetStates.length - 1][0]}
											cy={state.resnetStates[state.resnetStates.length - 1][1]}
											r="0.06"
											className="fill-blue-400"
										/>
									)}
								</svg>
							</div>
							<div className="text-xs font-mono text-slate-400">
								Error:{" "}
								<span className="text-blue-400">{resnetError.toFixed(3)}</span>
							</div>
						</div>

						{/* DDL Trajectory */}
						<div className="space-y-2">
							<div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
								Deep Delta Learning
							</div>
							<div className="relative h-48 bg-black/40 border border-slate-700/50 rounded">
								<svg
									viewBox="-1.2 -1.2 2.4 2.4"
									className="w-full h-full transform scale-y-[-1]"
								>
									{/* Grid */}
									<line
										x1="-1.2"
										y1="0"
										x2="1.2"
										y2="0"
										className="stroke-slate-800/30 stroke-[0.01]"
									/>
									<line
										x1="0"
										y1="-1.2"
										x2="0"
										y2="1.2"
										className="stroke-slate-800/30 stroke-[0.01]"
									/>

									{/* Target */}
									<circle
										cx={target[0]}
										cy={target[1]}
										r="0.08"
										className="fill-amber-500/30 stroke-amber-500 stroke-[0.02]"
									/>

									{/* Trajectory */}
									{showTrajectory &&
										state.ddlStates.map((pos, i) => {
											if (i === 0) return null;
											const prev = state.ddlStates[i - 1];
											return (
												<line
													key={i}
													x1={prev[0]}
													y1={prev[1]}
													x2={pos[0]}
													y2={pos[1]}
													className="stroke-emerald-400/50 stroke-[0.02]"
												/>
											);
										})}

									{/* Current position */}
									{state.ddlStates.length > 0 && (
										<circle
											cx={state.ddlStates[state.ddlStates.length - 1][0]}
											cy={state.ddlStates[state.ddlStates.length - 1][1]}
											r="0.06"
											className="fill-emerald-400"
										/>
									)}
								</svg>
							</div>
							<div className="text-xs font-mono text-slate-400">
								Error:{" "}
								<span className="text-emerald-400">{ddlError.toFixed(3)}</span>
							</div>
						</div>
					</div>

					{/* Progress Bar */}
					<div className="space-y-2">
						<div className="text-[10px] font-mono text-slate-500 uppercase">
							Layer Depth: {state.currentLayer} / {numLayers}
						</div>
						<div className="h-2 bg-slate-900 rounded-full overflow-hidden">
							<div
								className="h-full bg-gradient-to-r from-emerald-500 to-fuchsia-500 transition-all duration-300"
								style={{
									width: `${(state.currentLayer / numLayers) * 100}%`,
								}}
							/>
						</div>
					</div>

					{/* Legend */}
					<div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
						<div className="flex items-center gap-2">
							<div className="w-3 h-3 bg-blue-400 rounded-sm" />
							<span className="text-slate-400">ResNet (Additive Only)</span>
						</div>
						<div className="flex items-center gap-2">
							<div className="w-3 h-3 bg-emerald-400 rounded-sm" />
							<span className="text-slate-400">DDL (Geometric Transform)</span>
						</div>
					</div>
				</div>
			</SchematicCard>

			{/* RIGHT: Controls and Geometry Mode */}
			<SchematicCard title="DELTA_OPERATOR_CONTROL">
				<div className="space-y-6">
					{/* Geometry Mode Selector */}
					<div className="space-y-3">
						<div className="text-xs font-mono text-slate-500 uppercase tracking-wider">
							Geometric Mode (β parameter)
						</div>
						<div className="grid grid-cols-3 gap-2">
							<button
								type="button"
								onClick={() => setMode("identity")}
								className={`px-3 py-2 text-xs font-mono rounded transition-all ${
									mode === "identity"
										? "bg-blue-500 text-white border-2 border-blue-400"
										: "bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700"
								}`}
							>
								β=0
								<div className="text-[9px] mt-1">IDENTITY</div>
							</button>
							<button
								type="button"
								onClick={() => setMode("projection")}
								className={`px-3 py-2 text-xs font-mono rounded transition-all ${
									mode === "projection"
										? "bg-emerald-500 text-white border-2 border-emerald-400"
										: "bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700"
								}`}
							>
								β=1
								<div className="text-[9px] mt-1">PROJECT</div>
							</button>
							<button
								type="button"
								onClick={() => setMode("reflection")}
								className={`px-3 py-2 text-xs font-mono rounded transition-all ${
									mode === "reflection"
										? "bg-fuchsia-500 text-white border-2 border-fuchsia-400"
										: "bg-slate-800 text-slate-400 border border-slate-700 hover:bg-slate-700"
								}`}
							>
								β=2
								<div className="text-[9px] mt-1">REFLECT</div>
							</button>
						</div>
						<div className="p-3 bg-slate-900/50 border border-slate-700/30 rounded text-xs font-mono text-slate-300">
							<Zap className="inline w-3 h-3 mr-1 text-amber-500" />
							{getGeometryDescription()}
						</div>
					</div>

					{/* Geometric Visualization */}
					<div className="space-y-2">
						<div className="text-xs font-mono text-slate-500 uppercase">
							Delta Operator Geometry
						</div>
						<div className="relative h-48 bg-black/40 border border-slate-700/50 rounded">
							<svg
								viewBox="-1.5 -1.5 3 3"
								className="w-full h-full transform scale-y-[-1]"
							>
								{/* Hyperplane (perpendicular to k) */}
								<line
									x1="-1.5"
									y1="0"
									x2="1.5"
									y2="0"
									className="stroke-slate-700 stroke-[0.02] stroke-dasharray-[0.1,0.05]"
								/>
								<line
									x1="0"
									y1="-1.5"
									x2="0"
									y2="1.5"
									className="stroke-slate-700 stroke-[0.02] stroke-dasharray-[0.1,0.05]"
								/>

								{/* Direction k (rotates each layer) */}
								{state.currentLayer > 0 && (
									<>
										<line
											x1="0"
											y1="0"
											x2={Math.cos((state.currentLayer * Math.PI) / 4)}
											y2={Math.sin((state.currentLayer * Math.PI) / 4)}
											className="stroke-fuchsia-500 stroke-[0.03]"
										/>
										<circle
											cx={Math.cos((state.currentLayer * Math.PI) / 4)}
											cy={Math.sin((state.currentLayer * Math.PI) / 4)}
											r="0.05"
											className="fill-fuchsia-500"
										/>
									</>
								)}

								{/* Show transformation effect */}
								{state.ddlStates.length > 1 && (
									<>
										<line
											x1="0"
											y1="0"
											x2={state.ddlStates[state.ddlStates.length - 2][0]}
											y2={state.ddlStates[state.ddlStates.length - 2][1]}
											className="stroke-slate-400 stroke-[0.02]"
										/>
										<line
											x1="0"
											y1="0"
											x2={state.ddlStates[state.ddlStates.length - 1][0]}
											y2={state.ddlStates[state.ddlStates.length - 1][1]}
											className="stroke-emerald-400 stroke-[0.04]"
										/>
									</>
								)}
							</svg>
						</div>
					</div>

					{/* Controls */}
					<div className="space-y-3">
						<div className="flex gap-2">
							<SchematicButton
								onClick={isRunning ? stop : start}
								disabled={state.isComplete}
								className="flex-1"
							>
								{isRunning ? (
									<>
										<Pause size={16} /> PAUSE
									</>
								) : (
									<>
										<Play size={16} /> RUN LAYERS
									</>
								)}
							</SchematicButton>
							<SchematicButton onClick={reset} className="px-4">
								<RotateCcw size={16} />
							</SchematicButton>
						</div>

						<label className="flex items-center gap-2 text-xs font-mono text-slate-400 cursor-pointer">
							<input
								type="checkbox"
								checked={showTrajectory}
								onChange={(e) => setShowTrajectory(e.target.checked)}
								className="accent-emerald-500"
							/>
							Show trajectory paths
						</label>
					</div>

					{/* Performance Metrics */}
					<div className="border-t border-slate-800 pt-4 space-y-3">
						<div className="text-[10px] font-mono text-slate-500 uppercase">
							Convergence Analysis
						</div>
						<div className="grid grid-cols-2 gap-3">
							<div className="p-2 bg-slate-900/50 border border-slate-700/30 rounded">
								<div className="text-[9px] text-slate-600 uppercase">
									ResNet Error
								</div>
								<div className="text-lg font-mono text-blue-400">
									{resnetError.toFixed(3)}
								</div>
							</div>
							<div className="p-2 bg-slate-900/50 border border-slate-700/30 rounded">
								<div className="text-[9px] text-slate-600 uppercase">
									DDL Error
								</div>
								<div className="text-lg font-mono text-emerald-400">
									{ddlError.toFixed(3)}
								</div>
							</div>
						</div>
						{state.currentLayer > 0 && (
							<div className="text-xs font-mono text-slate-400">
								{ddlError < resnetError ? (
									<span className="text-emerald-400">
										✓ DDL converging{" "}
										{((1 - ddlError / resnetError) * 100).toFixed(1)}% faster
									</span>
								) : (
									<span className="text-blue-400">
										ResNet performing better in {mode} mode
									</span>
								)}
							</div>
						)}
					</div>

					{/* Explainer */}
					<div className="border-t border-slate-800 pt-4">
						<h5 className="text-[10px] font-mono text-slate-500 uppercase mb-2">
							What to watch for:
						</h5>
						<ul className="text-[11px] text-slate-400 space-y-1 font-mono list-disc pl-4">
							<li>
								<b className="text-emerald-400">Projection (β=1)</b>: DDL can
								"erase and write" to specific feature directions
							</li>
							<li>
								<b className="text-fuchsia-400">Reflection (β=2)</b>: DDL can
								flip features across hyperplanes (non-monotonic)
							</li>
							<li>
								<b className="text-blue-400">ResNet</b>: Always moves in a
								"direct path" due to additive bias
							</li>
							<li>
								DDL's geometric flexibility enables richer state transitions
							</li>
						</ul>
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default DeepDeltaSimulation;
