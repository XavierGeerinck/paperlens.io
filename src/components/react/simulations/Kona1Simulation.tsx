import type { FC } from "react";
import { useEffect, useState } from "react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import { Play, RotateCcw, Flame, Activity, BrainCircuit, ShieldCheck } from "lucide-react";

// 9x9 Full Sudoku
const BOARD_SIZE = 9;
const SQ_SIZE = 3;

// Problem: 0 = Unknown (Classic "Evil" Sudoku)
const PROBLEM = [
	0, 0, 0, 0, 0, 0, 0, 1, 2,
	0, 0, 0, 0, 3, 5, 0, 0, 0,
	0, 0, 0, 6, 0, 0, 0, 7, 0,
	7, 0, 0, 0, 0, 0, 3, 0, 0,
	0, 0, 0, 4, 0, 0, 8, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 1, 2, 0, 0, 0, 0,
	0, 8, 0, 0, 0, 0, 0, 4, 0,
	0, 5, 0, 0, 0, 0, 6, 0, 0,
];

const CELL_IDS = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, i) => {
	const row = Math.floor(i / BOARD_SIZE);
	const col = i % BOARD_SIZE;
	return `r${row}c${col}`;
});

type KonaState = {
	// The "Latent Thought" is a 81x9 tensor (logits for each number 1-9 at each cell)
	latent: number[][];
	board: number[]; // The discrete projection (argmax)
	energy: number;
	step: number;
	gradientTrace: { id: number; value: number }[];
};

// Helper: Softmax
const softmax = (logits: number[]) => {
	const max = Math.max(...logits);
	const exps = logits.map((l) => Math.exp(l - max));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
};

// Helper: Argmax
const argmax = (logits: number[]) => {
	let maxIdx = 0;
	for (let i = 1; i < logits.length; i++) {
		if (logits[i] > logits[maxIdx]) maxIdx = i;
	}
	return maxIdx + 1; // 1-based value (1-9)
};

const Kona1Simulation: FC = () => {
	// --- DIFFERENTIABLE ENERGY FUNCTION ---
	// Calculates energy based on PROBABILITIES, not discrete values.
	// This allows gradients to flow even when the answer is "wrong".
	const calculateEnergyAndGradients = (latent: number[][]) => {
		// 1. Forward Pass: Compute Probabilities (Soft Board)
		const probs = latent.map((logits) => softmax(logits));

		let energy = 0;
		const grads = latent.map((row) => Array(BOARD_SIZE).fill(0));

		// Helper to add gradient
		const addGrad = (cellIdx: number, valIdx: number, amount: number) => {
			grads[cellIdx][valIdx] += amount;
		};

		// CONSTRAINT 1: Fixed Cells (Anchor to ground truth)
		for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
			if (PROBLEM[i] !== 0) {
				const targetValIdx = PROBLEM[i] - 1;
				// Energy = -log(prob_correct) -> Cross Entropy
				const p = probs[i][targetValIdx];
				energy -= Math.log(p + 1e-6) * 20; // Strong anchor

				// Gradient: dE/dlogit = -20 * (delta(m, target) - p_target)
				// This pushes the target logit UP and other logits DOWN
				for (let m = 0; m < BOARD_SIZE; m++) {
					if (m === targetValIdx) {
						// Increase this logit (gradient descent will subtract, so negative grad)
						addGrad(i, m, -20.0 * (1 - p));
					} else {
						// Decrease other logits
						addGrad(i, m, 20.0 * probs[i][m]);
					}
				}
			}
		}

		// CONSTRAINT 2: Peer Repulsion (Row/Col/Block)
		// For every peer pair, they should not have high probability for the SAME number.
		// E = sum( P(cell_A=k) * P(cell_B=k) )
		for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
			for (let j = i + 1; j < BOARD_SIZE * BOARD_SIZE; j++) {
				if (!arePeers(i, j)) continue;

				for (let k = 0; k < BOARD_SIZE; k++) {
					const pi = probs[i][k];
					const pj = probs[j][k];

					const conflict = pi * pj;
					energy += conflict * 12.0; // Increased penalty weight

					// Correct gradient via chain rule:
					// dE/dpi_k = pj * 12.0, dE/dpj_k = pi * 12.0
					// dpi_k/dlogit_i_m = pi_k * delta(k,m) - pi_k * pi_m
					//                  = pi_k * (1 - pi_m)  if k=m
					//                  = -pi_k * pi_m       if k≠m

					const dE_dpi = pj * 12.0;
					const dE_dpj = pi * 12.0;

					// Update ALL logits for both cells (full softmax jacobian)
					for (let m = 0; m < BOARD_SIZE; m++) {
						if (m === k) {
							// Diagonal: increase logit when pi_k is involved
							addGrad(i, m, dE_dpi * pi * (1 - pi));
							addGrad(j, m, dE_dpj * pj * (1 - pj));
						} else {
							// Off-diagonal: decrease other logits
							addGrad(i, m, -dE_dpi * pi * probs[i][m]);
							addGrad(j, m, -dE_dpj * pj * probs[j][m]);
						}
					}
				}
			}
		}

		// CONSTRAINT 3: Completeness (Each row/col/block must have all numbers)
		// For each group, sum of probabilities for each number should be 1.0
		const groups: number[][] = [];
		// Rows
		for (let r = 0; r < BOARD_SIZE; r++) {
			const row = [];
			for (let c = 0; c < BOARD_SIZE; c++) {
				row.push(r * BOARD_SIZE + c);
			}
			groups.push(row);
		}
		// Columns
		for (let c = 0; c < BOARD_SIZE; c++) {
			const col = [];
			for (let r = 0; r < BOARD_SIZE; r++) {
				col.push(r * BOARD_SIZE + c);
			}
			groups.push(col);
		}
		// Blocks (3x3)
		for (let br = 0; br < SQ_SIZE; br++) {
			for (let bc = 0; bc < SQ_SIZE; bc++) {
				const block = [];
				for (let r = 0; r < SQ_SIZE; r++) {
					for (let c = 0; c < SQ_SIZE; c++) {
						block.push((br * SQ_SIZE + r) * BOARD_SIZE + (bc * SQ_SIZE + c));
					}
				}
				groups.push(block);
			}
		}

		for (const group of groups) {
			for (let k = 0; k < BOARD_SIZE; k++) {
				// Sum probability of number k+1 in this group
				const sumProb = group.reduce((sum, cellIdx) => sum + probs[cellIdx][k], 0);
				const deviation = sumProb - 1.0; // Should be exactly 1.0
				energy += deviation * deviation * 8.0; // Quadratic penalty

				// Gradient: dE/dlogit = 2 * deviation * dp/dlogit
				for (const cellIdx of group) {
					const p_k = probs[cellIdx][k];
					for (let m = 0; m < BOARD_SIZE; m++) {
						if (m === k) {
							addGrad(cellIdx, m, 2 * 8.0 * deviation * p_k * (1 - p_k));
						} else {
							addGrad(cellIdx, m, -2 * 8.0 * deviation * p_k * probs[cellIdx][m]);
						}
					}
				}
			}
		}

		return { energy, grads };
	};

	const arePeers = (i: number, j: number) => {
		const r1 = Math.floor(i / BOARD_SIZE),
			c1 = i % BOARD_SIZE;
		const r2 = Math.floor(j / BOARD_SIZE),
			c2 = j % BOARD_SIZE;
		if (r1 === r2 || c1 === c2) return true;
		const b1 = Math.floor(r1 / SQ_SIZE) * SQ_SIZE + Math.floor(c1 / SQ_SIZE);
		const b2 = Math.floor(r2 / SQ_SIZE) * SQ_SIZE + Math.floor(c2 / SQ_SIZE);
		return b1 === b2;
	};

	const createInitialState = (): KonaState => {
		// Init latent with random noise (Implicit CoT initialization)
		const latent = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, () =>
			Array.from({ length: BOARD_SIZE }, () => Math.random() * 2.0 - 1.0),
		);
		const { energy } = calculateEnergyAndGradients(latent);
		return {
			latent,
			board: latent.map(argmax),
			energy,
			step: 0,
			gradientTrace: [{ id: 0, value: energy }],
		};
	};

	const [initialState, setInitialState] = useState(createInitialState);

	const { isRunning, state, start, stop, reset } = useSimulation<KonaState>({
		initialState,
		tickRate: 50,
		onTick: (prev) => performLangevinStep(prev),
	});

	useEffect(() => {
		reset();
	}, [reset]);

	const performLangevinStep = (prev: KonaState): Partial<KonaState> => {
		const { grads } = calculateEnergyAndGradients(prev.latent);

		const lr = 0.5; // Moderate learning rate for stability
		const noiseScale = Math.max(0.01, 0.8 * Math.exp(-prev.step / 100)); // Slower noise decay

		const nextLatent = prev.latent.map((logits, i) => {
			return logits.map((val, k) => {
				const grad = grads[i][k];
				const noise = (Math.random() - 0.5) * noiseScale;
				return val - lr * grad + noise;
			});
		});

		// CRITICAL: Recalculate energy AFTER the update
		const { energy: nextEnergy } = calculateEnergyAndGradients(nextLatent);
		const nextStep = prev.step + 1;

		// Convergence check
		if (nextEnergy < 0.1 && noiseScale < 0.2) {
			// Stop?
		}

		return {
			latent: nextLatent,
			board: nextLatent.map(argmax),
			energy: nextEnergy,
			step: nextStep,
			gradientTrace: [
				...prev.gradientTrace,
				{ id: nextStep, value: nextEnergy },
			].slice(-60),
		};
	};

	// Rendering
	const { board, latent, energy, gradientTrace } = state;
	// Safety
	if (!latent) return null;

	const getCellColor = (val: number, idx: number) => {
		if (PROBLEM[idx] !== 0) return "bg-neutral-800 text-white font-bold";

		// Visualize Confidence using the latent softmax
		const probs = softmax(latent[idx]);
		const maxP = Math.max(...probs);

		if (maxP < 0.5) return "bg-purple-100 text-purple-800 opacity-50"; // Uncertain (Fluid)
		if (maxP < 0.8) return "bg-yellow-100 text-yellow-800"; // Solidifying
		return "bg-green-100 text-green-800 font-bold"; // Crystallized
	};

	return (
		<div className="grid grid-cols-1 lg:grid-cols-2 gap-6 font-sans">
			<SchematicCard title="IMPLICIT CoT VISUALIZER">
				<div className="flex flex-col items-center gap-6">
					<p className="text-sm text-neutral-600 dark:text-neutral-400 text-center">
						<b>Langevin Calibration</b>: We define a "Soft Thought" tensor and
						minimize its energy.
						<br />
						Watch the purple "liquid" thoughts solidify into green discrete
						answers.
					</p>

					{/* Sudoku Grid */}
					<div className="relative p-1 bg-zinc-950 rounded-xl shadow-2xl border border-zinc-800">
						<div className="grid grid-cols-9 gap-px bg-zinc-950 border-2 border-zinc-900 rounded-lg overflow-hidden">
							{board.map((val, i) => {
								const col = i % BOARD_SIZE;
								const row = Math.floor(i / BOARD_SIZE);
								const isBlockRight = col === 2 || col === 5;
								const isBlockBottom = row === 2 || row === 5;
								
								// Confidence calculation for visual effects
								const probs = softmax(latent[i]);
								const maxP = Math.max(...probs);
								const isGiven = PROBLEM[i] !== 0;
								
								return (
									<div
										key={CELL_IDS[i]}
										className={`
											relative w-10 h-10 flex flex-col items-center justify-center 
											bg-zinc-800
											${isBlockRight ? "mr-0.5" : ""} 
											${isBlockBottom ? "mb-0.5" : ""}
											transition-colors duration-200
										`}
									>
										{/* Cell Value */}
										<span 
											className={`
												z-10 text-lg leading-none
												${isGiven ? "font-black text-white" : "font-semibold"}
												${!isGiven && maxP < 0.5 ? "text-zinc-500" : ""}
												${!isGiven && maxP >= 0.5 && maxP < 0.75 ? "text-indigo-400" : ""}
												${!isGiven && maxP >= 0.75 && maxP < 0.95 ? "text-indigo-300" : ""}
												${!isGiven && maxP >= 0.95 ? "text-emerald-400 font-bold" : ""}
											`}
											style={{
												opacity: isGiven ? 1 : Math.max(0.5, maxP)
											}}
										>
											{val}
										</span>

										{/* Top Candidate Probabilities (Mini Bar Chart) */}
										{!isGiven && (
											<div className="absolute bottom-0.5 left-0.5 right-0.5 h-1.5 flex items-end justify-center gap-[1px] opacity-80">
												{probs.map((p, k) => (
													<div
														key={k}
														className={`w-full rounded-t-[1px] transition-all duration-75 ${
															k + 1 === val ? "bg-indigo-500" : "bg-zinc-700"
														}`}
														style={{ height: `${p * 100}%` }}
													/>
												))}
											</div>
										)}
										
										{/* Active "Thinking" Glow */}
										{!isGiven && maxP < 0.95 && (
											<div 
												className="absolute inset-0 bg-indigo-500/5 pointer-events-none rounded"
												style={{ opacity: (1 - maxP) * 0.5 }} 
											/>
										)}
									</div>
								);
							})}
						</div>
					</div>

					<div className="flex gap-4 w-full">
						<SchematicButton
							label={isRunning ? "Pause Thinking" : "Start Calibration"}
							icon={isRunning ? <Activity /> : <BrainCircuit />}
							onClick={isRunning ? stop : start}
							active={isRunning}
						/>
						<SchematicButton
							label="Reset"
							icon={<RotateCcw />}
							onClick={() => {
								stop();
								setInitialState(createInitialState());
							}}
						/>
					</div>
				</div>
			</SchematicCard>

			<SchematicCard title="ENERGY LANDSCAPE">
				<div className="h-full flex flex-col justify-between">
					<div className="p-4 bg-zinc-900 rounded border border-zinc-800 mb-4">
						<div className="text-xs text-zinc-400 uppercase font-bold flex items-center gap-2">
							<Flame size={12} /> Implicit Energy E(z)
						</div>
						<div
							className={`text-3xl font-mono mt-1 ${energy < 1 ? "text-emerald-400" : "text-orange-400"}`}
						>
							{energy.toFixed(3)}
						</div>
						<div className="text-xs text-zinc-500 mt-1">
							Lower is more consistent
						</div>
					</div>

					<div className="flex-grow bg-zinc-900 rounded border border-zinc-800 relative overflow-hidden p-2 h-32">
					<svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
						<title>Energy minimization trace over time</title>
						{/* Y-axis grid lines */}
						<line x1="0" y1="25" x2="100" y2="25" stroke="currentColor" strokeWidth="0.2" opacity="0.2" vectorEffect="non-scaling-stroke" />
						<line x1="0" y1="50" x2="100" y2="50" stroke="currentColor" strokeWidth="0.2" opacity="0.2" vectorEffect="non-scaling-stroke" />
						<line x1="0" y1="75" x2="100" y2="75" stroke="currentColor" strokeWidth="0.2" opacity="0.2" vectorEffect="non-scaling-stroke" />
						
						{gradientTrace.length > 1 && (() => {
							// Dynamic scaling based on actual energy range
							const energies = gradientTrace.map(p => p.value);
							const maxE = Math.max(...energies);
							const minE = Math.min(...energies);
							const range = maxE - minE;
							const padding = range * 0.1 || 1; // Add 10% padding, minimum 1
							
							const scaleY = (val: number) => {
								if (range < 0.01) return 50; // Flat line in middle if no variation
								return 95 - ((val - minE + padding) / (range + 2 * padding)) * 90;
							};
							
							const points = gradientTrace
								.map((p, i) => {
									const x = (i / Math.max(gradientTrace.length - 1, 1)) * 100;
									const y = scaleY(p.value);
									return `${x},${y}`;
								})
								.join(" ");
							
							return (
								<>
									<polyline
										points={points}
										fill="none"
										stroke="#10b981"
										strokeWidth="1"
										vectorEffect="non-scaling-stroke"
									/>
									{/* Current point indicator */}
									{gradientTrace.length > 0 && (
										<circle
											cx={(gradientTrace.length - 1) / Math.max(gradientTrace.length - 1, 1) * 100}
											cy={scaleY(gradientTrace[gradientTrace.length - 1].value)}
											r="1"
											fill="#10b981"
											vectorEffect="non-scaling-stroke"
										/>
									)}
								</>
							);
						})()}
					</svg>
					<div className="absolute top-2 right-2 text-xs text-zinc-500">
						Langevin Descent Trace
					</div>
					{gradientTrace.length > 1 && (() => {
						const energies = gradientTrace.map(p => p.value);
						const maxE = Math.max(...energies);
						const minE = Math.min(...energies);
						return (
							<>
								<div className="absolute top-2 left-2 text-xs text-zinc-500">
									{maxE.toFixed(1)}
								</div>
								<div className="absolute bottom-2 left-2 text-xs text-zinc-500">
									{minE.toFixed(1)}
								</div>
							</>
						);
					})()}
				</div>

				<div
					className={`mt-4 flex items-center gap-2 text-sm font-bold ${energy < 1 ? "text-emerald-400" : "text-orange-400"}`}
				>
					{energy < 1 ? <ShieldCheck size={18} /> : <Activity size={18} />}
					{energy < 1 ? "THOUGHT CONVERGED" : "CALIBRATING LATENT STATE..."}
				</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default Kona1Simulation;
