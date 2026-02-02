import React, { useState, useEffect, useRef } from "react";
import katex from "katex";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";

interface AlgorithmBlockProps {
	title: string;
	inputs?: string[];
	outputs?: string[];
	steps: string[];
	executor?: string; // Name of the executor to use
	initialState?: any;
}

interface ExecutionStep {
	step: number;
	state: Record<string, any>;
	description?: string;
}

// Registry of named executors
type ExecutorFn = (
	initialState: any,
) => AsyncGenerator<ExecutionStep, void, unknown>;
const executors: Record<string, ExecutorFn> = {};

const randn = () => {
	let u = 0;
	let v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

executors["langevin-calibration"] = async function* (initialState) {
	let { z, E, lambda } = initialState;

	// Dummy target for the latent thought: [0.0, 1.0, 0.0]
	const target = [0.0, 1.0, 0.0];

	for (let step = 1; step <= 5; step++) {
		// 1. Compute Gradients
		const grad = z.map((val: number, i: number) => val - target[i]);

		// 2. Langevin Update
		const noise = z.map(() => (Math.random() - 0.5) * 0.2);
		const z_new = z.map(
			(val: number, i: number) => val - lambda * grad[i] + noise[i],
		);

		// 3. New Energy
		E = E * 0.8; // Pretend energy goes down

		yield {
			step,
			state: {
				z_t: `[${z.map((n: number) => n.toFixed(2)).join(", ")}]`,
				E: E.toFixed(2),
				"\\nabla E": `[${grad.map((n: number) => n.toFixed(1)).join(", ")}]`,
			},
			description:
				"Gradient descent pushes z toward the valid region, noise explores local structure.",
		};

		z = z_new;
	}
};

// TTT-E2E Executor
executors["ttt-e2e"] = async function* (initialState) {
	const { W, x_t, eta } = initialState;

	// Step 1: Predict x_{t+1} using W
	const prediction = W.map((row: number[]) =>
		row.reduce((sum: number, w: number, j: number) => sum + w * x_t[j], 0),
	);

	yield {
		step: 1,
		state: { W, x_t, prediction },
		description:
			"Forward pass: multiply weight matrix W with input x_t to get prediction",
	};

	// Step 2: Observe true x_{t+1}
	const x_next = [0.8, 0.6];
	const loss = prediction.reduce(
		(sum: number, pred: number, i: number) =>
			sum + Math.pow(pred - x_next[i], 2),
		0,
	);

	yield {
		step: 2,
		state: { W, x_t, prediction, x_next, loss },
		description: "Observe ground truth x_{t+1} and compute loss",
	};

	// Step 3: Update W using SGD
	const gradient = W.map((row: number[], i: number) =>
		row.map((_: number, j: number) => 2 * (prediction[i] - x_next[i]) * x_t[j]),
	);

	const W_new = W.map((row: number[], i: number) =>
		row.map((w: number, j: number) => w - eta * gradient[i][j]),
	);

	yield {
		step: 3,
		state: { W_old: W, gradient, W_new, eta, x_next },
		description: "Apply gradient descent: W ← W - η∇L",
	};
};

executors["additive-secret-sharing"] = async function* (initialState) {
	const { x, multiplier } = initialState;

	// Step 1: Split x into random shares
	const x1 = Math.floor(Math.random() * x); // Random share 1
	const x2 = x - x1; // Share 2

	yield {
		step: 1,
		state: { x, x_1: x1, x_2: x2 },
		description: `Split secret ${x} into shares: ${x1} + ${x2} = ${x}`,
	};

	// Step 2: Distribute shares
	yield {
		step: 2,
		state: { "Server 1": x1, "Server 2": x2 },
		description: "Send shares to separate servers. Neither server knows x.",
	};

	// Step 3: Local Computation (Multiply by multiplier)
	const y1 = x1 * multiplier;
	const y2 = x2 * multiplier;

	yield {
		step: 3,
		state: {
			y_1: `${x1} \\times ${multiplier} = ${y1}`,
			y_2: `${x2} \\times ${multiplier} = ${y2}`,
		},
		description: `Each server multiplies its share by ${multiplier} locally.`,
	};

	// Step 4: Reconstruct
	const y = y1 + y2;
	yield {
		step: 4,
		state: {
			y: `${y1} + ${y2} = ${y}`,
			Expected: `${x} \\times ${multiplier} = ${x * multiplier}`,
		},
		description: "Sum the results from both servers to get the final answer.",
	};
};

// SEAL ReST-EM Executor
executors["seal-rest-em"] = async function* (initialState) {
	const { num_tasks, num_edits, top_k, iteration } = initialState;

	// Step 1: Sample task from distribution
	const task = { context: "New factual passage", type: "knowledge" };

	yield {
		step: 1,
		state: {
			task_context: task.context,
			task_type: task.type,
			num_tasks,
		},
		description:
			"Sample a task from the distribution (e.g., knowledge incorporation or few-shot learning)",
	};

	// Step 2: Generate N self-edits
	const edits = Array.from({ length: num_edits }, (_, i) => ({
		id: i,
		synthetic_data: [`Q: Fact ${i}? A: Answer ${i}`],
		lr: (1 + Math.random() * 9) * 1e-5,
	}));

	yield {
		step: 2,
		state: {
			num_edits,
			edits: `[${edits.length} candidates]`,
			sample_edit: JSON.stringify(edits[0], null, 2).slice(0, 60) + "...",
		},
		description: "Generate N candidate self-edits using current policy π_φ",
	};

	// Step 3: Apply each edit via SFT
	const updated_models = edits.map((_edit, i) => ({
		edit_id: i,
		sft_steps: Math.floor(Math.random() * 100 + 50),
	}));

	yield {
		step: 3,
		state: {
			updates: `${updated_models.length} models finetuned`,
			avg_sft_steps: Math.floor(
				updated_models.reduce((sum, m) => sum + m.sft_steps, 0) /
					updated_models.length,
			),
		},
		description: "Apply each self-edit to base model via supervised finetuning",
	};

	// Step 4: Evaluate and compute rewards
	const rewards = edits.map(() => 0.3 + Math.random() * 0.5);
	const avg_reward = rewards.reduce((a, b) => a + b, 0) / rewards.length;

	yield {
		step: 4,
		state: {
			rewards: `[${rewards.map((r) => r.toFixed(2)).join(", ")}]`,
			avg_reward: avg_reward.toFixed(3),
		},
		description: "Evaluate updated models on downstream task to get rewards",
	};

	// Step 5: Select top-k edits
	const sorted_indices = rewards
		.map((r, i) => ({ reward: r, idx: i }))
		.sort((a, b) => b.reward - a.reward)
		.slice(0, top_k);

	yield {
		step: 5,
		state: {
			top_k,
			selected_edits: sorted_indices.map(
				(s) => `Edit ${s.idx} (r=${s.reward.toFixed(2)})`,
			),
		},
		description: "Rejection sampling: keep top-k highest-reward edits per task",
	};

	// Step 6: Update policy via SFT
	const policy_improvement = (avg_reward - 0.5) * 100;

	yield {
		step: 6,
		state: {
			phi_update: "∇_φ E[log π_φ(e|c)]",
			policy_improvement: `+${policy_improvement.toFixed(1)}%`,
			next_iteration: iteration + 1,
		},
		description:
			"Finetune policy on high-reward edits to improve self-edit generation",
	};
};

// AlphaGenome Variant Scoring Executor
executors["alphagenome-variant-score"] = async function* (initialState) {
	const { ref_props, variant_effect } = initialState;

	// Step 1: Predict properties for reference sequence
	const reference = ref_props.map((v: number) => Number(v.toFixed(3)));

	yield {
		step: 1,
		state: {
			reference,
		},
		description:
			"Run the model on the reference sequence to predict regulatory properties.",
	};

	// Step 2: Predict properties for mutated sequence
	const mutated = reference.map(
		(v: number, i: number) => Number((v + variant_effect[i]).toFixed(3)),
	);

	yield {
		step: 2,
		state: {
			reference,
			mutated,
		},
		description:
			"Run the same model on the mutated sequence to capture the variant's effect.",
	};

	// Step 3: Compute delta and summarize impact
	const delta = mutated.map((v: number, i: number) =>
		Number((v - reference[i]).toFixed(3)),
	);
	const score =
		delta.reduce((sum: number, d: number) => sum + Math.abs(d), 0) /
		delta.length;

	yield {
		step: 3,
		state: {
			delta,
			score: Number(score.toFixed(3)),
		},
		description:
			"Summarize the variant by averaging absolute changes across modalities.",
	};
};

// VVPA Mechanism Executor
executors["vvpa-mechanism"] = async function* (initialState) {
	const { xt, m, theta } = initialState;

	// Step 1: Generate standard Value
	const W_V = [
		[0.1, 0.2, 0.3, 0.4],
		[0.5, 0.6, 0.7, 0.8],
		[0.9, 1.0, 1.1, 1.2],
		[1.3, 1.4, 1.5, 1.6],
	];
	const v_t = W_V.map((row) => row.reduce((sum, w, j) => sum + w * xt[j], 0));

	yield {
		step: 1,
		state: { xt, W_V, v_t },
		description: "Generate semantic value vector v_t = W_V * x_t",
	};

	// Step 2: Compute Positional Rotation
	const angle = m / Math.pow(theta, 0); // Simplified for visualization
	const R_m = [
		Math.cos(angle),
		Math.sin(angle),
		Math.cos(angle),
		Math.sin(angle),
	];

	yield {
		step: 2,
		state: { m, theta, R_m },
		description: "Compute RoPE-like rotational vector R_m for position m",
	};

	// Step 3: Inject Position
	const v_pos = v_t.map((v, i) => v * R_m[i]);

	yield {
		step: 3,
		state: { v_t, R_m, v_pos },
		description: "Modulate value vector with position: v_pos = v_t ⊙ R_m",
	};

	// Step 4: Cache
	yield {
		step: 4,
		state: { v_pos, cache_status: "SAVED", index: m },
		description: "Store position-aware value in KV cache",
	};
};

// Deep Delta Executor
executors["deep-delta"] = async function* (initialState) {
	const { X, k, beta, v } = initialState;

	// Step 1: Calculate projection k^T * X (simplified for 2D case)
	const proj_scalar = k[0] * X[0][0] + k[1] * X[1][0];

	yield {
		step: 1,
		state: { X, k, proj_scalar },
		description: "Calculate the projection k^T * X along the direction vector",
	};

	// Step 2: Compute difference (v - proj)
	const diff = v.map((vi: number) => vi - proj_scalar);

	yield {
		step: 2,
		state: { v, proj_scalar, diff },
		description: "Compute the difference between target value v and projection",
	};

	// Step 3: Update X_{l+1} = X_l + β * k * (v - proj)
	const X_new = X.map((row: number[], i: number) =>
		row.map((x: number) => x + beta * k[i] * diff[i]),
	);

	yield {
		step: 3,
		state: { X_old: X, beta, k, diff, X_new },
		description: "Update state: X_{l+1} = X_l + β * k * (v - k^T X)",
	};
};

// Langevin Dynamics Executor
executors["langevin-dynamics"] = async function* (initialState) {
	const { z, lambda, temperature, noise } = initialState;

	// Step 1: Evaluate Energy (quadratic proxy)
	const energy = z.reduce((sum: number, v: number) => sum + v * v, 0);
	const E = 0.5 * energy;

	yield {
		step: 1,
		state: { z, E },
		description: "Compute energy E(z) = 1/2 ||z||^2 as a proxy landscape",
	};

	// Step 2: Compute Gradient
	const grad = z.map((v: number) => v);

	yield {
		step: 2,
		state: { z, grad },
		description: "Gradient of E is ∇E(z) = z for the quadratic proxy",
	};

	// Step 3: Langevin Update
	const scale = Math.sqrt(2 * lambda * temperature);
	const eps = z.map(() => randn() * noise);
	const z_next = z.map(
		(v: number, i: number) => v - lambda * grad[i] + scale * eps[i],
	);

	yield {
		step: 3,
		state: { z, grad, epsilon: eps, z_next, lambda, temperature },
		description: "Update with noise: z_{t+1} = z_t - λ∇E(z_t) + √(2λT) ε",
	};
};

// Helper function to render LaTeX inline math
function renderMath(text: string): string {
	return text.replace(/\$([^$]+)\$/g, (_, math) => {
		try {
			return katex.renderToString(math, {
				throwOnError: false,
				displayMode: false,
			});
		} catch (e) {
			return `$${math}$`;
		}
	});
}

// Helper to format values for display
function formatValue(value: any): string {
	if (Array.isArray(value)) {
		if (value.length > 0 && Array.isArray(value[0])) {
			return `[${value.map((row) => `[${row.map((v: number) => v.toFixed(4)).join(", ")}]`).join(", ")}]`;
		}
		return `[${value.map((v: number) => (typeof v === "number" ? v.toFixed(4) : v)).join(", ")}]`;
	}
	if (typeof value === "number") {
		return value.toFixed(4);
	}
	return String(value);
}

export default function AlgorithmBlock({
	title,
	inputs,
	outputs,
	steps,
	executor,
	initialState,
}: AlgorithmBlockProps) {
	// Look up the executor function from the registry
	const execute = executor ? executors[executor] : undefined;

	console.log("[AlgorithmBlock] Props received:", {
		title,
		executor,
		hasExecute: !!execute,
		initialState,
	});

	const [currentStep, setCurrentStep] = useState(0);
	const [executionHistory, setExecutionHistory] = useState<ExecutionStep[]>([]);
	const [isPlaying, setIsPlaying] = useState(false);
	const [isComplete, setIsComplete] = useState(false);
	const generatorRef = useRef<AsyncGenerator<
		ExecutionStep,
		void,
		unknown
	> | null>(null);
	const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

	const totalSteps = steps.length;

	const resetExecution = () => {
		setCurrentStep(0);
		setExecutionHistory([]);
		setIsPlaying(false);
		setIsComplete(false);
		generatorRef.current = null;
		if (playIntervalRef.current) {
			clearInterval(playIntervalRef.current);
			playIntervalRef.current = null;
		}
	};

	const stepForward = async () => {
		if (!execute || !initialState || isComplete) return;

		if (!generatorRef.current) {
			generatorRef.current = execute(initialState);
		}

		const result = await generatorRef.current!.next();

		if (result.done) {
			setIsPlaying(false);
			setIsComplete(true);
			if (playIntervalRef.current) {
				clearInterval(playIntervalRef.current);
				playIntervalRef.current = null;
			}
			return;
		}

		const step = result.value;
		setExecutionHistory((prev) => [...prev, step]);
		setCurrentStep(step.step);
	};

	const togglePlay = () => {
		setIsPlaying(!isPlaying);
	};

	// Handle play loop
	useEffect(() => {
		if (isPlaying && !playIntervalRef.current) {
			stepForward();
			playIntervalRef.current = setInterval(() => {
				stepForward();
			}, 1000);
		} else if (!isPlaying && playIntervalRef.current) {
			clearInterval(playIntervalRef.current);
			playIntervalRef.current = null;
		}

		return () => {
			if (playIntervalRef.current) {
				clearInterval(playIntervalRef.current);
				playIntervalRef.current = null;
			}
		};
	}, [isPlaying]);

	const currentState = executionHistory[executionHistory.length - 1];

	return (
		<div className="my-8 rounded-none border border-zinc-800 bg-zinc-900/50 shadow-sm">
			{/* Header */}
			<div className="border-b border-zinc-800 bg-zinc-950/50 p-6">
				<h3 className="text-sm font-mono font-bold uppercase tracking-widest text-white">
					{title}
				</h3>

				{inputs && (
					<div className="mt-3 font-mono text-sm text-zinc-400">
						<span className="font-bold text-zinc-200">Input:</span>{" "}
						{inputs.map((input, idx) => (
							<span key={idx}>
								<span dangerouslySetInnerHTML={{ __html: renderMath(input) }} />
								{idx < inputs.length - 1 && ", "}
							</span>
						))}
					</div>
				)}
			</div>

			{/* Algorithm Steps */}
			<div className="p-6">
				<ol className="space-y-2 font-mono text-sm leading-relaxed text-zinc-300 list-decimal list-inside mb-6">
					{steps.map((step, idx) => {
						const stepNumber = idx + 1;
						const isActive = currentStep === stepNumber;
						const isPast = currentStep > stepNumber;

						return (
							<li
								key={idx}
								className={`transition-all duration-300 ${
									isActive
										? "text-white font-bold bg-indigo-500/10 -mx-2 px-2 py-1 border-l-2 border-indigo-500"
										: isPast
											? "text-zinc-500"
											: "text-zinc-300"
								}`}
								dangerouslySetInnerHTML={{ __html: renderMath(step) }}
							/>
						);
					})}
				</ol>

				{/* Execution Controls */}
				{execute && initialState && (
					<div className="space-y-4">
						<div className="flex items-center gap-2 border-t border-zinc-800 pt-4">
							<button
								onClick={togglePlay}
								disabled={isComplete}
								className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white font-mono text-xs uppercase tracking-wider transition-colors cursor-pointer"
							>
								{isPlaying ? (
									<>
										<Pause className="w-4 h-4" />
										Pause
									</>
								) : (
									<>
										<Play className="w-4 h-4" />
										Play
									</>
								)}
							</button>

							<button
								onClick={stepForward}
								disabled={isPlaying || isComplete}
								className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 disabled:bg-zinc-800 disabled:text-zinc-600 text-zinc-300 font-mono text-xs uppercase tracking-wider transition-colors cursor-pointer"
							>
								<SkipForward className="w-4 h-4" />
								Step
							</button>

							<button
								onClick={resetExecution}
								className="flex items-center gap-2 px-4 py-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 font-mono text-xs uppercase tracking-wider transition-colors cursor-pointer"
							>
								<RotateCcw className="w-4 h-4" />
								Reset
							</button>

							<div className="ml-auto font-mono text-xs text-zinc-500">
								Step {currentStep} / {totalSteps}
							</div>
						</div>

						{/* State Visualization */}
						{currentState && (
							<div className="border border-zinc-800 bg-zinc-950/50 p-4">
								<div className="mb-3 text-xs font-mono font-bold uppercase tracking-widest text-zinc-400">
									Current State
								</div>
								<div className="space-y-2">
									{Object.entries(currentState.state).map(([key, value]) => (
										<div
											key={key}
											className="flex items-start gap-3 font-mono text-sm"
										>
											<span className="text-indigo-400 min-w-[80px]">
												{key}:
											</span>
											<span className="text-zinc-200 font-medium break-all">
												{formatValue(value)}
											</span>
										</div>
									))}
								</div>
								{currentState.description && (
									<div className="mt-3 pt-3 border-t border-zinc-800 text-sm text-zinc-400 italic">
										{currentState.description}
									</div>
								)}
							</div>
						)}
					</div>
				)}
			</div>

			{/* Outputs */}
			{outputs && (
				<div className="border-t border-zinc-800 bg-zinc-950/50 p-6">
					<div className="font-mono text-sm text-zinc-400">
						<span className="font-bold text-zinc-200">Output:</span>{" "}
						{outputs.map((output, idx) => (
							<span key={idx}>
								<span
									dangerouslySetInnerHTML={{ __html: renderMath(output) }}
								/>
								{idx < outputs.length - 1 && ", "}
							</span>
						))}
					</div>
				</div>
			)}
		</div>
	);
}
