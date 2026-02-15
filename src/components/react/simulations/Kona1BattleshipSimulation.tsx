import type { FC } from "react";
import { useEffect, useMemo, useState } from "react";
import {
	Activity,
	BrainCircuit,
	Flame,
	Gauge,
	RotateCcw,
	Ship,
	Waves,
} from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

const GRID_ROWS = 9;
const GRID_COLS = 15;
const CELL_COUNT = GRID_ROWS * GRID_COLS;

type ShipLength = 1 | 2 | 3 | 4 | 5;
type FleetCounts = Record<ShipLength, number>;

type Placement = {
	id: number;
	length: ShipLength;
	cells: number[];
	rowHits: number[];
	colHits: number[];
};

type FixedClue = {
	row: number;
	col: number;
	target: 0 | 1;
	label: string;
};

type Edge = {
	i: number;
	j: number;
	weight: number;
};

type EnergyBreakdown = {
	rows: number;
	cols: number;
	fleet: number;
	adjacency: number;
	fixed: number;
	binary: number;
	total: number;
};

type EvalResult = {
	energy: number;
	breakdown: EnergyBreakdown;
	gradLogits: number[];
	occupancy: number[];
	rowSums: number[];
	colSums: number[];
	fleetSums: FleetCounts;
};

type DiscreteAnalysis = {
	rowCounts: number[];
	colCounts: number[];
	fleetCounts: FleetCounts;
	invalidShips: number;
	diagonalContacts: number;
	fixedViolations: number;
};

type BattleshipState = {
	logits: number[];
	energy: number;
	breakdown: EnergyBreakdown;
	occupancy: number[];
	rowSums: number[];
	colSums: number[];
	fleetSums: FleetCounts;
	discreteFleet: FleetCounts;
	hardViolations: number;
	diagonalContacts: number;
	invalidShips: number;
	fixedViolations: number;
	step: number;
	trace: { id: number; value: number }[];
	phase: string;
};

const SHIP_LENGTHS: ShipLength[] = [1, 2, 3, 4, 5];

// Trafalgar Square-inspired 15x9 puzzle profile (rows x cols totals = 27)
const ROW_TARGETS = [3, 1, 2, 7, 2, 3, 3, 2, 4];
const COL_TARGETS = [3, 1, 1, 2, 2, 1, 4, 0, 1, 3, 1, 1, 5, 0, 2];

const FLEET_TARGET: FleetCounts = {
	1: 6,
	2: 3,
	3: 2,
	4: 1,
	5: 1,
};

// Used only as optional visual reference overlay.
const SOLUTION_CELLS: Array<[number, number]> = [
	[0, 12],
	[1, 12],
	[2, 12],
	[3, 12],
	[4, 12], // Aircraft carrier (col 13, rows 1-5)
	[3, 6],
	[4, 6],
	[5, 6],
	[6, 6], // 4-long ship (col 7, rows 4-7)
	[3, 0],
	[3, 1],
	[3, 2], // Cruiser
	[3, 8],
	[3, 9], // Destroyer
	[8, 9],
	[8, 10],
	[8, 11], // Cruiser
	[6, 0],
	[7, 0], // Destroyer
	[5, 3],
	[6, 3], // Destroyer
	[0, 4],
	[0, 14],
	[2, 4],
	[5, 9],
	[7, 14],
	[8, 5], // 6 submarines
];

const FIXED_CLUES: FixedClue[] = [
	{ row: 0, col: 12, target: 1, label: "Ship cap" },
	{ row: 4, col: 12, target: 1, label: "Ship cap" },
	{ row: 3, col: 6, target: 1, label: "Ship cap" },
	{ row: 6, col: 6, target: 1, label: "Ship cap" },
	{ row: 0, col: 11, target: 0, label: "Water" },
	{ row: 0, col: 13, target: 0, label: "Water" },
	{ row: 1, col: 11, target: 0, label: "Water" },
	{ row: 1, col: 13, target: 0, label: "Water" },
	{ row: 3, col: 5, target: 0, label: "Water" },
	{ row: 3, col: 7, target: 0, label: "Water" },
	{ row: 6, col: 5, target: 0, label: "Water" },
	{ row: 6, col: 7, target: 0, label: "Water" },
];

const idx = (row: number, col: number) => row * GRID_COLS + col;
const rowOf = (i: number) => Math.floor(i / GRID_COLS);
const colOf = (i: number) => i % GRID_COLS;

const clamp = (value: number, min: number, max: number) =>
	Math.max(min, Math.min(max, value));

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const randn = () => {
	let u = 0;
	let v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

const makeFleetCounts = (fill = 0): FleetCounts => ({
	1: fill,
	2: fill,
	3: fill,
	4: fill,
	5: fill,
});

const buildPlacements = (): Placement[] => {
	const placements: Placement[] = [];
	let id = 0;

	const addPlacement = (cells: number[], length: ShipLength) => {
		const rowHits = Array<number>(GRID_ROWS).fill(0);
		const colHits = Array<number>(GRID_COLS).fill(0);

		for (const cell of cells) {
			rowHits[rowOf(cell)] += 1;
			colHits[colOf(cell)] += 1;
		}

		placements.push({
			id,
			length,
			cells,
			rowHits,
			colHits,
		});
		id += 1;
	};

	for (const length of SHIP_LENGTHS) {
		if (length === 1) {
			for (let row = 0; row < GRID_ROWS; row++) {
				for (let col = 0; col < GRID_COLS; col++) {
					addPlacement([idx(row, col)], 1);
				}
			}
			continue;
		}

		for (let row = 0; row < GRID_ROWS; row++) {
			for (let col = 0; col <= GRID_COLS - length; col++) {
				const cells = Array.from({ length }, (_, d) => idx(row, col + d));
				addPlacement(cells, length);
			}
		}

		for (let row = 0; row <= GRID_ROWS - length; row++) {
			for (let col = 0; col < GRID_COLS; col++) {
				const cells = Array.from({ length }, (_, d) => idx(row + d, col));
				addPlacement(cells, length);
			}
		}
	}

	return placements;
};

const PLACEMENTS = buildPlacements();

const FIXED_BY_CELL = new Map<number, FixedClue>();
for (const clue of FIXED_CLUES) {
	FIXED_BY_CELL.set(idx(clue.row, clue.col), clue);
}

const SOLUTION_MASK = Array<number>(CELL_COUNT).fill(0);
for (const [row, col] of SOLUTION_CELLS) {
	SOLUTION_MASK[idx(row, col)] = 1;
}

const conflictWeight = (a: Placement, b: Placement): number => {
	let overlap = false;
	let touching = false;

	for (const cellA of a.cells) {
		const rowA = rowOf(cellA);
		const colA = colOf(cellA);

		for (const cellB of b.cells) {
			if (cellA === cellB) {
				overlap = true;
				touching = true;
				continue;
			}

			const dr = Math.abs(rowA - rowOf(cellB));
			const dc = Math.abs(colA - colOf(cellB));
			if (dr <= 1 && dc <= 1) {
				touching = true;
			}
		}
	}

	if (overlap) return 3;
	if (touching) return 1;
	return 0;
};

const CONFLICT_EDGES: Edge[] = [];
for (let i = 0; i < PLACEMENTS.length; i++) {
	for (let j = i + 1; j < PLACEMENTS.length; j++) {
		const weight = conflictWeight(PLACEMENTS[i], PLACEMENTS[j]);
		if (weight > 0) {
			CONFLICT_EDGES.push({ i, j, weight });
		}
	}
}

const WEIGHTS = {
	row: 1.1,
	col: 1.1,
	fleet: 9.0,
	adjacency: 3.2,
	fixed: 30.0,
	binary: 0.35,
};

const evaluate = (logits: number[], temperature: number): EvalResult => {
	const safeTemp = Math.max(0.2, temperature);
	const activations = logits.map((logit) => sigmoid(logit / safeTemp));

	const rowSums = Array<number>(GRID_ROWS).fill(0);
	const colSums = Array<number>(GRID_COLS).fill(0);
	const occupancy = Array<number>(CELL_COUNT).fill(0);
	const fleetSums = makeFleetCounts(0);

	for (const placement of PLACEMENTS) {
		const v = activations[placement.id];
		fleetSums[placement.length] += v;

		for (let row = 0; row < GRID_ROWS; row++) {
			if (placement.rowHits[row] !== 0) {
				rowSums[row] += placement.rowHits[row] * v;
			}
		}

		for (let col = 0; col < GRID_COLS; col++) {
			if (placement.colHits[col] !== 0) {
				colSums[col] += placement.colHits[col] * v;
			}
		}

		for (const cell of placement.cells) {
			occupancy[cell] += v;
		}
	}

	const gradV = Array<number>(PLACEMENTS.length).fill(0);
	const fixedCellGrad = Array<number>(CELL_COUNT).fill(0);

	let eRows = 0;
	let eCols = 0;
	let eFleet = 0;
	let eAdj = 0;
	let eFixed = 0;
	let eBinary = 0;

	const rowFactors = Array<number>(GRID_ROWS).fill(0);
	const colFactors = Array<number>(GRID_COLS).fill(0);
	const fleetFactors = makeFleetCounts(0);

	for (let row = 0; row < GRID_ROWS; row++) {
		const diff = rowSums[row] - ROW_TARGETS[row];
		eRows += WEIGHTS.row * diff * diff;
		rowFactors[row] = 2 * WEIGHTS.row * diff;
	}

	for (let col = 0; col < GRID_COLS; col++) {
		const diff = colSums[col] - COL_TARGETS[col];
		eCols += WEIGHTS.col * diff * diff;
		colFactors[col] = 2 * WEIGHTS.col * diff;
	}

	for (const length of SHIP_LENGTHS) {
		const diff = fleetSums[length] - FLEET_TARGET[length];
		eFleet += WEIGHTS.fleet * diff * diff;
		fleetFactors[length] = 2 * WEIGHTS.fleet * diff;
	}

	for (const clue of FIXED_CLUES) {
		const cell = idx(clue.row, clue.col);
		const diff = occupancy[cell] - clue.target;
		eFixed += WEIGHTS.fixed * diff * diff;
		fixedCellGrad[cell] += 2 * WEIGHTS.fixed * diff;
	}

	for (const edge of CONFLICT_EDGES) {
		const vi = activations[edge.i];
		const vj = activations[edge.j];
		const scaledWeight = WEIGHTS.adjacency * edge.weight;
		eAdj += scaledWeight * vi * vj;
		gradV[edge.i] += scaledWeight * vj;
		gradV[edge.j] += scaledWeight * vi;
	}

	for (const placement of PLACEMENTS) {
		let gradient = gradV[placement.id];

		for (let row = 0; row < GRID_ROWS; row++) {
			if (placement.rowHits[row] !== 0) {
				gradient += rowFactors[row] * placement.rowHits[row];
			}
		}

		for (let col = 0; col < GRID_COLS; col++) {
			if (placement.colHits[col] !== 0) {
				gradient += colFactors[col] * placement.colHits[col];
			}
		}

		gradient += fleetFactors[placement.length];

		for (const cell of placement.cells) {
			gradient += fixedCellGrad[cell];
		}

		const v = activations[placement.id];
		eBinary += WEIGHTS.binary * v * (1 - v);
		gradient += WEIGHTS.binary * (1 - 2 * v);

		gradV[placement.id] = gradient;
	}

	const gradLogits = activations.map(
		(v, i) => (gradV[i] * v * (1 - v)) / safeTemp,
	);

	const total = eRows + eCols + eFleet + eAdj + eFixed + eBinary;

	return {
		energy: total,
		breakdown: {
			rows: eRows,
			cols: eCols,
			fleet: eFleet,
			adjacency: eAdj,
			fixed: eFixed,
			binary: eBinary,
			total,
		},
		gradLogits,
		occupancy,
		rowSums,
		colSums,
		fleetSums,
	};
};

const analyzeDiscreteBoard = (occupancy: number[]): DiscreteAnalysis => {
	const board = occupancy.map((value) => (value >= 0.5 ? 1 : 0));
	const visited = Array<boolean>(CELL_COUNT).fill(false);
	const componentId = Array<number>(CELL_COUNT).fill(-1);
	const fleetCounts = makeFleetCounts(0);
	let invalidShips = 0;
	let componentCounter = 0;

	const dirs = [
		[-1, 0],
		[1, 0],
		[0, -1],
		[0, 1],
	];

	for (let cell = 0; cell < CELL_COUNT; cell++) {
		if (board[cell] === 0 || visited[cell]) continue;

		const queue = [cell];
		const componentCells: number[] = [];
		visited[cell] = true;
		componentId[cell] = componentCounter;

		while (queue.length > 0) {
			const current = queue.shift();
			if (current === undefined) break;

			componentCells.push(current);
			const row = rowOf(current);
			const col = colOf(current);

			for (const [dr, dc] of dirs) {
				const nr = row + dr;
				const nc = col + dc;
				if (nr < 0 || nr >= GRID_ROWS || nc < 0 || nc >= GRID_COLS) continue;
				const neighbor = idx(nr, nc);
				if (board[neighbor] === 0 || visited[neighbor]) continue;
				visited[neighbor] = true;
				componentId[neighbor] = componentCounter;
				queue.push(neighbor);
			}
		}

		const length = componentCells.length;
		const rows = new Set(componentCells.map((c) => rowOf(c)));
		const cols = new Set(componentCells.map((c) => colOf(c)));
		const isStraight = rows.size === 1 || cols.size === 1;

		if (!isStraight || length < 1 || length > 5) {
			invalidShips += 1;
		} else {
			fleetCounts[length as ShipLength] += 1;
		}

		componentCounter += 1;
	}

	let diagonalContacts = 0;
	for (let row = 0; row < GRID_ROWS; row++) {
		for (let col = 0; col < GRID_COLS; col++) {
			const current = idx(row, col);
			if (board[current] === 0) continue;

			for (const [dr, dc] of [
				[1, 1],
				[1, -1],
			]) {
				const nr = row + dr;
				const nc = col + dc;
				if (nr < 0 || nr >= GRID_ROWS || nc < 0 || nc >= GRID_COLS) continue;
				const neighbor = idx(nr, nc);
				if (board[neighbor] === 0) continue;
				if (componentId[neighbor] !== componentId[current]) {
					diagonalContacts += 1;
				}
			}
		}
	}

	const rowCounts = Array<number>(GRID_ROWS).fill(0);
	const colCounts = Array<number>(GRID_COLS).fill(0);
	for (let cell = 0; cell < CELL_COUNT; cell++) {
		if (board[cell] === 0) continue;
		rowCounts[rowOf(cell)] += 1;
		colCounts[colOf(cell)] += 1;
	}

	let fixedViolations = 0;
	for (const clue of FIXED_CLUES) {
		const value = board[idx(clue.row, clue.col)];
		if (value !== clue.target) {
			fixedViolations += 1;
		}
	}

	return {
		rowCounts,
		colCounts,
		fleetCounts,
		invalidShips,
		diagonalContacts,
		fixedViolations,
	};
};

const calculateHardViolations = (analysis: DiscreteAnalysis): number => {
	let violations = 0;

	for (let row = 0; row < GRID_ROWS; row++) {
		violations += Math.abs(analysis.rowCounts[row] - ROW_TARGETS[row]);
	}

	for (let col = 0; col < GRID_COLS; col++) {
		violations += Math.abs(analysis.colCounts[col] - COL_TARGETS[col]);
	}

	for (const length of SHIP_LENGTHS) {
		violations += Math.abs(
			analysis.fleetCounts[length] - FLEET_TARGET[length],
		);
	}

	violations += analysis.invalidShips * 3;
	violations += analysis.diagonalContacts * 3;
	violations += analysis.fixedViolations * 4;

	return violations;
};

const derivePhase = (
	step: number,
	energy: number,
	hardViolations: number,
): string => {
	if (hardViolations === 0 && energy < 35) {
		return "STEP 4 · GLOBAL MINIMUM REACHED";
	}
	if (step < 10) {
		return "STEP 1 · INITIAL HIGH-ENERGY STATE";
	}
	if (hardViolations > 0 && energy > 70) {
		return "STEP 2 · CONTINUOUS GRADIENT DESCENT";
	}
	return "STEP 3 · HOLISTIC PARALLEL REVISION";
};

const createInitialLogits = () => {
	const logits = Array<number>(PLACEMENTS.length).fill(0);

	for (const placement of PLACEMENTS) {
		let logit = -2.2 + (Math.random() - 0.5) * 1.8;

		let fixedShipHits = 0;
		let fixedWaterHits = 0;
		for (const cell of placement.cells) {
			const clue = FIXED_BY_CELL.get(cell);
			if (!clue) continue;
			if (clue.target === 1) fixedShipHits += 1;
			if (clue.target === 0) fixedWaterHits += 1;
		}

		logit += fixedShipHits * 2.1;
		logit -= fixedWaterHits * 3.2;

		logits[placement.id] = clamp(logit, -7, 7);
	}

	return logits;
};

const toState = (
	logits: number[],
	step: number,
	trace: { id: number; value: number }[],
	temperature: number,
	evaluated?: EvalResult,
): BattleshipState => {
	const evalResult = evaluated ?? evaluate(logits, temperature);
	const discrete = analyzeDiscreteBoard(evalResult.occupancy);
	const hardViolations = calculateHardViolations(discrete);
	const phase = derivePhase(step, evalResult.energy, hardViolations);

	const nextTrace = [...trace, { id: step, value: evalResult.energy }].slice(-90);

	return {
		logits,
		energy: evalResult.energy,
		breakdown: evalResult.breakdown,
		occupancy: evalResult.occupancy,
		rowSums: evalResult.rowSums,
		colSums: evalResult.colSums,
		fleetSums: evalResult.fleetSums,
		discreteFleet: discrete.fleetCounts,
		hardViolations,
		diagonalContacts: discrete.diagonalContacts,
		invalidShips: discrete.invalidShips,
		fixedViolations: discrete.fixedViolations,
		step,
		trace: nextTrace,
		phase,
	};
};

const Kona1BattleshipSimulation: FC = () => {
	const [learningRate, setLearningRate] = useState(0.09);
	const [noiseScale, setNoiseScale] = useState(0.11);
	const [temperature, setTemperature] = useState(0.9);
	const [showReference, setShowReference] = useState(false);

	const makeInitialState = () => {
		const logits = createInitialLogits();
		return toState(logits, 0, [], temperature);
	};

	const [initialState, setInitialState] = useState<BattleshipState>(
		makeInitialState,
	);

	const { isRunning, state, start, stop, reset } = useSimulation<BattleshipState>(
		{
			initialState,
			tickRate: 120,
			onTick: (prev) => {
				const prevEval = evaluate(prev.logits, temperature);
				const annealedNoise = noiseScale * Math.exp(-prev.step / 220);

				const nextLogits = prev.logits.map((value, i) =>
					clamp(
						value -
							learningRate * prevEval.gradLogits[i] +
							randn() * annealedNoise,
						-8,
						8,
					),
				);

				const nextEval = evaluate(nextLogits, temperature);
				return toState(
					nextLogits,
					prev.step + 1,
					prev.trace,
					temperature,
					nextEval,
				);
			},
		},
	);

	useEffect(() => {
		reset();
	}, [reset]);

	useEffect(() => {
		if (isRunning && state.hardViolations === 0 && state.energy < 35) {
			stop();
		}
	}, [isRunning, state.hardViolations, state.energy, stop]);

	const tracePoints = useMemo(() => {
		if (state.trace.length <= 1) return "";
		const values = state.trace.map((point) => point.value);
		const min = Math.min(...values);
		const max = Math.max(...values);
		const range = Math.max(1, max - min);
		return state.trace
			.map((point, index) => {
				const x = (index / Math.max(1, state.trace.length - 1)) * 100;
				const y = 95 - ((point.value - min) / range) * 90;
				return `${x},${y}`;
			})
			.join(" ");
	}, [state.trace]);

	return (
		<div className="grid grid-cols-1 xl:grid-cols-[1.35fr_1fr] gap-6 text-slate-200">
			<SchematicCard title="KONA-1 BATTLESHIP CSP (15x9)">
				<div className="space-y-5">
					<p className="text-xs leading-relaxed text-slate-400">
						This simulation uses a global energy function over the whole grid:
						<span className="text-emerald-300"> E_rows</span>,
						<span className="text-blue-300"> E_cols</span>,
						<span className="text-fuchsia-300"> E_fleet</span>,
						<span className="text-amber-300"> E_adjacency</span>, and
						<span className="text-rose-300"> E_fixed</span>. Every ship
						placement variable is updated in parallel with Langevin-style
						descent.
					</p>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-3">
						<label className="block">
							<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500 mb-1">
								<span>Learning Rate</span>
								<span className="text-slate-200">{learningRate.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min={0.03}
								max={0.2}
								step={0.01}
								value={learningRate}
								onChange={(event) => setLearningRate(Number(event.target.value))}
								className="w-full accent-emerald-400"
							/>
						</label>

						<label className="block">
							<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500 mb-1">
								<span>Noise Scale</span>
								<span className="text-slate-200">{noiseScale.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min={0}
								max={0.25}
								step={0.01}
								value={noiseScale}
								onChange={(event) => setNoiseScale(Number(event.target.value))}
								className="w-full accent-fuchsia-400"
							/>
						</label>

						<label className="block">
							<div className="flex justify-between text-[10px] font-mono uppercase text-slate-500 mb-1">
								<span>Temperature</span>
								<span className="text-slate-200">{temperature.toFixed(2)}</span>
							</div>
							<input
								type="range"
								min={0.35}
								max={1.2}
								step={0.05}
								value={temperature}
								onChange={(event) => setTemperature(Number(event.target.value))}
								className="w-full accent-blue-400"
							/>
						</label>

						<label className="flex items-center gap-2 text-[11px] font-mono uppercase text-slate-400">
							<input
								type="checkbox"
								checked={showReference}
								onChange={(event) => setShowReference(event.target.checked)}
								className="accent-amber-400"
							/>
							Show Reference Solution
						</label>
					</div>

					<div className="flex flex-wrap gap-2">
						<SchematicButton
							label={isRunning ? "Pause Solver" : "Run Solver"}
							icon={isRunning ? <Activity size={14} /> : <BrainCircuit size={14} />}
							onClick={isRunning ? stop : start}
							active={isRunning}
						/>
						<SchematicButton
							label="Reset State"
							icon={<RotateCcw size={14} />}
							onClick={() => {
								stop();
								setInitialState(makeInitialState());
							}}
						/>
					</div>

					<div className="overflow-x-auto">
						<div className="inline-block min-w-max rounded-lg border border-slate-700/50 bg-black/40 p-2">
							<div className="flex pl-10 pb-1 gap-0.5">
								{COL_TARGETS.map((target, col) => {
									const diff = Math.abs(state.colSums[col] - target);
									return (
										<div
											key={`col-target-${col}`}
											className="w-6 text-center font-mono leading-none"
										>
											<div className="text-[10px] text-slate-400">{target}</div>
											<div
												className={`text-[9px] ${
													diff < 0.25 ? "text-emerald-400" : "text-amber-400"
												}`}
											>
												{state.colSums[col].toFixed(1)}
											</div>
										</div>
									);
								})}
							</div>

							{Array.from({ length: GRID_ROWS }, (_, row) => (
								<div key={`row-${row}`} className="flex items-center gap-0.5">
									<div className="w-10 text-right font-mono leading-none pr-1">
										<div className="text-[10px] text-slate-400">
											{ROW_TARGETS[row]}
										</div>
										<div
											className={`text-[9px] ${
												Math.abs(state.rowSums[row] - ROW_TARGETS[row]) < 0.25
													? "text-emerald-400"
													: "text-amber-400"
											}`}
										>
											{state.rowSums[row].toFixed(1)}
										</div>
									</div>

									{Array.from({ length: GRID_COLS }, (_, col) => {
										const cell = idx(row, col);
										const occupancy = clamp(state.occupancy[cell], 0, 1);
										const active = occupancy >= 0.5;
										const clue = FIXED_BY_CELL.get(cell);
										const inReference = SOLUTION_MASK[cell] === 1;

										let borderColor = "rgba(51,65,85,0.65)";
										if (showReference && inReference) borderColor = "#34d399";
										if (clue?.target === 1) borderColor = "#10b981";
										if (clue?.target === 0) borderColor = "#475569";

										const bg = clue
											? clue.target === 1
												? "rgba(16,185,129,0.5)"
												: "rgba(51,65,85,0.38)"
											: active
												? `rgba(56,189,248,${0.24 + occupancy * 0.62})`
												: `rgba(15,23,42,${0.6 + (1 - occupancy) * 0.25})`;

										return (
											<div
												key={`cell-${row}-${col}`}
												className="w-6 h-6 flex items-center justify-center text-[10px] font-mono"
												style={{
													backgroundColor: bg,
													border: `1px solid ${borderColor}`,
												}}
												title={`r${row + 1} c${col + 1} = ${occupancy.toFixed(2)}${clue ? ` (${clue.label})` : ""}`}
											>
												{clue?.target === 0 && (
													<span className="text-slate-300">×</span>
												)}
												{clue?.target === 1 && (
													<span className="text-emerald-100">●</span>
												)}
												{!clue && active && (
													<span className="text-cyan-100 opacity-80">■</span>
												)}
											</div>
										);
									})}
								</div>
							))}
						</div>
					</div>

					<div className="text-[11px] leading-relaxed text-slate-500 border border-slate-700/50 rounded-md p-3 bg-slate-900/40">
						<div className="font-mono uppercase text-slate-300 mb-1">
							What to try
						</div>
						<div>1. Set noise to 0.16 and watch how conflicts spike before cooling.</div>
						<div>
							2. Lower temperature to 0.45 to force crisper 0/1 ship placement.
						</div>
						<div>
							3. Raise learning rate above 0.15 to see oscillation from
							over-correction.
						</div>
					</div>
				</div>
			</SchematicCard>

			<SchematicCard title="ENERGY BREAKDOWN + FLEET CHECK">
				<div className="h-full flex flex-col gap-4">
					<div className="grid grid-cols-2 gap-3">
						<div className="border border-slate-700/50 rounded-md p-3 bg-black/30">
							<div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-2">
								<Flame size={12} /> Total Energy
							</div>
							<div className="text-3xl font-mono text-emerald-300 mt-1">
								{state.energy.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/50 rounded-md p-3 bg-black/30">
							<div className="text-[10px] font-mono uppercase text-slate-500 flex items-center gap-2">
								<Gauge size={12} /> Hard Violations
							</div>
							<div
								className={`text-3xl font-mono mt-1 ${
									state.hardViolations === 0
										? "text-emerald-300"
										: "text-amber-300"
								}`}
							>
								{state.hardViolations}
							</div>
						</div>
					</div>

					<div className="text-[11px] font-mono uppercase text-slate-400 border border-slate-700/50 rounded-md p-2 bg-slate-900/40">
						{state.phase}
					</div>

					<div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_rows</div>
							<div className="text-emerald-300">
								{state.breakdown.rows.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_cols</div>
							<div className="text-blue-300">
								{state.breakdown.cols.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_fleet</div>
							<div className="text-fuchsia-300">
								{state.breakdown.fleet.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_adjacency</div>
							<div className="text-amber-300">
								{state.breakdown.adjacency.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_fixed</div>
							<div className="text-rose-300">
								{state.breakdown.fixed.toFixed(1)}
							</div>
						</div>
						<div className="border border-slate-700/40 rounded p-2 bg-slate-900/30">
							<div className="text-slate-500">E_binary</div>
							<div className="text-slate-300">
								{state.breakdown.binary.toFixed(1)}
							</div>
						</div>
					</div>

					<div className="border border-slate-700/50 rounded-md p-3 bg-black/30">
						<div className="text-[10px] font-mono uppercase text-slate-500 mb-2 flex items-center gap-2">
							<Ship size={12} /> Fleet Balance (Target / Soft / Hard)
						</div>
						<div className="space-y-1 text-[11px] font-mono">
							{[5, 4, 3, 2, 1].map((length) => {
								const l = length as ShipLength;
								const hardDiff = Math.abs(state.discreteFleet[l] - FLEET_TARGET[l]);
								return (
									<div key={`fleet-${l}`} className="flex justify-between">
										<span className="text-slate-400">L{l}</span>
										<span className="text-slate-200">{FLEET_TARGET[l]}</span>
										<span className="text-blue-300">
											{state.fleetSums[l].toFixed(2)}
										</span>
										<span
											className={
												hardDiff === 0 ? "text-emerald-300" : "text-amber-300"
											}
										>
											{state.discreteFleet[l]}
										</span>
									</div>
								);
							})}
						</div>
					</div>

					<div className="border border-slate-700/50 rounded-md p-2 bg-slate-900/30 h-36 relative overflow-hidden">
						<svg
							viewBox="0 0 100 100"
							className="w-full h-full"
							preserveAspectRatio="none"
						>
							<line
								x1="0"
								y1="25"
								x2="100"
								y2="25"
								stroke="currentColor"
								strokeWidth="0.2"
								opacity="0.2"
							/>
							<line
								x1="0"
								y1="50"
								x2="100"
								y2="50"
								stroke="currentColor"
								strokeWidth="0.2"
								opacity="0.2"
							/>
							<line
								x1="0"
								y1="75"
								x2="100"
								y2="75"
								stroke="currentColor"
								strokeWidth="0.2"
								opacity="0.2"
							/>
							{tracePoints && (
								<polyline
									points={tracePoints}
									fill="none"
									stroke="#10b981"
									strokeWidth="1.2"
								/>
							)}
						</svg>
						<div className="absolute top-2 right-2 text-[10px] font-mono text-slate-500 uppercase">
							Energy Trace
						</div>
					</div>

					<div className="text-[11px] text-slate-500 flex items-center gap-2 font-mono uppercase">
						<Waves size={12} />
						Diagonal contacts:{" "}
						<span
							className={
								state.diagonalContacts === 0
									? "text-emerald-300"
									: "text-amber-300"
							}
						>
							{state.diagonalContacts}
						</span>
						<span className="text-slate-700">|</span>
						Invalid ships:{" "}
						<span
							className={
								state.invalidShips === 0 ? "text-emerald-300" : "text-amber-300"
							}
						>
							{state.invalidShips}
						</span>
						<span className="text-slate-700">|</span>
						Fixed clues broken:{" "}
						<span
							className={
								state.fixedViolations === 0
									? "text-emerald-300"
									: "text-amber-300"
							}
						>
							{state.fixedViolations}
						</span>
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default Kona1BattleshipSimulation;
