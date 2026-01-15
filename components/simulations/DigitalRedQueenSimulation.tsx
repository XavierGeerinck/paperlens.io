import React, { useState, useEffect, useCallback } from "react";
import {
	Play,
	Pause,
	RotateCcw,
	ChevronRight,
	Disc,
	Crosshair,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

// --- CORE WAR CONSTANTS ---
const CORE_SIZE = 128; // Larger core for more interesting patterns
const MAX_CYCLES = 800;
const MAX_PROCESSES_PER_PLAYER = 32; // Allow more processes for "swarm" behaviors

type InstructionType = "DAT" | "MOV" | "ADD" | "JMP" | "SPL";

interface Instruction {
	op: InstructionType;
	a: number; // A-field value (simplified relative address)
	b: number; // B-field value (simplified relative address)
	owner: number; // 0 = none, 1 = Player 1 (Red), 2 = Player 2 (Blue)
}

interface Process {
	pc: number; // Program Counter
	player: 1 | 2;
	id: string; // Unique ID for React keys
}

interface CoreState {
	memory: Instruction[];
	processes: Process[];
	cycles: number;
	winner: number | null; // 0 = draw, 1 = p1, 2 = p2
}

type Strategy = "IMP" | "DWARF" | "SWARM";

// --- INITIAL PROGRAMS (Simplified Redcode) ---

// Simple Imp: Copies itself to the next cell continuously.
const IMP_PROG = [{ op: "MOV", a: 0, b: 1 }];

// Dwarf: Bombs memory at interval 4, moves slowly.
const DWARF_PROG = [
	{ op: "ADD", a: 4, b: 3 },
	{ op: "MOV", a: 2, b: 2 }, // Simplified: Treats 'b' as destination relative to current PC
	{ op: "JMP", a: -2, b: 0 },
	{ op: "DAT", a: 0, b: 0 }, // Bomb payload
];

// Evolved Swarm: Uses SPL to create a dense multi-process attack (mimics DRQ complex behaviors).
// Splits, jumps ahead, and leaves DAT bombs behind.
const SWARM_PROG = [
	{ op: "SPL", a: 3, b: 0 }, // Fork new process ahead
	{ op: "MOV", a: 3, b: 5 }, // Copy bomb ahead
	{ op: "ADD", a: 5, b: 1 }, // Modify the MOV instruction's B field so next copy is further
	{ op: "JMP", a: -2, b: 0 }, // Loop back
	{ op: "DAT", a: 0, b: 0 }, // The Bomb
];

const DigitalRedQueenSimulation: React.FC = () => {
	const [p1Strategy, setP1Strategy] = useState<Strategy>("SWARM");
	const [p2Strategy, setP2Strategy] = useState<Strategy>("DWARF");

	const getProgram = (strat: Strategy) => {
		switch (strat) {
			case "IMP":
				return IMP_PROG;
			case "DWARF":
				return DWARF_PROG;
			case "SWARM":
				return SWARM_PROG;
			default:
				return IMP_PROG;
		}
	};

	// --- ENGINE HELPERS ---
	const initCore = useCallback((): CoreState => {
		const memory: Instruction[] = Array(CORE_SIZE)
			.fill(null)
			.map(() => ({
				op: "DAT",
				a: 0,
				b: 0,
				owner: 0,
			}));

		const processes: Process[] = [];

		// Load P1 (Red) at index 0
		const prog1 = getProgram(p1Strategy);
		prog1.forEach((instr, i) => {
			memory[i] = { ...instr, owner: 1 } as Instruction;
		});
		processes.push({ pc: 0, player: 1, id: `p1_init` });

		// Load P2 (Blue) at opposite side
		const offset = Math.floor(CORE_SIZE / 2);
		const prog2 = getProgram(p2Strategy);
		prog2.forEach((instr, i) => {
			memory[(offset + i) % CORE_SIZE] = { ...instr, owner: 2 } as Instruction;
		});
		processes.push({ pc: offset, player: 2, id: `p2_init` });

		return { memory, processes, cycles: 0, winner: null };
	}, [p1Strategy, p2Strategy]);

	// --- SIMULATION HOOK ---
	const { state, isRunning, start, stop, reset } = useSimulation<CoreState>({
		initialState: initCore(),
		tickRate: 50, // Faster tick rate for more chaos
		onTick: (prevState) => {
			if (prevState.winner !== null || prevState.cycles >= MAX_CYCLES) {
				stop();
				return prevState;
			}

			let nextMem = [...prevState.memory];
			let activeProcs = [...prevState.processes];
			let nextGenProcs: Process[] = [];

			// Process execution loop (Simplified: process all active once per tick)
			for (const proc of activeProcs) {
				const playerCount =
					activeProcs.filter((p) => p.player === proc.player).length +
					nextGenProcs.filter((p) => p.player === proc.player).length;

				const currIdx = (proc.pc + CORE_SIZE) % CORE_SIZE;
				const instr = nextMem[currIdx];
				let newPc = currIdx + 1;
				let processSurvived = true;

				// Simplified Execution Logic (Note: Real Redcode addressing is much more complex)
				const getAddr = (rel: number) =>
					(currIdx + rel + CORE_SIZE) % CORE_SIZE;

				switch (instr.op) {
					case "DAT":
						processSurvived = false; // Crash
						break;
					case "MOV":
						// Copy instruction at A to B (relative addressing)
						nextMem[getAddr(instr.b)] = {
							...nextMem[getAddr(instr.a)],
							owner: proc.player,
						};
						break;
					case "ADD":
						// Add A-field of source to A-field of dest (self-modification)
						const target = nextMem[getAddr(instr.b)];
						nextMem[getAddr(instr.b)] = {
							...target,
							a: target.a + instr.a,
							owner: proc.player,
						};
						break;
					case "JMP":
						newPc = getAddr(instr.a);
						break;
					case "SPL":
						// Split execution if under process limit
						if (playerCount < MAX_PROCESSES_PER_PLAYER) {
							nextGenProcs.push({
								pc: getAddr(instr.a),
								player: proc.player,
								id: `${proc.id}_spl_${prevState.cycles}`,
							});
						}
						// Current process continues to next instruction
						break;
				}

				if (processSurvived) {
					nextGenProcs.push({ ...proc, pc: newPc });
				}
			}

			// Win condition check
			const p1Alive = nextGenProcs.some((p) => p.player === 1);
			const p2Alive = nextGenProcs.some((p) => p.player === 2);
			let winner = null;
			if (!p1Alive && !p2Alive)
				winner = 0; // Draw by simultaneous crash
			else if (!p1Alive) winner = 2;
			else if (!p2Alive) winner = 1;

			return {
				memory: nextMem,
				processes: nextGenProcs,
				cycles: prevState.cycles + 1,
				winner,
			};
		},
	});

	// Reset when strategies change
	useEffect(() => {
		if (!isRunning) reset();
	}, [p1Strategy, p2Strategy]);

	// --- RENDER HELPERS ---
	const getCellData = (i: number) => {
		const instr = state.memory[i];
		// Find processes currently at this instruction
		const procsHere = state.processes.filter(
			(p) => (p.pc + CORE_SIZE) % CORE_SIZE === i,
		);
		const p1Proc = procsHere.find((p) => p.player === 1);
		const p2Proc = procsHere.find((p) => p.player === 2);

		let baseClasses =
			"aspect-square rounded-sm border transition-colors duration-75 flex items-center justify-center relative ";

		// Memory Ownership Coloring
		if (instr.owner === 1) baseClasses += "bg-red-950/50 border-red-900/50 ";
		else if (instr.owner === 2)
			baseClasses += "bg-blue-950/50 border-blue-900/50 ";
		else if (instr.op === "DAT")
			baseClasses += "bg-zinc-900/50 border-zinc-800 ";
		else baseClasses += "bg-zinc-800/50 border-zinc-700 ";

		// Process Execution Coloring (Overwrites memory color)
		if (p1Proc && p2Proc)
			baseClasses =
				baseClasses.replace(/bg-[\w-\/]+/, "bg-purple-600 ") +
				"ring-1 ring-white";
		else if (p1Proc)
			baseClasses =
				baseClasses.replace(/bg-[\w-\/]+/, "bg-red-600 ") + "ring-1 ring-white";
		else if (p2Proc)
			baseClasses =
				baseClasses.replace(/bg-[\w-\/]+/, "bg-blue-600 ") +
				"ring-1 ring-white";

		return { classes: baseClasses, instr, p1Proc, p2Proc };
	};

	return (
		<div className="w-full max-w-4xl mx-auto p-4 space-y-6">
			<SchematicCard title="DIGITAL RED QUEEN: EVOLUTIONARY ARENA">
				<div className="space-y-4">
					{/* Controls & Legend */}
					<div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-zinc-900/50 p-4 rounded-lg border border-zinc-800">
						{/* Strategy Selectors */}
						<div className="space-y-2 col-span-1">
							<div className="flex justify-between items-center">
								<span className="text-xs text-red-400 font-mono font-bold flex items-center gap-1">
									<ChevronRight size={12} /> P1 (RED)
								</span>
								<select
									value={p1Strategy}
									onChange={(e) => setP1Strategy(e.target.value as Strategy)}
									className="bg-zinc-800 border border-zinc-700 text-xs rounded px-2 py-1 font-mono text-red-200 outline-none focus:border-red-500"
								>
									<option value="IMP">Basic Imp (Replicator)</option>
									<option value="DWARF">Basic Dwarf (Bomber)</option>
									<option value="SWARM">DRQ Evolved Swarm</option>
								</select>
							</div>
							<div className="flex justify-between items-center">
								<span className="text-xs text-blue-400 font-mono font-bold flex items-center gap-1">
									<ChevronRight size={12} /> P2 (BLUE)
								</span>
								<select
									value={p2Strategy}
									onChange={(e) => setP2Strategy(e.target.value as Strategy)}
									className="bg-zinc-800 border border-zinc-700 text-xs rounded px-2 py-1 font-mono text-blue-200 outline-none focus:border-blue-500"
								>
									<option value="IMP">Basic Imp (Replicator)</option>
									<option value="DWARF">Basic Dwarf (Bomber)</option>
									<option value="SWARM">DRQ Evolved Swarm</option>
								</select>
							</div>
						</div>

						{/* Playback Controls */}
						<div className="flex flex-col items-center justify-center gap-2 col-span-1">
							<div className="flex items-center gap-2">
								<SchematicButton
									onClick={isRunning ? stop : start}
									icon={isRunning ? Pause : Play}
									label={isRunning ? "PAUSE" : "BATTLE"}
									variant={isRunning ? "default" : "accent"}
								/>
								<SchematicButton
									onClick={reset}
									icon={RotateCcw}
									label="RESET"
									variant="outline"
								/>
							</div>
							<div className="font-mono text-xs text-zinc-500">
								CYCLE:{" "}
								<span
									className={`font-bold ${isRunning ? "text-emerald-400" : "text-white"}`}
								>
									{state.cycles}
								</span>{" "}
								/ {MAX_CYCLES}
							</div>
						</div>

						{/* Stats / Legend */}
						<div className="text-xs text-zinc-400 font-mono flex flex-col justify-center gap-1 col-span-1">
							<div className="flex justify-between">
								<span>P1 Procs:</span>{" "}
								<span className="text-red-400">
									{state.processes.filter((p) => p.player === 1).length}
								</span>
							</div>
							<div className="flex justify-between">
								<span>P2 Procs:</span>{" "}
								<span className="text-blue-400">
									{state.processes.filter((p) => p.player === 2).length}
								</span>
							</div>
							<div className="border-t border-zinc-800 mt-1 pt-1 flex flex-wrap gap-2 text-[10px]">
								<span className="flex items-center gap-1">
									<ChevronRight size={10} className="text-red-500" /> P1 Exec
								</span>
								<span className="flex items-center gap-1">
									<ChevronRight size={10} className="text-blue-500" /> P2 Exec
								</span>
								<span className="flex items-center gap-1">
									<Disc size={10} className="text-zinc-600" /> DAT (Bomb)
								</span>
							</div>
						</div>
					</div>

					{/* The Core Grid */}
					<div className="relative bg-black rounded-lg overflow-hidden border border-zinc-800 shadow-[inset_0_0_20px_rgba(0,0,0,0.8)] p-1">
						{/* Winner Overlay */}
						{state.winner !== null && (
							<div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70 backdrop-blur-sm">
								<div className="text-3xl font-bold font-mono flex flex-col items-center gap-4 animate-in zoom-in duration-200">
									{state.winner === 1 && (
										<span className="text-red-500 drop-shadow-[0_0_10px_rgba(239,68,68,0.8)]">
											RED DOMINANCE
										</span>
									)}
									{state.winner === 2 && (
										<span className="text-blue-500 drop-shadow-[0_0_10px_rgba(59,130,246,0.8)]">
											BLUE DOMINANCE
										</span>
									)}
									{state.winner === 0 && (
										<span className="text-zinc-400">MUTUAL DESTRUCTION</span>
									)}
								</div>
							</div>
						)}

						{/* Grid Rendering */}
						<div className="grid grid-cols-16 gap-px">
							{state.memory.map((_, i) => {
								const { classes, instr, p1Proc, p2Proc } = getCellData(i);
								return (
									<div
										key={i}
										className={classes}
										title={`@${i}: ${instr.op} ${instr.a}, ${instr.b}`}
									>
										{/* Render Instruction Indicators */}
										{instr.op === "DAT" && (
											<Disc size={8} className="text-zinc-600/80" />
										)}
										{instr.op === "SPL" && (
											<Crosshair size={8} className="text-zinc-500/80" />
										)}

										{/* Render Process Pointers (Arrows) */}
										{p1Proc && (
											<ChevronRight
												size={14}
												className="absolute text-white z-10 animate-pulse"
											/>
										)}
										{p2Proc && (
											<ChevronRight
												size={14}
												className="absolute text-white z-10 animate-pulse"
											/>
										)}
									</div>
								);
							})}
						</div>
					</div>

					{/* Explanation Section */}
					<div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-zinc-800">
						<div>
							<h4 className="text-sm font-bold text-zinc-300 mb-2 font-mono">
								STRATEGY INTELLIGENCE
							</h4>
							<ul className="space-y-2 text-xs text-zinc-400">
								<li>
									<strong className="text-emerald-400">IMP:</strong> A simple
									virus. It copies itself forward blindly. Watch for rapid
									single-file color changes.
								</li>
								<li>
									<strong className="text-purple-400">DWARF:</strong> A bomber.
									It moves slowly but leaves deadly{" "}
									<span className="text-zinc-500">DAT</span> commands behind.
									Use against Imps.
								</li>
								<li>
									<strong className="text-orange-400">SWARM:</strong> The DRQ
									meta. It splits into multiple process threads to overwhelm the
									opponent while bombing.
								</li>
							</ul>
						</div>
						<div>
							<h4 className="text-sm font-bold text-zinc-300 mb-2 font-mono">
								VISUAL DECODER
							</h4>
							<ul className="space-y-2 text-xs text-zinc-400">
								<li className="flex items-center gap-2">
									<ChevronRight size={12} className="text-white" />{" "}
									<span className="text-white">White Arrows</span> are active
									process threads. If they hit a bomb, they die.
								</li>
								<li className="flex items-center gap-2">
									<Disc size={12} className="text-zinc-600" />{" "}
									<span className="text-zinc-500">Dark Dots</span> are Data
									Bombs. Hitting these crashes a thread.
								</li>
								<li className="flex items-center gap-2">
									<div className="w-3 h-3 bg-red-900/50 border border-red-800 rounded-sm"></div>{" "}
									<span className="text-red-400">Background Color</span>{" "}
									indicates code ownership. Red owns this memory block.
								</li>
							</ul>
						</div>
					</div>

					{/* Note about simplifications */}
					<div className="text-[10px] font-mono text-zinc-500 text-center">
						*Simulation uses a simplified Redcode instruction set for
						visualization purposes. Real DRQ warriors utilize complex addressing
						modes (`#`, `@`, {"<"}, {">"}) not shown here.
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default DigitalRedQueenSimulation;
