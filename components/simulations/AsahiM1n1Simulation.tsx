import React, { useState, useEffect } from "react";
import {
	Cpu,
	ShieldCheck,
	Terminal,
	Zap,
	RefreshCw,
	Search,
	CheckCircle2,
	AlertTriangle,
	Play,
	RotateCcw
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

// Hardware blocks to discover
const HARDWARE_MAP = [
	{ id: "uart", name: "SAMSUNG_UART", addr: "0x23520000", type: "Serial" },
	{ id: "aic", name: "APPLE_AIC", addr: "0x23b10000", type: "IRQ" },
	{ id: "gpio", name: "GPIO_CTRL", addr: "0x23c00000", type: "IO" },
	{ id: "dart", name: "DART_IOMMU", addr: "0x23500000", type: "MMU" },
	{ id: "pci", name: "PCIE_BRIDGE", addr: "0x30000000", type: "Bus" },
	{ id: "disp", name: "DISP_CTRL", addr: "0x20000000", type: "Video" },
];

const AsahiM1n1Simulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			// Legacy (Left) State
			legacyPhase: "CODING", // CODING -> COMPILING -> BOOTING -> CRASH
			legacyProgress: 0,
			legacyIterations: 0,
			legacyLog: ["Initializing kernel build..."],

			// m1n1 (Right) State
			m1n1Phase: "IDLE",
			m1n1Iterations: 0,
			m1n1Log: [">>> import m1n1.proxy as p", ">>> u = p.UART(0x23520000)"],
			discoveredBlocks: [] as string[], // IDs of discovered blocks
		},
		onTick: (prev, tick) => {
			let next = { ...prev };

			// --- LEGACY CYCLE (Slow, Painful) ---
			if (prev.legacyPhase === "CODING") {
				if (tick % 10 === 0) {
					next.legacyPhase = "COMPILING";
					next.legacyLog = ["Compiling kernel drivers...", ...prev.legacyLog.slice(0, 4)];
				}
			} else if (prev.legacyPhase === "COMPILING") {
				if (tick % 15 === 0) {
					next.legacyPhase = "BOOTING";
					next.legacyProgress = 0;
					next.legacyLog = ["Linking image...", "Starting boot sequence...", ...prev.legacyLog.slice(0, 4)];
				}
			} else if (prev.legacyPhase === "BOOTING") {
				next.legacyProgress = Math.min(prev.legacyProgress + 5, 100);
				if (next.legacyProgress >= 100) {
					next.legacyPhase = "CRASH";
					next.legacyLog = ["KERNEL PANIC: Sync Abort", "PC: 0xffff0000...", ...prev.legacyLog.slice(0, 4)];
				}
			} else if (prev.legacyPhase === "CRASH") {
				if (tick % 20 === 0) {
					next.legacyPhase = "CODING";
					next.legacyIterations += 1;
					next.legacyLog = ["Rebooting system...", ...prev.legacyLog.slice(0, 4)];
				}
			}

			// --- M1N1 CYCLE (Fast, Interactive) ---
			// Every 3 ticks, we do something useful
			if (tick % 3 === 0) {
				next.m1n1Phase = "PROBING";
				next.m1n1Iterations += 1;
				
				// Pick a random block to "discover" or "probe"
				const target = HARDWARE_MAP[Math.floor(Math.random() * HARDWARE_MAP.length)];
				const isNew = !prev.discoveredBlocks.includes(target.id);
				
				if (isNew && Math.random() > 0.3) {
					next.discoveredBlocks = [...prev.discoveredBlocks, target.id];
					next.m1n1Log = [`>>> probe(${target.addr})`, `Found: ${target.name}`, ...prev.m1n1Log.slice(0, 5)];
				} else {
					next.m1n1Log = [`>>> read32(${target.addr})`, `0x${Math.floor(Math.random()*0xffffffff).toString(16)}`, ...prev.m1n1Log.slice(0, 5)];
				}
			} else {
				next.m1n1Phase = "IDLE";
			}

			return next;
		},
		tickRate: 100, // Fast ticks for the REPL feel
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="REVERSE_ENGINEERING_WORKFLOW">
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-8 py-4">
					
					{/* LEFT: TRADITIONAL APPROACH */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-90">
						<div className="flex items-center justify-between border-b border-slate-800 pb-2">
							<div className="flex items-center gap-2 text-slate-400 font-mono text-sm uppercase">
								<RefreshCw size={16} /> Traditional Cycle
							</div>
							<div className="text-xs font-mono text-slate-500">
								ITERATIONS: {state.legacyIterations}
							</div>
						</div>

						{/* Status Display */}
						<div className={`h-32 rounded border flex flex-col items-center justify-center gap-2 transition-all duration-300 ${
							state.legacyPhase === "CRASH" ? "bg-red-950/30 border-red-500/50" : 
							state.legacyPhase === "BOOTING" ? "bg-slate-900 border-blue-500/30" :
							"bg-slate-950 border-slate-800"
						}`}>
							{state.legacyPhase === "CODING" && <Terminal className="text-slate-600" />}
							{state.legacyPhase === "COMPILING" && <RefreshCw className="text-blue-400 animate-spin" />}
							{state.legacyPhase === "BOOTING" && (
								<div className="w-full px-8">
									<div className="text-xs text-center mb-2 text-blue-400">BOOTING KERNEL...</div>
									<div className="h-1 bg-slate-800 rounded-full overflow-hidden">
										<div className="h-full bg-blue-500 transition-all duration-100" style={{ width: `${state.legacyProgress}%` }} />
									</div>
								</div>
							)}
							{state.legacyPhase === "CRASH" && (
								<>
									<Zap className="text-red-500 animate-bounce" />
									<div className="text-red-500 font-bold font-mono">KERNEL PANIC</div>
								</>
							)}
						</div>

						{/* Log Output */}
						<div className="h-32 bg-black rounded border border-slate-800 p-2 font-mono text-[10px] text-slate-400 overflow-hidden flex flex-col-reverse">
							{state.legacyLog.map((line: string, i: number) => (
								<div key={i} className={line.includes("PANIC") ? "text-red-400" : ""}>
									{line}
								</div>
							))}
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
							<strong className="text-red-400">The Old Way:</strong> Change code, compile, reboot, crash, repeat. Cycle time: Minutes.
						</div>
					</div>

					{/* RIGHT: M1N1 APPROACH */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center justify-between border-b border-slate-800 pb-2">
							<div className="flex items-center gap-2 text-emerald-400 font-mono text-sm uppercase">
								<ShieldCheck size={16} /> m1n1 + Python
							</div>
							<div className="text-xs font-mono text-emerald-500">
								ITERATIONS: {state.m1n1Iterations}
							</div>
						</div>

						{/* Hardware Map */}
						<div className="h-32 bg-slate-950 rounded border border-emerald-500/30 p-3 grid grid-cols-3 gap-2">
							{HARDWARE_MAP.map((hw) => {
								const isDiscovered = state.discoveredBlocks.includes(hw.id);
								return (
									<div key={hw.id} className={`
										rounded border flex flex-col items-center justify-center text-[9px] font-mono transition-all duration-500
										${isDiscovered 
											? "bg-emerald-900/20 border-emerald-500/50 text-emerald-400 shadow-[0_0_10px_rgba(16,185,129,0.1)]" 
											: "bg-slate-900 border-slate-800 text-slate-700"}
									`}>
										{isDiscovered ? (
											<>
												<div className="font-bold">{hw.name}</div>
												<div className="text-[8px] opacity-70">{hw.addr}</div>
											</>
										) : (
											<div className="text-slate-800">???</div>
										)}
									</div>
								);
							})}
						</div>

						{/* Python REPL */}
						<div className="h-32 bg-slate-950 rounded border border-slate-800 p-2 font-mono text-[10px] text-emerald-300/80 overflow-hidden flex flex-col-reverse shadow-inner">
							{state.m1n1Log.map((line: string, i: number) => (
								<div key={i} className="truncate">
									{line}
								</div>
							))}
						</div>

						<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded border-l-4 border-l-emerald-500 text-[11px] text-slate-400 leading-relaxed">
							<strong className="text-emerald-400">The Asahi Way:</strong> Use Python to probe hardware live via m1n1. No reboots. Cycle time: Milliseconds.
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-slate-800/50 border border-slate-700 rounded-lg">
						<h4 className="text-[10px] font-bold text-slate-300 uppercase mb-1">Why is this hard?</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							Apple Silicon is undocumented. Traditional driver development requires guessing how hardware works, writing a driver, and rebooting to test. If you guess wrong, the kernel panics.
						</p>
					</div>
					<div className="p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1">The m1n1 Revolution</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							m1n1 acts as a bridge, allowing a Python script on another computer to "touch" the hardware directly. Developers can map the entire chip interactively before writing a single line of C code.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start} icon={isRunning ? <Zap size={14}/> : <Play size={14}/>}>
						{isRunning ? "PAUSE_SIMULATION" : "START_COMPARISON"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary" icon={<RotateCcw size={14}/>}>
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default AsahiM1n1Simulation;
