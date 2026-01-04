import React from "react";
import {
	Cpu,
	ShieldCheck,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface M1n1State {
	bootStage: string;
	trapCount: number;
	isTrapped: boolean;
	lastAddr: string;
	lastVal: string;
}

const AsahiM1n1Simulation: React.FC = () => {
	const { isRunning, state, start, stop, reset } = useSimulation({
		initialState: {
			bootStage: "SecureROM",
			trapCount: 0,
			isTrapped: false,
			lastAddr: "0x0",
			lastVal: "0x0",
		},
		onTick: (prev, tick) => {
			const stages = ["SecureROM", "iBoot1", "iBoot2", "m1n1", "Linux"];
			// Speed up boot: Change stage every 4 ticks instead of 10
			const stageIdx = Math.min(Math.floor(tick / 4), stages.length - 1);
			
			// Traps happen during m1n1 and Linux stages
			const canTrap = ["m1n1", "Linux"].includes(stages[stageIdx]);
			const shouldTrap = canTrap && Math.random() > 0.6;
			
			return {
				...prev,
				bootStage: stages[stageIdx],
				isTrapped: shouldTrap,
				trapCount: prev.trapCount + (shouldTrap ? 1 : 0),
				lastAddr: shouldTrap ? `0x23${Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, "0")}` : prev.lastAddr,
				lastVal: shouldTrap ? `0x${Math.floor(Math.random() * 0xffffffff).toString(16).padStart(8, "0")}` : prev.lastVal,
			};
		},
		tickRate: 400,
	});

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="HYPERVISOR_BOOT_COMPARISON">
				<div className="grid grid-cols-1 md:grid-cols-2 gap-8 py-4">
					{/* LEFT: STANDARD BOOT */}
					<div className="flex flex-col gap-4 border-r border-slate-800 pr-6 opacity-80">
						<div className="flex items-center gap-2 text-slate-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<Cpu size={16} /> Standard Boot
						</div>

						<div className={`p-4 rounded border min-h-[120px] flex flex-col justify-center items-center gap-2 transition-colors duration-500 ${state.bootStage === "Linux" ? "bg-red-950/30 border-red-500/50" : "bg-slate-950 border-slate-800"}`}>
							<div className="text-[10px] text-slate-500 uppercase">Status</div>
							<div className={`text-xl font-mono ${state.bootStage === "Linux" ? "text-red-500 animate-pulse font-bold" : "text-slate-300"}`}>
								{state.bootStage === "Linux" ? "KERNEL_PANIC" : state.bootStage}
							</div>
							{state.bootStage === "Linux" && (
								<div className="text-[10px] text-red-400 font-mono mt-1">Unknown Register Access</div>
							)}
						</div>

						<div className="p-3 bg-slate-800/30 rounded border border-slate-700 text-[11px] text-slate-400 leading-relaxed">
							Standard kernels run directly on hardware. If they encounter an unknown Apple Silicon register, they <span className="text-red-400 font-bold">Panic</span> because they cannot probe the hardware safely.
						</div>
					</div>

					{/* RIGHT: M1N1 HYPERVISOR */}
					<div className="flex flex-col gap-4 pl-2">
						<div className="flex items-center gap-2 text-orange-400 font-mono text-sm uppercase border-b border-slate-800 pb-2">
							<ShieldCheck size={16} /> m1n1 Hypervisor
						</div>

						<div className="p-4 bg-slate-950 rounded border border-orange-500/30 min-h-[120px] relative overflow-hidden">
							<div className="text-[10px] text-orange-400 uppercase mb-2">Execution Level: EL2</div>
							<div className="flex flex-col gap-1">
								<div className="flex justify-between text-[10px] font-mono">
									<span className="text-slate-500">STAGE:</span>
									<span className="text-orange-300">{state.bootStage}</span>
								</div>
								<div className="flex justify-between text-[10px] font-mono">
									<span className="text-slate-500">TRAPS:</span>
									<span className="text-orange-300">{state.trapCount}</span>
								</div>
								{state.isTrapped && (
									<div className="mt-2 p-2 bg-orange-500/10 border border-orange-500/50 rounded animate-pulse">
										<div className="text-[9px] text-orange-400 font-bold">MMIO TRAP DETECTED</div>
										<div className="text-[8px] font-mono text-orange-200 truncate">{state.lastAddr}</div>
									</div>
								)}
							</div>
						</div>

						<div className="grid grid-cols-2 gap-4">
							<div className="p-2 bg-slate-950 rounded border border-slate-800">
								<div className="text-[9px] text-slate-500 uppercase">Uptime</div>
								<div className="text-lg font-mono font-bold text-slate-300">
									{state.bootStage === "Linux" ? "LIVE" : "BOOTING"}
								</div>
							</div>
							<div className="p-2 bg-slate-950 rounded border border-slate-800">
								<div className="text-[9px] text-slate-500 uppercase">Proxy</div>
								<div className="text-lg font-mono font-bold text-emerald-400">
									ACTIVE
								</div>
							</div>
						</div>
					</div>
				</div>

				{/* Explanation Notes */}
				<div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4 border-t border-slate-800 pt-6">
					<div className="p-3 bg-orange-900/10 border border-orange-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-orange-400 uppercase mb-1">What to watch for: MMIO Trapping</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							m1n1 runs at <span className="text-orange-400 font-bold">EL2</span>, allowing it to intercept (trap) every hardware access the Linux kernel makes. This is how developers reverse-engineer Apple's proprietary hardware.
						</p>
					</div>
					<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
						<h4 className="text-[10px] font-bold text-blue-400 uppercase mb-1">What to watch for: Python Proxy</h4>
						<p className="text-[11px] text-slate-400 leading-relaxed">
							The trapped events are sent over USB to a <span className="text-blue-400 font-bold">Python script</span>. This allows for rapid prototyping of drivers without rebooting the machine for every change.
						</p>
					</div>
				</div>

				<div className="flex gap-4 mt-6 border-t border-slate-800 pt-4">
					<SchematicButton onClick={isRunning ? stop : start}>
						{isRunning ? "HALT_SYSTEM" : "BOOT_SYSTEM"}
					</SchematicButton>
					<SchematicButton onClick={reset} variant="secondary">
						RESET
					</SchematicButton>
				</div>
			</SchematicCard>
		</div>
	);
};

export default AsahiM1n1Simulation;
