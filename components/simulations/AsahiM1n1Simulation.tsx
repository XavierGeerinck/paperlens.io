import React, { useRef, useEffect } from "react";
import {
	Play,
	RotateCcw,
	Cpu,
	Terminal,
	Pause,
	Zap,
	Activity,
	ShieldCheck,
} from "lucide-react";
import { useSimulation } from "../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

interface M1n1State {
	guestPC: string;
	lastAccess: {
		type: "READ" | "WRITE";
		addr: string;
		val: string;
	} | null;
	trapCount: number;
	isTrapped: boolean;
	memoryMapped: boolean;
	bootStage:
		| "SecureROM"
		| "iBoot1"
		| "iBoot2"
		| "m1n1_S1"
		| "m1n1_S2"
		| "Linux";
}

const M1N1_INIT: M1n1State = {
	guestPC: "0x0",
	lastAccess: null,
	trapCount: 0,
	isTrapped: false,
	memoryMapped: false,
	bootStage: "SecureROM",
};

const m1n1Tick = (prev: M1n1State, tick: number): Partial<M1n1State> => {
	// Handle Boot Sequence
	if (tick < 10) return { bootStage: "SecureROM", guestPC: "0x0" };
	if (tick < 20) return { bootStage: "iBoot1", guestPC: "0x1000" };
	if (tick < 30) return { bootStage: "iBoot2", guestPC: "0x2000" };
	if (tick < 40) return { bootStage: "m1n1_S1", guestPC: "0x80000000" };
	if (tick < 50) return { bootStage: "m1n1_S2", guestPC: "0x90000000" };

	// Linux Execution
	if (prev.bootStage !== "Linux") {
		return { bootStage: "Linux", guestPC: "0xffff800008000000" };
	}

	if (prev.isTrapped) {
		return { isTrapped: false };
	}

	const shouldTrap = Math.random() > 0.7;
	if (shouldTrap) {
		const isWrite = Math.random() > 0.5;
		const addr =
			"0x23" +
			Math.floor(Math.random() * 0xffffff)
				.toString(16)
				.padStart(6, "0");
		const val =
			"0x" +
			Math.floor(Math.random() * 0xffffffff)
				.toString(16)
				.padStart(8, "0");

		return {
			isTrapped: true,
			trapCount: prev.trapCount + 1,
			lastAccess: {
				type: isWrite ? "WRITE" : "READ",
				addr,
				val,
			},
			guestPC:
				"0xffff80000" +
				Math.floor(Math.random() * 0xffffff)
					.toString(16)
					.padStart(6, "0"),
		};
	}

	return {
		guestPC:
			"0xffff80000" +
			(parseInt(prev.guestPC.slice(-6), 16) + 4).toString(16).padStart(6, "0"),
	};
};

const m1n1Log = (state: M1n1State, tick: number) => {
	if (tick === 10) return "[BOOT] SecureROM -> iBoot1";
	if (tick === 20) return "[BOOT] iBoot1 -> iBoot2";
	if (tick === 30) return "[BOOT] iBoot2 -> m1n1 Stage 1";
	if (tick === 40) return "[BOOT] m1n1 Stage 1 -> m1n1 Stage 2";
	if (tick === 50) return "[BOOT] m1n1 Stage 2 -> Linux Kernel";

	if (state.isTrapped && state.lastAccess) {
		return `[m1n1] TRAP: ${state.lastAccess.type} @ ${state.lastAccess.addr} (Val: ${state.lastAccess.val}) from PC: ${state.guestPC}`;
	}
	return null;
};

const AsahiM1n1Simulation: React.FC = () => {
	const logsContainerRef = useRef<HTMLDivElement>(null);
	const { isRunning, state, logs, epoch, start, stop, reset } =
		useSimulation<M1n1State>({
			initialState: M1N1_INIT,
			onTick: m1n1Tick,
			onLog: m1n1Log,
			tickRate: 300,
		});

	useEffect(() => {
		if (logsContainerRef.current) {
			logsContainerRef.current.scrollTop =
				logsContainerRef.current.scrollHeight;
		}
	}, [logs]);

	return (
		<div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-500">
			<div className="lg:col-span-2 space-y-6">
				<SchematicCard title="HYPERVISOR_CONTROL_EL2">
					<div className="flex items-center justify-between mb-8">
						<div className="flex items-center gap-2 text-orange-400 font-mono text-sm">
							<ShieldCheck className="w-4 h-4" />
							<span>m1n1_RUNTIME_V2.4</span>
						</div>
						<div className="flex gap-2">
							{!isRunning ? (
								<SchematicButton onClick={start}>
									<Play className="w-3 h-3 inline mr-2" /> BOOT_GUEST
								</SchematicButton>
							) : (
								<SchematicButton onClick={stop}>
									<Pause className="w-3 h-3 inline mr-2" /> HALT_GUEST
								</SchematicButton>
							)}
							<button
								onClick={reset}
								className="p-3 border border-zinc-700 text-zinc-400 hover:text-white transition-colors"
							>
								<RotateCcw className="w-4 h-4" />
							</button>
						</div>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
						{/* Guest State */}
						<div className="bg-zinc-950 border border-zinc-800 p-4 relative overflow-hidden">
							<div className="text-[10px] font-mono text-zinc-500 uppercase mb-4 flex justify-between">
								<span>Guest_OS (EL1)</span>
								<span className={isRunning ? "text-green-500" : "text-red-500"}>
									{isRunning ? state.bootStage : "HALTED"}
								</span>
							</div>
							<div className="space-y-3">
								<div className="flex justify-between items-center">
									<span className="text-xs font-mono text-zinc-400">PC:</span>
									<span className="text-xs font-mono text-zinc-200">
										{state.guestPC}
									</span>
								</div>
								<div className="h-1 w-full bg-zinc-900 rounded-full overflow-hidden">
									<div
										className="h-full bg-orange-500 transition-all duration-100"
										style={{
											width: isRunning ? "100%" : "0%",
											opacity: isRunning ? 0.5 : 0,
										}}
									/>
								</div>
							</div>
							{state.isTrapped && (
								<div className="absolute inset-0 bg-orange-500/10 flex items-center justify-center animate-pulse">
									<Zap className="w-8 h-8 text-orange-500" />
								</div>
							)}
						</div>

						{/* MMIO Trap Info */}
						<div className="bg-zinc-950 border border-zinc-800 p-4">
							<div className="text-[10px] font-mono text-zinc-500 uppercase mb-4">
								<span>MMIO_TRAP_STATUS</span>
							</div>
							{state.lastAccess ? (
								<div className="space-y-2">
									<div className="flex justify-between text-xs font-mono">
										<span className="text-zinc-500">TYPE:</span>
										<span
											className={
												state.lastAccess.type === "WRITE"
													? "text-red-400"
													: "text-blue-400"
											}
										>
											{state.lastAccess.type}
										</span>
									</div>
									<div className="flex justify-between text-xs font-mono">
										<span className="text-zinc-500">ADDR:</span>
										<span className="text-zinc-200">
											{state.lastAccess.addr}
										</span>
									</div>
									<div className="flex justify-between text-xs font-mono">
										<span className="text-zinc-500">VAL:</span>
										<span className="text-zinc-200">
											{state.lastAccess.val}
										</span>
									</div>
								</div>
							) : (
								<div className="h-full flex items-center justify-center text-zinc-700 font-mono text-[10px]">
									WAITING_FOR_ACCESS...
								</div>
							)}
						</div>
					</div>

					<div className="mt-6 flex items-center gap-4 text-xs font-mono text-zinc-500 border-t border-zinc-800 pt-4">
						<div className="flex items-center gap-2">
							<Activity className="w-3 h-3 text-orange-500" />
							TRAPS_CAPTURED: {state.trapCount}
						</div>
						<div className="ml-auto">UPTIME: {epoch} ticks</div>
					</div>
				</SchematicCard>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<SchematicCard
						title="PYTHON_PROXY_HV.PY"
						className="font-mono text-[10px]"
					>
						<div className="text-zinc-600 mb-2"># Live MMIO Tracing Script</div>
						<div className="text-blue-400">def on_mmio_trap(event):</div>
						<div className="text-zinc-400 pl-2">
							if event.addr.startswith("0x23"):
						</div>
						<div className="text-zinc-300 pl-4">
							print(f"Found Hardware Register!")
						</div>
						<div className="text-zinc-300 pl-4">
							log_access(event.pc, event.val)
						</div>
						<div className="text-orange-500 mt-2 animate-pulse">
							{state.isTrapped ? ">>> TRAP_EVENT_RECEIVED" : ">>> LISTENING..."}
						</div>
					</SchematicCard>

					<SchematicCard title="HARDWARE_REGISTERS">
						<div className="space-y-2">
							<div className="flex items-center gap-2">
								<div
									className={`w-2 h-2 rounded-full ${state.isTrapped && state.lastAccess?.addr.includes("23") ? "bg-orange-500 animate-ping" : "bg-zinc-800"}`}
								/>
								<span className="text-[10px] font-mono text-zinc-400">
									UART_CTRL (0x235e0000)
								</span>
							</div>
							<div className="flex items-center gap-2">
								<div
									className={`w-2 h-2 rounded-full ${state.isTrapped && state.lastAccess?.addr.includes("24") ? "bg-orange-500 animate-ping" : "bg-zinc-800"}`}
								/>
								<span className="text-[10px] font-mono text-zinc-400">
									GPIO_CFG (0x23c00000)
								</span>
							</div>
							<div className="flex items-center gap-2">
								<div
									className={`w-2 h-2 rounded-full ${state.isTrapped && state.lastAccess?.addr.includes("25") ? "bg-orange-500 animate-ping" : "bg-zinc-800"}`}
								/>
								<span className="text-[10px] font-mono text-zinc-400">
									AIC_EVENT (0x23b10000)
								</span>
							</div>
						</div>
					</SchematicCard>
				</div>
			</div>

			<div className="lg:col-span-1">
				<SchematicCard
					title="SERIAL_CONSOLE"
					className="h-[500px] flex flex-col"
				>
					<div
						ref={logsContainerRef}
						className="flex-grow overflow-y-auto space-y-1 text-zinc-500 font-mono text-[10px] scrollbar-hide"
					>
						{logs.map((log, i) => (
							<div
								key={i}
								className="break-all border-l-2 border-orange-900/30 pl-2"
							>
								<span className="text-zinc-700 mr-2">
									{new Date().toLocaleTimeString().split(" ")[0]}
								</span>
								{log}
							</div>
						))}
						{logs.length === 0 && (
							<div className="text-zinc-800 italic">
								Waiting for guest boot...
							</div>
						)}
					</div>
				</SchematicCard>
			</div>
		</div>
	);
};

export default AsahiM1n1Simulation;
