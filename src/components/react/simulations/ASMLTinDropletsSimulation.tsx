import React, { useRef, useEffect, useState, useMemo } from "react";
import {
	Play,
	Pause,
	RotateCcw,
	Zap,
	Activity,
	AlertTriangle,
	Crosshair,
	Info,
} from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";
import * as D3 from "d3";

// Constants
const NOZZLE_DIAMETER_MICRONS = 10;
const CANVAS_WIDTH = 600;
const CANVAS_HEIGHT = 400;
const DROPLET_STREAM_X = CANVAS_WIDTH / 2;
const TARGET_ZONE_Y = 300; // where the laser hits

interface Droplet {
	id: number;
	y: number; // position
	r: number; // radius
	v: number; // velocity
	isSatellite: boolean;
	hit: boolean; // if laser hit it
	vaporized: boolean; // if it turned to plasma
}

interface SimulationState {
	droplets: Droplet[];
	time: number;
	lastGenTime: number;
	stats: {
		totalDroplets: number;
		perfectHits: number;
		satellites: number;
		stabilityScore: number; // 0-100%
	};
}

const ASMLTinDropletsSimulation: React.FC = () => {
	// --- Simulation Parameters (User Controllable) ---
	const [pressure, setPressure] = useState<number>(100); // bar, affects velocity
	const [frequency, setFrequency] = useState<number>(50); // kHz
	const [driveAmplitude, setDriveAmplitude] = useState<number>(50); // % - 0 means natural breakup
	const [laserEnabled, setLaserEnabled] = useState<boolean>(true);
	const [optimizedWaveform, setOptimizedWaveform] = useState<boolean>(false);

	// Derived physics constants
	const jetVelocity = useMemo(
		() => Math.sqrt(2 * pressure * 1000) * 0.5,
		[pressure],
	); // simplified v ~ sqrt(P)
	const jetRadius = NOZZLE_DIAMETER_MICRONS / 2;
	const rayleighOptimumFreq = useMemo(() => {
		// f_opt = v / (4.51 * 2 * r0) approx
		// ideal wavelength is ~9 * r0
		const optimalWavelength = 9.02 * jetRadius;
		return (jetVelocity / optimalWavelength) * 10; // scaling factor to match readable kHz units
	}, [jetVelocity, jetRadius]);

	const canvasRef = useRef<HTMLCanvasElement>(null);

	// --- Simulation Hook ---
	const { isRunning, state, start, stop, reset } =
		useSimulation<SimulationState>({
			initialState: {
				droplets: [],
				time: 0,
				lastGenTime: 0,
				stats: {
					totalDroplets: 0,
					perfectHits: 0,
					satellites: 0,
					stabilityScore: 100,
				},
			},
			tickRate: 20, // ms
			onTick: (currentState, deltaTime) => {
				const dt = deltaTime / 1000; // seconds
				const nextState = { ...currentState };
				nextState.time += dt;

				// 1. Generate Droplets
				const isDriven = driveAmplitude > 10;

				let effectiveFreq = isDriven ? frequency : rayleighOptimumFreq;

				const timeScale = 0.002;
				const realPeriod = 1 / effectiveFreq;
				const simPeriod = realPeriod / timeScale;

				if (nextState.time - nextState.lastGenTime > simPeriod) {
					nextState.lastGenTime = nextState.time;

					const freqDeviation =
						Math.abs(frequency - rayleighOptimumFreq) / rayleighOptimumFreq;

					let satelliteChance = 0;
					let posJitter = 0;
					let sizeJitter = 0;

					if (!isDriven) {
						// Chaos mode (Rayleigh-Plateau)
						satelliteChance = 0.3;
						posJitter = 20;
						sizeJitter = 0.2;
						effectiveFreq = rayleighOptimumFreq * (0.8 + Math.random() * 0.4);
					} else {
						// Driven
						if (freqDeviation > 0.15 && !optimizedWaveform) {
							satelliteChance = 0.8;
						}
						posJitter = (100 - driveAmplitude) / 5;
						sizeJitter = (100 - driveAmplitude) / 500;
					}

					// Main Droplet
					const mainRadius =
						jetRadius * (1.8 + (Math.random() - 0.5) * sizeJitter);
					const startY = 0 - (Math.random() - 0.5) * posJitter;

					nextState.droplets.push({
						id: Math.random(),
						y: startY,
						r: mainRadius,
						v: jetVelocity * 0.2,
						isSatellite: false,
						hit: false,
						vaporized: false,
					});

					nextState.stats.totalDroplets++;

					// Maybe create satellite
					if (Math.random() < satelliteChance) {
						nextState.droplets.push({
							id: Math.random(),
							y: startY - simPeriod * jetVelocity * 0.1,
							r: jetRadius * 0.5,
							v: jetVelocity * 0.2 * 0.95,
							isSatellite: true,
							hit: false,
							vaporized: false,
						});
						nextState.stats.satellites++;
					}
				}

				// 2. Move Droplets
				nextState.droplets.forEach((d) => {
					if (!d.vaporized) {
						d.y += d.v;
					}
				});

				// 3. Laser Interaction
				if (laserEnabled) {
					const targetZoneStart = TARGET_ZONE_Y - 5;
					const targetZoneEnd = TARGET_ZONE_Y + 5;

					nextState.droplets.forEach((d) => {
						if (
							!d.hit &&
							!d.isSatellite &&
							d.y >= targetZoneStart &&
							d.y <= targetZoneEnd
						) {
							const hitChance = isDriven ? 0.99 : 0.4;
							if (Math.random() < hitChance) {
								d.hit = true;
								d.vaporized = true;
								nextState.stats.perfectHits++;
							}
						}
					});
				}

				// 4. Cleanup
				nextState.droplets = nextState.droplets.filter(
					(d) => d.y < CANVAS_HEIGHT + 50,
				);

				// 5. Update Stability Score
				const total = nextState.stats.totalDroplets || 1;
				const perfectRatio =
					nextState.stats.perfectHits / (total - nextState.stats.satellites);
				nextState.stats.stabilityScore = isDriven
					? optimizedWaveform
						? 99
						: perfectRatio * 100
					: 40;

				return nextState;
			},
		});

	// --- Rendering ---
	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		// Clear with transparent bg (handled by container)
		ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

		// Draw Nozzle
		ctx.fillStyle = "#94a3b8"; // slate-400
		ctx.fillRect(DROPLET_STREAM_X - 15, 0, 30, 40);
		ctx.fillStyle = "#64748b"; // slate-500
		ctx.fillRect(DROPLET_STREAM_X - 10, 40, 20, 10);

		// Draw Target Zone
		ctx.strokeStyle = "rgba(239, 68, 68, 0.4)"; // red-500
		ctx.setLineDash([5, 5]);
		ctx.beginPath();
		ctx.moveTo(0, TARGET_ZONE_Y);
		ctx.lineTo(CANVAS_WIDTH, TARGET_ZONE_Y);
		ctx.stroke();
		ctx.setLineDash([]);

		// Label
		ctx.fillStyle = "rgba(239, 68, 68, 0.6)";
		ctx.font = "10px monospace";
		ctx.fillText("LASER INTERACTION ZONE", 10, TARGET_ZONE_Y - 5);

		// Draw Laser (Visual effect when hit)
		const hits = state.droplets.filter(
			(d) => d.vaporized && d.y < TARGET_ZONE_Y + 20,
		);
		hits.forEach((h) => {
			// Laser Beam
			ctx.strokeStyle = "#d946ef"; // fuchsia-500
			ctx.lineWidth = 2;
			ctx.beginPath();
			ctx.moveTo(CANVAS_WIDTH, TARGET_ZONE_Y);
			ctx.lineTo(DROPLET_STREAM_X, h.y);
			ctx.stroke();

			// Plasma Burst
			const gradient = ctx.createRadialGradient(
				DROPLET_STREAM_X,
				h.y,
				1,
				DROPLET_STREAM_X,
				h.y,
				40,
			);
			gradient.addColorStop(0, "#fff");
			gradient.addColorStop(0.2, "#d946ef");
			gradient.addColorStop(1, "rgba(217, 70, 239, 0)");
			ctx.fillStyle = gradient;
			ctx.beginPath();
			ctx.arc(DROPLET_STREAM_X, h.y, 40, 0, Math.PI * 2);
			ctx.fill();
		});

		// Draw Droplets
		state.droplets.forEach((d) => {
			if (d.vaporized) return;

			ctx.beginPath();
			ctx.arc(DROPLET_STREAM_X, d.y, d.r, 0, Math.PI * 2);

			if (d.isSatellite) {
				ctx.fillStyle = "#475569"; // slate-600
			} else {
				ctx.fillStyle = "#e2e8f0"; // slate-200
			}
			ctx.fill();
			ctx.strokeStyle = "#0f172a"; // slate-900 border
			ctx.lineWidth = 1;
			ctx.stroke();

			// Specular highlight
			ctx.fillStyle = "white";
			ctx.globalAlpha = 0.5;
			ctx.beginPath();
			ctx.arc(
				DROPLET_STREAM_X - d.r * 0.3,
				d.y - d.r * 0.3,
				d.r * 0.3,
				0,
				Math.PI * 2,
			);
			ctx.fill();
			ctx.globalAlpha = 1;
		});
	}, [state]);

	return (
		<SchematicCard title="Active Droplet Generator Control">
			<div className="flex flex-col gap-6">
				{/* Top Stats */}
				<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
					<div className="bg-slate-800/50 p-3 rounded-md border border-slate-700/50">
						<div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">
							Stability
						</div>
						<div
							className={`text-2xl font-mono font-bold ${state.stats.stabilityScore > 90 ? "text-emerald-400" : "text-amber-400"}`}
						>
							{state.stats.stabilityScore.toFixed(0)}%
						</div>
					</div>
					<div className="bg-slate-800/50 p-3 rounded-md border border-slate-700/50">
						<div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">
							Total Drops
						</div>
						<div className="text-2xl font-mono font-bold text-slate-200">
							{state.stats.totalDroplets}
						</div>
					</div>
					<div className="bg-slate-800/50 p-3 rounded-md border border-slate-700/50">
						<div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">
							Satellites
						</div>
						<div
							className={`text-2xl font-mono font-bold ${state.stats.satellites > 5 ? "text-rose-400" : "text-slate-200"}`}
						>
							{state.stats.satellites}
						</div>
					</div>
					<div className="bg-slate-800/50 p-3 rounded-md border border-slate-700/50">
						<div className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-1">
							EUV Shots
						</div>
						<div className="text-2xl font-mono font-bold text-fuchsia-400">
							{state.stats.perfectHits}
						</div>
					</div>
				</div>

				<div className="flex flex-col md:flex-row gap-6">
					{/* Visualizer */}
					<div className="relative border border-slate-700/50 rounded-lg overflow-hidden bg-black/40 shadow-inner flex-grow h-[400px]">
						<canvas
							ref={canvasRef}
							width={CANVAS_WIDTH}
							height={CANVAS_HEIGHT}
							className="w-full h-full object-contain"
						/>

						{/* Overlay Info */}
						<div className="absolute top-4 left-4 text-[10px] text-emerald-500/80 font-mono space-y-1">
							<div>JET VELOCITY: {jetVelocity.toFixed(1)} m/s</div>
							<div>TARGET FREQ: {rayleighOptimumFreq.toFixed(1)} kHz</div>
							<div>ACTUAL FREQ: {frequency.toFixed(1)} kHz</div>
							<div>
								DEVIATION:{" "}
								{Math.abs(frequency - rayleighOptimumFreq).toFixed(1)} kHz
							</div>
						</div>
					</div>

					{/* Controls Panel */}
					<div className="w-full md:w-80 flex flex-col gap-6 bg-slate-800/30 p-4 rounded-lg border border-slate-700/50 backdrop-blur-sm">
						<div className="flex items-center gap-2 mb-2 border-b border-slate-700/50 pb-2">
							<Activity className="w-4 h-4 text-slate-400" />
							<h3 className="font-bold text-sm text-slate-200 uppercase tracking-wide">
								Acoustic Driver
							</h3>
						</div>

						<div className="space-y-6">
							<div>
								<div className="flex justify-between text-[11px] font-mono mb-2">
									<span className="text-slate-400 uppercase">
										Drive Frequency
									</span>
									<span className="text-slate-200">{frequency} kHz</span>
								</div>
								<input
									type="range"
									min="20"
									max="100"
									className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
									value={frequency}
									onChange={(e) => setFrequency(Number(e.target.value))}
								/>
								<div className="text-[10px] text-slate-600 mt-2 flex justify-between font-mono">
									<span>20kHz</span>
									<span className="text-emerald-500 font-bold">
										OPT: {rayleighOptimumFreq.toFixed(0)}
									</span>
									<span>100kHz</span>
								</div>
							</div>

							<div>
								<div className="flex justify-between text-[11px] font-mono mb-2">
									<span className="text-slate-400 uppercase">
										PZT Amplitude
									</span>
									<span className="text-slate-200">{driveAmplitude}%</span>
								</div>
								<input
									type="range"
									min="0"
									max="100"
									className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
									value={driveAmplitude}
									onChange={(e) => setDriveAmplitude(Number(e.target.value))}
								/>
								<div className="text-[10px] text-slate-500 mt-2 font-mono">
									{driveAmplitude < 10
										? "⚠️ NATURAL BREAKUP (CHAOS)"
										: "LOCKED DRIVE"}
								</div>
							</div>

							<div>
								<div className="flex justify-between text-[11px] font-mono mb-2">
									<span className="text-slate-400 uppercase">Pressure</span>
									<span className="text-slate-200">{pressure} bar</span>
								</div>
								<input
									type="range"
									min="50"
									max="200"
									className="w-full accent-blue-500 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer"
									value={pressure}
									onChange={(e) => setPressure(Number(e.target.value))}
								/>
							</div>

							<div className="pt-4 border-t border-slate-700/50 space-y-4">
								<label className="flex items-center justify-between cursor-pointer group">
									<span className="text-[11px] font-bold text-slate-300 uppercase flex items-center gap-2 group-hover:text-fuchsia-400 transition-colors">
										<Zap className="w-3.5 h-3.5" />
										Main Pulse Laser
									</span>
									<input
										type="checkbox"
										checked={laserEnabled}
										onChange={(e) => setLaserEnabled(e.target.checked)}
										className="scale-110 accent-fuchsia-500"
									/>
								</label>

								<label className="flex items-center justify-between cursor-pointer group">
									<span className="text-[11px] font-bold text-slate-300 uppercase flex items-center gap-2 group-hover:text-blue-400 transition-colors">
										<Activity className="w-3.5 h-3.5" />
										Optimized Waveform
									</span>
									<input
										type="checkbox"
										checked={optimizedWaveform}
										onChange={(e) => setOptimizedWaveform(e.target.checked)}
										className="scale-110 accent-blue-500"
									/>
								</label>
							</div>
						</div>

						<div className="mt-auto flex gap-2 pt-4">
							<SchematicButton
								onClick={isRunning ? stop : start}
								active={isRunning}
								icon={isRunning ? Pause : Play}
								label={isRunning ? "PAUSE" : "RUN"}
							/>
							<SchematicButton
								onClick={reset}
								icon={RotateCcw}
								label="RESET"
								variant="secondary"
							/>
						</div>
					</div>
				</div>

				{/* Context Explanation */}
				<div className="bg-blue-950/30 border border-blue-500/20 p-4 rounded-lg text-xs leading-relaxed text-blue-200/80">
					<div className="font-bold font-mono text-blue-400 flex items-center gap-2 mb-2 uppercase tracking-wide">
						<Info className="w-4 h-4" />
						Simulation Insight
					</div>
					The Rayleigh-Plateau instability naturally breaks the jet into random
					sizes. By driving the piezoelectric nozzle at the{" "}
					<strong className="text-blue-200">Rayleigh Optimum Frequency</strong>{" "}
					(approx 9x jet radius wavelength), we force a synchronized breakup.
					Mismatching this frequency creates satellite droplets which ruin EUV
					generation.
				</div>
			</div>
		</SchematicCard>
	);
};

export default ASMLTinDropletsSimulation;
