import React, { useEffect, useMemo, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Undo2 } from "lucide-react";
import { SchematicCard, SchematicButton, DataReadout, TechBadge } from "../SketchElements";

// ---------------------------------------------------------------------------
// Deterministic RNG (mulberry32). Keeps every figure identical on every load.
// ---------------------------------------------------------------------------
function rng(seed: number) {
	let s = seed >>> 0;
	return () => {
		s = (s + 0x6d2b79f5) >>> 0;
		let t = s;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// ---------------------------------------------------------------------------
// A toy of the interface, not of the model.
//
// EchoWM's claim is that one signal — camera intent — is enough to drive an
// enterable world. Two things follow from that, and both are visible here:
//
//   1. Discrete commands ("move forward") and continuous poses are not separate
//      control paths. Both are mapped into a single relative 6-DoF trajectory,
//      and every modality is generated from that one trajectory at once.
//
//   2. That trajectory is metric-scale. Training data comes from sources whose
//      native units disagree wildly, so "forward" would otherwise mean a
//      different distance depending on which dataset an example came from.
//      Dataset-level calibration is what makes the interface mean one thing.
//
// Nothing here generates video or audio. The lanes show *when* each modality is
// produced relative to the trajectory, which is the synchronisation claim.
// ---------------------------------------------------------------------------

type CmdKind = "fwd" | "back" | "left" | "right" | "up" | "down";

const COMMANDS: { kind: CmdKind; label: string }[] = [
	{ kind: "fwd", label: "forward" },
	{ kind: "back", label: "back" },
	{ kind: "left", label: "turn left" },
	{ kind: "right", label: "turn right" },
	{ kind: "up", label: "rise" },
	{ kind: "down", label: "descend" },
];

/** Heterogeneous training sources, with the step size each one natively uses. */
const SOURCES = [
	{ key: "aerial", label: "aerial capture", native: 2.4, color: "var(--purple)" },
	{ key: "indoor", label: "indoor walkthrough", native: 0.45, color: "var(--orange)" },
	{ key: "engine", label: "game engine", native: 1.0, color: "var(--aqua)" },
];

const MODALITIES = [
	{ key: "video", label: "720p video", color: "var(--green)" },
	{ key: "ambient", label: "environmental sound", color: "var(--aqua)" },
	{ key: "music", label: "music", color: "var(--purple)" },
	{ key: "speech", label: "speech", color: "var(--orange)" },
];

const DEFAULT_SEQUENCE: CmdKind[] = ["fwd", "fwd", "right", "fwd", "left", "fwd", "fwd", "left", "fwd", "right", "fwd"];

interface Pose {
	x: number;
	y: number;
	z: number;
	yaw: number;
}

/**
 * Fold a command sequence into a 6-DoF trajectory at one source's native scale.
 * Calibration replaces that native scale with the shared metric one.
 */
function trajectory(seq: CmdKind[], step: number, jitterSeed: number): Pose[] {
	const r = rng(jitterSeed);
	const out: Pose[] = [{ x: 0, y: 0, z: 0, yaw: 0 }];

	for (const kind of seq) {
		const prev = out[out.length - 1];
		// Real capture is never exact; a little per-step noise keeps the paths honest.
		const noise = 1 + (r() - 0.5) * 0.08;
		const d = step * noise;
		const next: Pose = { ...prev };

		if (kind === "fwd" || kind === "back") {
			const sign = kind === "fwd" ? 1 : -1;
			next.x = prev.x + Math.cos(prev.yaw) * d * sign;
			next.z = prev.z + Math.sin(prev.yaw) * d * sign;
		} else if (kind === "left") {
			next.yaw = prev.yaw - Math.PI / 4;
		} else if (kind === "right") {
			next.yaw = prev.yaw + Math.PI / 4;
		} else if (kind === "up") {
			next.y = prev.y + d * 0.6;
		} else {
			next.y = prev.y - d * 0.6;
		}
		out.push(next);
	}
	return out;
}

function pathLength(poses: Pose[]): number {
	let total = 0;
	for (let i = 1; i < poses.length; i++) {
		const a = poses[i - 1];
		const b = poses[i];
		total += Math.hypot(b.x - a.x, b.z - a.z, b.y - a.y);
	}
	return total;
}

// ---------------------------------------------------------------------------

const EchoWMSimulation: React.FC = () => {
	const [seq, setSeq] = useState<CmdKind[]>(DEFAULT_SEQUENCE);
	const [calibrated, setCalibrated] = useState(true);
	const [thirdPerson, setThirdPerson] = useState(false);
	const [playing, setPlaying] = useState(true);
	const [head, setHead] = useState(0);

	const canvasRef = useRef<HTMLCanvasElement>(null);
	const wrapRef = useRef<HTMLDivElement>(null);
	const [width, setWidth] = useState(560);

	const paths = useMemo(
		() =>
			SOURCES.map((s, i) => ({
				...s,
				poses: trajectory(seq, calibrated ? 1.0 : s.native, 0x51ed + i * 977),
			})),
		[seq, calibrated],
	);

	const lengths = paths.map((p) => pathLength(p.poses));
	const spread = lengths.length
		? ((Math.max(...lengths) - Math.min(...lengths)) / Math.max(...lengths)) * 100
		: 0;

	// --- responsive canvas: fill the pane, never overflow it -----------------
	useEffect(() => {
		if (typeof window === "undefined" || !wrapRef.current) return;
		const el = wrapRef.current;
		const measure = () => setWidth(el.clientWidth || 560);
		measure();
		const ro = new ResizeObserver(measure);
		ro.observe(el);
		return () => ro.disconnect();
	}, []);

	useEffect(() => {
		if (typeof window === "undefined") return;
		if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
			setPlaying(false);
			setHead(seq.length);
		}
	}, [seq.length]);

	useEffect(() => {
		if (!playing) return;
		const id = window.setInterval(() => setHead((h) => (h >= seq.length ? 0 : h + 1)), 420);
		return () => window.clearInterval(id);
	}, [playing, seq.length]);

	useEffect(() => setHead(seq.length), [seq]);

	// --- the top-down map ----------------------------------------------------
	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;

		const height = Math.round(Math.min(300, Math.max(200, width * 0.42)));
		const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;
		canvas.width = width * dpr;
		canvas.height = height * dpr;
		canvas.style.width = `${width}px`;
		canvas.style.height = `${height}px`;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

		const css = getComputedStyle(document.documentElement);
		const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
		/** "var(--purple)" -> "#8b6cf2". Canvas needs a resolved colour. */
		const resolve = (token: string, fallback: string) => {
			const m = token.match(/var\((--[\w-]+)\)/);
			return m ? v(m[1], fallback) : token;
		};

		ctx.clearRect(0, 0, width, height);
		ctx.fillStyle = v("--bg0h", "#080a0d");
		ctx.fillRect(0, 0, width, height);

		// Fit every path, so switching calibration visibly changes the extent.
		const all = paths.flatMap((p) => p.poses);
		const xs = all.map((p) => p.x);
		const zs = all.map((p) => p.z);
		const pad = 26;
		const minX = Math.min(...xs);
		const maxX = Math.max(...xs);
		const minZ = Math.min(...zs);
		const maxZ = Math.max(...zs);
		const scale = Math.min(
			(width - pad * 2) / Math.max(maxX - minX, 0.001),
			(height - pad * 2) / Math.max(maxZ - minZ, 0.001),
		);
		const cx = width / 2 - ((minX + maxX) / 2) * scale;
		const cy = height / 2 - ((minZ + maxZ) / 2) * scale;
		const px = (p: Pose) => [p.x * scale + cx, p.z * scale + cy] as const;

		// Metric grid — one line per metre, so scale is readable rather than implied.
		ctx.strokeStyle = v("--bg2", "#22272f");
		ctx.lineWidth = 1;
		const gridStep = Math.max(scale, 12);
		for (let gx = cx % gridStep; gx < width; gx += gridStep) {
			ctx.beginPath();
			ctx.moveTo(gx, 0);
			ctx.lineTo(gx, height);
			ctx.stroke();
		}
		for (let gy = cy % gridStep; gy < height; gy += gridStep) {
			ctx.beginPath();
			ctx.moveTo(0, gy);
			ctx.lineTo(width, gy);
			ctx.stroke();
		}

		const shown = Math.max(1, head);

		paths.forEach((path, idx) => {
			const poses = path.poses.slice(0, shown + 1);
			if (poses.length < 2) return;

			// Nudge each strand sideways so coincident paths stay individually
			// readable; the offset is in screen pixels and never changes the geometry.
			const off = (idx - (paths.length - 1) / 2) * 2.6;

			const stroke = resolve(path.color, "#98a2ae");
			ctx.strokeStyle = stroke;
			ctx.lineWidth = 2.4;
			ctx.lineJoin = "round";
			ctx.lineCap = "round";
			ctx.beginPath();
			poses.forEach((p, i) => {
				const [x, y] = px(p);
				if (i === 0) ctx.moveTo(x, y + off);
				else ctx.lineTo(x, y + off);
			});
			ctx.stroke();

			// The observer, with its heading — this is the camera intent made visible.
			const last = poses[poses.length - 1];
			const [hx0, hy0] = px(last);
			const hx = hx0;
			const hy = hy0 + off;
			ctx.fillStyle = stroke;
			ctx.beginPath();
			ctx.arc(hx, hy, 3.5, 0, Math.PI * 2);
			ctx.fill();

			ctx.strokeStyle = stroke;
			ctx.lineWidth = 1.5;
			ctx.beginPath();
			ctx.moveTo(hx, hy);
			ctx.lineTo(hx + Math.cos(last.yaw) * 14, hy + Math.sin(last.yaw) * 14);
			ctx.stroke();

			// In third person the camera trails the subject rather than being it.
			if (thirdPerson) {
				const ox = hx - Math.cos(last.yaw) * 30;
				const oy = hy - Math.sin(last.yaw) * 30;
				ctx.globalAlpha = 0.5;
				ctx.setLineDash([3, 3]);
				ctx.beginPath();
				ctx.moveTo(ox, oy);
				ctx.lineTo(hx, hy);
				ctx.stroke();
				ctx.setLineDash([]);
				ctx.strokeRect(ox - 3, oy - 3, 6, 6);
				ctx.globalAlpha = 1;
			}
		});

		// Scale bar: one metre, so "metric-scale" is something you can measure.
		ctx.strokeStyle = v("--fg4", "#6b7583");
		ctx.fillStyle = v("--fg4", "#6b7583");
		ctx.lineWidth = 1;
		const barY = height - 14;
		ctx.beginPath();
		ctx.moveTo(14, barY);
		ctx.lineTo(14 + scale, barY);
		ctx.moveTo(14, barY - 3);
		ctx.lineTo(14, barY + 3);
		ctx.moveTo(14 + scale, barY - 3);
		ctx.lineTo(14 + scale, barY + 3);
		ctx.stroke();
		ctx.font = "10px ui-monospace, monospace";
		ctx.fillText("1 m", 18 + scale, barY + 3);
	}, [paths, width, head, calibrated, thirdPerson]);

	const push = (kind: CmdKind) => setSeq((s) => [...s, kind].slice(-16));

	return (
		<div className="flex flex-col gap-4">
			<SchematicCard
				title="ECHOWM · one camera intent, every modality"
				status={`${thirdPerson ? "third" : "first"} person · ${calibrated ? "calibrated" : "raw source units"}`}
			>
				<p className="text-[12.5px] text-ink2 leading-relaxed mb-3">
					Three training sources run the <span className="text-ink">same command sequence</span>. Their native
					units disagree — an aerial clip's “forward” covers far more ground than an indoor walkthrough's. The
					shared 6-DoF trajectory is what makes one command mean one thing.
				</p>

				<div ref={wrapRef} className="w-full">
					<canvas ref={canvasRef} className="block rounded border border-bg2" />
				</div>

				<div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[11px] font-mono">
					{paths.map((p, i) => (
						<span key={p.key} className="flex items-center gap-1.5">
							<span className="inline-block w-2.5 h-2.5 rounded-[1px]" style={{ background: p.color }} />
							<span className="text-mute">{p.label}</span>
							<span className="text-ink1 tabular-nums">{lengths[i].toFixed(1)} m</span>
						</span>
					))}
				</div>
			</SchematicCard>

			<div className="grid gap-4 md:grid-cols-2">
				<SchematicCard title="CAMERA INTENT">
					<span className="text-[11px] font-mono text-mute block mb-1.5">discrete commands</span>
					<div className="flex flex-wrap gap-1.5 mb-4">
						{COMMANDS.map((c) => (
							<SchematicButton key={c.kind} onClick={() => push(c.kind)}>
								{c.label}
							</SchematicButton>
						))}
					</div>

					<div className="flex flex-wrap gap-2 mb-4">
						<SchematicButton active={calibrated} onClick={() => setCalibrated((v) => !v)}>
							dataset calibration {calibrated ? "on" : "off"}
						</SchematicButton>
						<SchematicButton active={thirdPerson} onClick={() => setThirdPerson((v) => !v)}>
							{thirdPerson ? "third person" : "first person"}
						</SchematicButton>
					</div>

					<div className="flex flex-wrap gap-2">
						<SchematicButton onClick={() => setPlaying((p) => !p)} icon={playing ? Pause : Play}>
							{playing ? "pause" : "play"}
						</SchematicButton>
						<SchematicButton onClick={() => setSeq((s) => s.slice(0, -1))} icon={Undo2} disabled={seq.length < 2}>
							undo
						</SchematicButton>
						<SchematicButton
							onClick={() => {
								setSeq(DEFAULT_SEQUENCE);
								setCalibrated(true);
								setThirdPerson(false);
							}}
							icon={RotateCcw}
						>
							reset
						</SchematicButton>
					</div>

					<div className="grid grid-cols-2 gap-x-4 gap-y-3 mt-4 pt-3 border-t border-bg2">
						<DataReadout label="commands" value={`${seq.length}`} />
						<DataReadout
							label="path-length spread"
							value={
								<span style={{ color: spread > 5 ? "var(--red)" : "var(--green)" }}>
									{spread.toFixed(0)}%
								</span>
							}
						/>
					</div>
					<p className="text-[11px] font-mono text-mute mt-2">
						{calibrated
							? "one metric scale — the same command travels the same distance in every source"
							: "raw source units — the same command means a different distance in each source"}
					</p>
				</SchematicCard>

				<SchematicCard title="MODALITIES" status="generated together">
					<p className="text-[12px] text-ink2 leading-relaxed mb-3">
						Every modality is produced from the same trajectory step, not stitched together afterwards. The
						playhead is the step being generated.
					</p>

					<div className="flex flex-col gap-2">
						{MODALITIES.map((m) => (
							<div key={m.key}>
								<div className="flex items-baseline justify-between mb-1">
									<span className="text-[11px] font-mono" style={{ color: m.color }}>
										{m.label}
									</span>
								</div>
								<div className="flex gap-[2px]">
									{seq.map((_, i) => (
										<div
											key={i}
											className="flex-1 rounded-[1px]"
											style={{
												height: 14,
												minWidth: 3,
												background: m.color,
												opacity: i < head ? 0.85 : 0.13,
												outline: i === head - 1 ? "1px solid var(--fg)" : undefined,
											}}
										/>
									))}
								</div>
							</div>
						))}
					</div>

					<div className="grid grid-cols-2 gap-x-4 gap-y-3 mt-4 pt-3 border-t border-bg2">
						<DataReadout label="step" value={`${head} / ${seq.length}`} />
						<DataReadout label="in sync" value={<span className="text-mint-400">4 / 4 modalities</span>} />
					</div>
				</SchematicCard>
			</div>

			<SchematicCard title="WHAT TO TRY">
				<ul className="text-[12.5px] text-ink2 leading-relaxed flex flex-col gap-2">
					<li>
						<TechBadge label="calibration off" /> — the three paths fan apart and the spread jumps. The same
						command sequence now traces a different route in every source, which is the problem calibration
						exists to solve.
					</li>
					<li>
						<TechBadge label="add commands" /> — every button writes into the same 6-DoF trajectory. There is no
						separate path for discrete commands and continuous poses; that unification is the paper's interface
						claim.
					</li>
					<li>
						<TechBadge label="third person" /> — the camera detaches and trails the subject. The trajectory does
						not change, which is why no view-specific controller is needed.
					</li>
					<li>
						<TechBadge label="watch the lanes" /> — all four modalities advance on the same step. Omnimodal means
						generated together, not generated separately and aligned later.
					</li>
				</ul>
				<p className="text-[11.5px] text-mute mt-4 leading-relaxed">
					A toy of the control interface, not of the model. Nothing here generates video or audio, the source
					scales are invented to make the calibration problem visible, and the paths are a 2D projection of a
					6-DoF trajectory. The paper's measured results are in the entry above.
				</p>
			</SchematicCard>
		</div>
	);
};

export default EchoWMSimulation;
