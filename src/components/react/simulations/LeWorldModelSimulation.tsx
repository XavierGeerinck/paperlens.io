import React, { useEffect, useMemo, useRef, useState } from "react";
import {
	AlertTriangle,
	CheckCircle2,
	Pause,
	Play,
	RefreshCw,
	Shuffle,
} from "lucide-react";
import { SchematicCard, SchematicButton, DataReadout } from "../SketchElements";

// ---------------------------------------------------------------------------
// Deterministic RNG (mulberry32). Keeps every visualization reproducible.
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
function gauss(r: () => number): number {
	const u1 = Math.max(r(), 1e-12);
	const u2 = r();
	return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Toy environment — a 2D particle bouncing inside a unit square.
// State is (x, y, vx, vy) ∈ [0, 1]^2 × ℝ^2.
// ---------------------------------------------------------------------------
type Particle = { x: number; y: number; vx: number; vy: number };

function stepParticle(p: Particle, dt: number): Particle {
	let { x, y, vx, vy } = p;
	x += vx * dt;
	y += vy * dt;
	if (x < 0) { x = -x; vx = -vx; }
	if (x > 1) { x = 2 - x; vx = -vx; }
	if (y < 0) { y = -y; vy = -vy; }
	if (y > 1) { y = 2 - y; vy = -vy; }
	return { x, y, vx, vy };
}

function spawnParticle(r: () => number, speed: number): Particle {
	return {
		x: 0.1 + 0.8 * r(),
		y: 0.1 + 0.8 * r(),
		vx: (r() - 0.5) * 2 * speed,
		vy: (r() - 0.5) * 2 * speed,
	};
}

// Render a particle to a smooth GRID×GRID occupancy frame as a Gaussian blob.
const GRID = 12;
const FRAME_DIM = GRID * GRID; // 144

function renderFrame(p: Particle): Float64Array {
	const buf = new Float64Array(FRAME_DIM);
	const sigma = 0.06;
	const s2 = sigma * sigma;
	for (let i = 0; i < GRID; i++) {
		const gy = (i + 0.5) / GRID;
		for (let j = 0; j < GRID; j++) {
			const gx = (j + 0.5) / GRID;
			const d2 = (gx - p.x) ** 2 + (gy - p.y) ** 2;
			buf[i * GRID + j] = Math.exp(-d2 / (2 * s2));
		}
	}
	return buf;
}

// ---------------------------------------------------------------------------
// Tiny MLP (in_dim → hidden → out_dim) with ReLU. Hand-derived gradients.
// ---------------------------------------------------------------------------
class TinyMLP {
	inDim: number;
	hDim: number;
	outDim: number;
	W1: Float64Array;
	b1: Float64Array;
	W2: Float64Array;
	b2: Float64Array;

	constructor(inDim: number, hDim: number, outDim: number, seed: number) {
		this.inDim = inDim;
		this.hDim = hDim;
		this.outDim = outDim;
		const r = rng(seed);
		const s1 = Math.sqrt(2 / inDim);
		const s2 = Math.sqrt(2 / hDim);
		this.W1 = new Float64Array(hDim * inDim);
		this.b1 = new Float64Array(hDim);
		this.W2 = new Float64Array(outDim * hDim);
		this.b2 = new Float64Array(outDim);
		for (let i = 0; i < this.W1.length; i++) this.W1[i] = gauss(r) * s1;
		for (let i = 0; i < this.W2.length; i++) this.W2[i] = gauss(r) * s2;
	}

	forward(x: Float64Array): {
		pre: Float64Array;
		h: Float64Array;
		out: Float64Array;
	} {
		const pre = new Float64Array(this.hDim);
		const h = new Float64Array(this.hDim);
		for (let i = 0; i < this.hDim; i++) {
			let s = this.b1[i];
			const off = i * this.inDim;
			for (let j = 0; j < this.inDim; j++) s += this.W1[off + j] * x[j];
			pre[i] = s;
			h[i] = s > 0 ? s : 0;
		}
		const out = new Float64Array(this.outDim);
		for (let i = 0; i < this.outDim; i++) {
			let s = this.b2[i];
			const off = i * this.hDim;
			for (let j = 0; j < this.hDim; j++) s += this.W2[off + j] * h[j];
			out[i] = s;
		}
		return { pre, h, out };
	}

	backward(
		x: Float64Array,
		pre: Float64Array,
		h: Float64Array,
		gradOut: Float64Array,
	): {
		gW1: Float64Array;
		gb1: Float64Array;
		gW2: Float64Array;
		gb2: Float64Array;
		gX: Float64Array;
	} {
		const gW2 = new Float64Array(this.outDim * this.hDim);
		const gb2 = new Float64Array(this.outDim);
		const gH = new Float64Array(this.hDim);
		for (let i = 0; i < this.outDim; i++) {
			gb2[i] = gradOut[i];
			const off = i * this.hDim;
			for (let j = 0; j < this.hDim; j++) {
				gW2[off + j] = gradOut[i] * h[j];
				gH[j] += this.W2[off + j] * gradOut[i];
			}
		}
		const gPre = new Float64Array(this.hDim);
		for (let i = 0; i < this.hDim; i++) gPre[i] = pre[i] > 0 ? gH[i] : 0;

		const gW1 = new Float64Array(this.hDim * this.inDim);
		const gb1 = new Float64Array(this.hDim);
		const gX = new Float64Array(this.inDim);
		for (let i = 0; i < this.hDim; i++) {
			gb1[i] = gPre[i];
			const off = i * this.inDim;
			for (let j = 0; j < this.inDim; j++) {
				gW1[off + j] = gPre[i] * x[j];
				gX[j] += this.W1[off + j] * gPre[i];
			}
		}
		return { gW1, gb1, gW2, gb2, gX };
	}

	applyGrad(
		gW1: Float64Array,
		gb1: Float64Array,
		gW2: Float64Array,
		gb2: Float64Array,
		lr: number,
	) {
		for (let i = 0; i < this.W1.length; i++) this.W1[i] -= lr * gW1[i];
		for (let i = 0; i < this.b1.length; i++) this.b1[i] -= lr * gb1[i];
		for (let i = 0; i < this.W2.length; i++) this.W2[i] -= lr * gW2[i];
		for (let i = 0; i < this.b2.length; i++) this.b2[i] -= lr * gb2[i];
	}
}

// ---------------------------------------------------------------------------
// LeWM training step over a batch of (frame_t, frame_{t+1}) pairs.
// ---------------------------------------------------------------------------
type StepResult = {
	Lpred: number;
	Lgauss: number;
	latents: Float64Array[];
	sigma: Float64Array;
};

function trainStep(
	enc: TinyMLP,
	pred: TinyMLP,
	frames_t: Float64Array[],
	frames_tp1: Float64Array[],
	lambda: number,
	lr: number,
): StepResult {
	const N = frames_t.length;
	const D = enc.outDim;

	const encFwdT = frames_t.map((f) => enc.forward(f));
	const Zt = encFwdT.map((e) => e.out);
	const Ztp1 = frames_tp1.map((f) => enc.forward(f).out);
	const predFwd = Zt.map((z) => pred.forward(z));
	const Zhat = predFwd.map((f) => f.out);

	let Lpred = 0;
	const gradZhat: Float64Array[] = [];
	for (let n = 0; n < N; n++) {
		const g = new Float64Array(D);
		for (let i = 0; i < D; i++) {
			const diff = Zhat[n][i] - Ztp1[n][i];
			Lpred += diff * diff;
			g[i] = (2 * diff) / N;
		}
		gradZhat.push(g);
	}
	Lpred /= N;

	const mu = new Float64Array(D);
	for (let n = 0; n < N; n++)
		for (let i = 0; i < D; i++) mu[i] += Zt[n][i] / N;
	const sigma = new Float64Array(D * D);
	for (let n = 0; n < N; n++) {
		for (let i = 0; i < D; i++) {
			const di = Zt[n][i] - mu[i];
			for (let j = 0; j < D; j++) {
				const dj = Zt[n][j] - mu[j];
				sigma[i * D + j] += (di * dj) / N;
			}
		}
	}
	let Lgauss = 0;
	for (let i = 0; i < D; i++) Lgauss += mu[i] * mu[i];
	for (let i = 0; i < D; i++)
		for (let j = 0; j < D; j++) {
			const target = i === j ? 1 : 0;
			Lgauss += (sigma[i * D + j] - target) ** 2;
		}

	const gradZtFromReg: Float64Array[] = [];
	for (let n = 0; n < N; n++) {
		const g = new Float64Array(D);
		for (let i = 0; i < D; i++) {
			let acc = (2 / N) * mu[i];
			for (let j = 0; j < D; j++) {
				const target = i === j ? 1 : 0;
				acc += (4 / N) * (sigma[i * D + j] - target) * (Zt[n][j] - mu[j]);
			}
			g[i] = lambda * acc;
		}
		gradZtFromReg.push(g);
	}

	const accPredGrad = {
		gW1: new Float64Array(pred.W1.length),
		gb1: new Float64Array(pred.b1.length),
		gW2: new Float64Array(pred.W2.length),
		gb2: new Float64Array(pred.b2.length),
	};
	const gradZtFromPred: Float64Array[] = [];
	for (let n = 0; n < N; n++) {
		const bw = pred.backward(Zt[n], predFwd[n].pre, predFwd[n].h, gradZhat[n]);
		for (let i = 0; i < accPredGrad.gW1.length; i++) accPredGrad.gW1[i] += bw.gW1[i];
		for (let i = 0; i < accPredGrad.gb1.length; i++) accPredGrad.gb1[i] += bw.gb1[i];
		for (let i = 0; i < accPredGrad.gW2.length; i++) accPredGrad.gW2[i] += bw.gW2[i];
		for (let i = 0; i < accPredGrad.gb2.length; i++) accPredGrad.gb2[i] += bw.gb2[i];
		gradZtFromPred.push(bw.gX);
	}

	const gradZt: Float64Array[] = [];
	for (let n = 0; n < N; n++) {
		const g = new Float64Array(D);
		for (let i = 0; i < D; i++) g[i] = gradZtFromPred[n][i] + gradZtFromReg[n][i];
		gradZt.push(g);
	}

	const accEncGrad = {
		gW1: new Float64Array(enc.W1.length),
		gb1: new Float64Array(enc.b1.length),
		gW2: new Float64Array(enc.W2.length),
		gb2: new Float64Array(enc.b2.length),
	};
	for (let n = 0; n < N; n++) {
		const bw = enc.backward(frames_t[n], encFwdT[n].pre, encFwdT[n].h, gradZt[n]);
		for (let i = 0; i < accEncGrad.gW1.length; i++) accEncGrad.gW1[i] += bw.gW1[i];
		for (let i = 0; i < accEncGrad.gb1.length; i++) accEncGrad.gb1[i] += bw.gb1[i];
		for (let i = 0; i < accEncGrad.gW2.length; i++) accEncGrad.gW2[i] += bw.gW2[i];
		for (let i = 0; i < accEncGrad.gb2.length; i++) accEncGrad.gb2[i] += bw.gb2[i];
	}

	pred.applyGrad(accPredGrad.gW1, accPredGrad.gb1, accPredGrad.gW2, accPredGrad.gb2, lr);
	enc.applyGrad(accEncGrad.gW1, accEncGrad.gb1, accEncGrad.gW2, accEncGrad.gb2, lr);

	return { Lpred, Lgauss, latents: Zt, sigma };
}

// ---------------------------------------------------------------------------
// Build a random training batch of (frame_t, frame_{t+1}) pairs.
// ---------------------------------------------------------------------------
function buildBatch(N: number, speed: number, dt: number, seedBase: number): {
	frames_t: Float64Array[];
	frames_tp1: Float64Array[];
	states_t: Particle[];
} {
	const frames_t: Float64Array[] = [];
	const frames_tp1: Float64Array[] = [];
	const states_t: Particle[] = [];
	for (let n = 0; n < N; n++) {
		const r = rng(seedBase + n * 7919);
		const p0 = spawnParticle(r, speed);
		const p1 = stepParticle(p0, dt);
		frames_t.push(renderFrame(p0));
		frames_tp1.push(renderFrame(p1));
		states_t.push(p0);
	}
	return { frames_t, frames_tp1, states_t };
}

// ---------------------------------------------------------------------------
// Linear probe fit via ridge least squares.
// ---------------------------------------------------------------------------
function fitLinearProbe(
	latents: Float64Array[],
	targets: number[][],
): { W: number[][]; b: number[] } {
	const N = latents.length;
	const D = latents[0].length;
	const M = D + 1;
	const Z: number[][] = latents.map((l) => [...Array.from(l), 1]);
	const ZtZ: number[][] = Array.from({ length: M }, () => new Array(M).fill(0));
	const ZtY: number[][] = Array.from({ length: M }, () => [0, 0]);
	for (let n = 0; n < N; n++) {
		for (let i = 0; i < M; i++) {
			for (let j = 0; j < M; j++) ZtZ[i][j] += Z[n][i] * Z[n][j];
			ZtY[i][0] += Z[n][i] * targets[n][0];
			ZtY[i][1] += Z[n][i] * targets[n][1];
		}
	}
	for (let i = 0; i < M; i++) ZtZ[i][i] += 1e-4;
	const A: number[][] = ZtZ.map((row, i) => [...row, ZtY[i][0], ZtY[i][1]]);
	for (let i = 0; i < M; i++) {
		let pivot = i;
		for (let k = i + 1; k < M; k++)
			if (Math.abs(A[k][i]) > Math.abs(A[pivot][i])) pivot = k;
		[A[i], A[pivot]] = [A[pivot], A[i]];
		const div = A[i][i] || 1e-12;
		for (let j = 0; j < M + 2; j++) A[i][j] /= div;
		for (let k = 0; k < M; k++) {
			if (k === i) continue;
			const f = A[k][i];
			for (let j = 0; j < M + 2; j++) A[k][j] -= f * A[i][j];
		}
	}
	const wMat: number[][] = [[], []];
	for (let d = 0; d < D; d++) {
		wMat[0][d] = A[d][M];
		wMat[1][d] = A[d][M + 1];
	}
	const bVec: number[] = [A[D][M], A[D][M + 1]];
	return { W: wMat, b: bVec };
}

function decodeProbe(probe: { W: number[][]; b: number[] }, z: Float64Array): [number, number] {
	let x = probe.b[0];
	let y = probe.b[1];
	for (let i = 0; i < z.length; i++) {
		x += probe.W[0][i] * z[i];
		y += probe.W[1][i] * z[i];
	}
	return [x, y];
}

// ---------------------------------------------------------------------------
// UI primitives.
// ---------------------------------------------------------------------------

const FrameGrid: React.FC<{ frame: Float64Array; size?: number }> = ({ frame, size = 96 }) => {
	const cell = size / GRID;
	return (
		<svg width={size} height={size} className="block border border-zinc-800 bg-zinc-950">
			{Array.from({ length: FRAME_DIM }).map((_, k) => {
				const i = Math.floor(k / GRID);
				const j = k % GRID;
				const v = frame[k];
				return (
					<rect
						key={k}
						x={j * cell}
						y={i * cell}
						width={cell}
						height={cell}
						fill={`rgba(99, 102, 241, ${Math.min(1, v)})`}
					/>
				);
			})}
		</svg>
	);
};

const ScatterPanel: React.FC<{
	title: string;
	subtitle: string;
	points: { x: number; y: number }[];
	color: string;
	showUnitEllipse?: boolean;
	status: React.ReactNode;
}> = ({ title, subtitle, points, color, showUnitEllipse, status }) => {
	const W = 220;
	const H = 200;
	const range: [number, number] = [-3, 3];
	const sx = (x: number) => ((x - range[0]) / (range[1] - range[0])) * W;
	const sy = (y: number) => H - ((y - range[0]) / (range[1] - range[0])) * H;
	return (
		<div className="flex flex-col gap-1">
			<div className="flex items-baseline justify-between">
				<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-300">{title}</div>
				<div className="text-[9px] font-mono text-zinc-500">{subtitle}</div>
			</div>
			<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block bg-zinc-950 border border-zinc-800">
				<line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#27272a" />
				<line x1={W / 2} y1={0} x2={W / 2} y2={H} stroke="#27272a" />
				{showUnitEllipse && (
					<>
						<ellipse
							cx={W / 2}
							cy={H / 2}
							rx={W / 6}
							ry={H / 6}
							fill="none"
							stroke="#f59e0b"
							strokeWidth={1}
							strokeDasharray="3 3"
							opacity={0.8}
						/>
						<text x={W / 2 + W / 6 + 4} y={H / 2 - 4} fontFamily="monospace" fontSize={9} fill="#f59e0b">
							N(0,I)
						</text>
					</>
				)}
				{points.map((p, i) => (
					<circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2} fill={color} opacity={0.55} />
				))}
			</svg>
			<div className="text-[10px] font-mono">{status}</div>
		</div>
	);
};

const LineChart: React.FC<{
	series: { values: number[]; color: string; label: string }[];
	yLog?: boolean;
	height?: number;
	yLabel?: string;
}> = ({ series, yLog = false, height = 120, yLabel }) => {
	const W = 280;
	const H = height;
	const N = Math.max(...series.map((s) => s.values.length), 1);
	const all = series.flatMap((s) => s.values);
	const minRaw = Math.min(...all, 0);
	const maxRaw = Math.max(...all, 1);
	const tr = (v: number) => (yLog ? Math.log10(Math.max(v, 1e-6)) : v);
	const lo = tr(Math.max(minRaw, 1e-6));
	const hi = tr(Math.max(maxRaw, 1e-6));
	const span = Math.max(hi - lo, 1e-6);
	const sx = (i: number) => (i / Math.max(N - 1, 1)) * (W - 30) + 25;
	const sy = (v: number) => H - 14 - ((tr(v) - lo) / span) * (H - 24);
	return (
		<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
			<line x1={25} y1={H - 14} x2={W} y2={H - 14} stroke="#3f3f46" />
			{yLabel && (
				<text x={4} y={14} fontFamily="monospace" fontSize={9} fill="#71717a">
					{yLabel}
				</text>
			)}
			{series.map((s, k) => (
				<g key={k}>
					<path
						d={s.values
							.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i)},${sy(v)}`)
							.join(" ")}
						fill="none"
						stroke={s.color}
						strokeWidth={1.5}
					/>
					<text x={W - 88} y={14 + k * 12} fontFamily="monospace" fontSize={10} fill={s.color}>
						{s.label}
					</text>
				</g>
			))}
		</svg>
	);
};

// ---------------------------------------------------------------------------
// Integrated dashboard. Trains naive and LeWM in parallel on the same batches.
// ---------------------------------------------------------------------------
const LeWorldModelSimulation: React.FC = () => {
	const [seed, setSeed] = useState<number>(7);
	const [speed, setSpeed] = useState<number>(1.0);
	const [lambda, setLambda] = useState<number>(1.0);
	const [running, setRunning] = useState<boolean>(true);
	const [rolloutSeed, setRolloutSeed] = useState<number>(0);

	const encNaive = useRef<TinyMLP | null>(null);
	const predNaive = useRef<TinyMLP | null>(null);
	const encLewm = useRef<TinyMLP | null>(null);
	const predLewm = useRef<TinyMLP | null>(null);

	const [step, setStep] = useState(0);
	const [latNaive, setLatNaive] = useState<{ x: number; y: number }[]>([]);
	const [latLewm, setLatLewm] = useState<{ x: number; y: number }[]>([]);
	const [traceNaive, setTraceNaive] = useState(1);
	const [traceLewm, setTraceLewm] = useState(1);
	const [traceNaiveHist, setTraceNaiveHist] = useState<number[]>([]);
	const [traceLewmHist, setTraceLewmHist] = useState<number[]>([]);
	const [LpredNaive, setLpredNaive] = useState(0);
	const [LpredLewm, setLpredLewm] = useState(0);
	const [LgaussLewm, setLgaussLewm] = useState(0);

	// Demo particle — bounces continuously so the pixel panel looks like a single
	// object moving through time, rather than a fresh sample per training tick.
	// Training itself still uses independent (frame_t, frame_{t+1}) batches.
	const demoParticleRef = useRef<Particle | null>(null);
	const [liveFrame, setLiveFrame] = useState<Float64Array>(new Float64Array(FRAME_DIM));

	const [rollout, setRollout] = useState<{
		truth: [number, number][];
		naive: [number, number][];
		lewm: [number, number][];
	}>({ truth: [], naive: [], lewm: [] });

	// Reset both networks on seed change.
	useEffect(() => {
		encNaive.current = new TinyMLP(FRAME_DIM, 8, 2, seed);
		predNaive.current = new TinyMLP(2, 8, 2, seed + 1);
		encLewm.current = new TinyMLP(FRAME_DIM, 8, 2, seed);
		predLewm.current = new TinyMLP(2, 8, 2, seed + 1);
		demoParticleRef.current = spawnParticle(rng(seed * 97 + 5), 1.0);
		setLiveFrame(renderFrame(demoParticleRef.current));
		setStep(0);
		setLatNaive([]);
		setLatLewm([]);
		setTraceNaiveHist([]);
		setTraceLewmHist([]);
		setRollout({ truth: [], naive: [], lewm: [] });
	}, [seed]);

	// Continuous demo-particle animation. Independent of the training tick:
	// shows the viewer a coherent object moving so the pixel panel reads as
	// "this is what the encoder sees", not "noise sampled per step".
	useEffect(() => {
		const id = window.setInterval(() => {
			const p = demoParticleRef.current;
			if (!p) return;
			// dt scales with speed so the slider has a visible effect.
			const next = stepParticle(p, 0.04 * speed);
			demoParticleRef.current = next;
			setLiveFrame(renderFrame(next));
		}, 50);
		return () => clearInterval(id);
	}, [speed]);

	// Recompute rollouts from current weights and a fresh start state.
	const recomputeRollout = (rTick: number) => {
		const enN = encNaive.current;
		const enL = encLewm.current;
		const prN = predNaive.current;
		const prL = predLewm.current;
		if (!enN || !enL || !prN || !prL) return;
		// Fit fresh probes for both models on a held-out batch.
		const probeBatch = buildBatch(96, speed, 0.1, seed * 41 + 1234);
		const targets = probeBatch.states_t.map((s) => [s.x, s.y]);
		const probeN = fitLinearProbe(
			probeBatch.frames_t.map((f) => enN.forward(f).out),
			targets,
		);
		const probeL = fitLinearProbe(
			probeBatch.frames_t.map((f) => enL.forward(f).out),
			targets,
		);
		const r = rng(seed * 7 + 11 + rTick * 313);
		let truth = spawnParticle(r, speed);
		const truthPts: [number, number][] = [[truth.x, truth.y]];
		let zN = enN.forward(renderFrame(truth)).out;
		let zL = enL.forward(renderFrame(truth)).out;
		const naivePts: [number, number][] = [decodeProbe(probeN, zN)];
		const lewmPts: [number, number][] = [decodeProbe(probeL, zL)];
		const HORIZON = 25;
		for (let k = 0; k < HORIZON; k++) {
			truth = stepParticle(truth, 0.1);
			truthPts.push([truth.x, truth.y]);
			zN = prN.forward(zN).out;
			zL = prL.forward(zL).out;
			naivePts.push(decodeProbe(probeN, zN));
			lewmPts.push(decodeProbe(probeL, zL));
		}
		setRollout({ truth: truthPts, naive: naivePts, lewm: lewmPts });
	};

	// Training loop.
	useEffect(() => {
		if (!running) return;
		const id = window.setInterval(() => {
			const enN = encNaive.current;
			const prN = predNaive.current;
			const enL = encLewm.current;
			const prL = predLewm.current;
			if (!enN || !prN || !enL || !prL) return;
			setStep((prevStep) => {
				const nextStep = prevStep + 1;
				const batch = buildBatch(16, speed, 0.1, seed * 13 + nextStep * 911);
				const resN = trainStep(enN, prN, batch.frames_t, batch.frames_tp1, 0, 0.05);
				const resL = trainStep(enL, prL, batch.frames_t, batch.frames_tp1, lambda, 0.05);

				setLpredNaive(resN.Lpred);
				setLpredLewm(resL.Lpred);
				setLgaussLewm(resL.Lgauss);
				const trN = resN.sigma[0] + resN.sigma[3];
				const trL = resL.sigma[0] + resL.sigma[3];
				setTraceNaive(trN);
				setTraceLewm(trL);
				setTraceNaiveHist((h) => [...h.slice(-79), Math.max(trN, 1e-6)]);
				setTraceLewmHist((h) => [...h.slice(-79), Math.max(trL, 1e-6)]);
				setLatNaive((prev) => {
					const next = prev.slice(-180);
					for (const z of resN.latents) next.push({ x: z[0], y: z[1] });
					return next.slice(-200);
				});
				setLatLewm((prev) => {
					const next = prev.slice(-180);
					for (const z of resL.latents) next.push({ x: z[0], y: z[1] });
					return next.slice(-200);
				});

				// Refresh rollouts every 25 ticks.
				if (nextStep % 25 === 0) recomputeRollout(nextStep);

				return nextStep;
			});
		}, 80);
		return () => clearInterval(id);
	}, [running, seed, speed, lambda]);

	// Recompute rollout when user requests a new starting state.
	useEffect(() => {
		if (step > 0) recomputeRollout(rolloutSeed);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [rolloutSeed]);

	const naiveCollapsed = traceNaive < 0.05 && step > 30;
	const lewmHealthy = traceLewm > 0.5 && step > 30;

	const W = 280;
	const H = 180;
	const sx = (v: number) => v * W;
	const sy = (v: number) => H - v * H;

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="LIVE TRAINING · NAIVE vs LeWM (SAME SEED, SAME BATCHES)" status={`STEP ${step}`}>
				<div className="flex flex-col gap-4">
					<p className="text-xs font-mono text-zinc-400 leading-relaxed">
						Two identical networks train on the same stream of pixel pairs. Only one extra term — the
						Gaussian moment-matching regularizer <code>‖μ‖² + ‖Σ − I‖²_F</code> — separates them. Watch the
						naive run's latents shrink to a point while the LeWM run keeps the unit-variance shape that
						makes the latent informative enough to plan in.
					</p>

					<div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-start">
						<label className="flex flex-col gap-1">
							<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
								Particle speed: {speed.toFixed(2)}
							</span>
							<input
								type="range"
								min={0.5}
								max={2.0}
								step={0.1}
								value={speed}
								onChange={(e) => setSpeed(Number(e.target.value))}
							/>
						</label>
						<label className="flex flex-col gap-1">
							<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
								λ (LeWM regularizer): {lambda.toFixed(2)}
							</span>
							<input
								type="range"
								min={0}
								max={2}
								step={0.05}
								value={lambda}
								onChange={(e) => setLambda(Number(e.target.value))}
							/>
						</label>
						<SchematicButton
							onClick={() => setRunning((r) => !r)}
							icon={running ? Pause : Play}
							active={running}
						>
							{running ? "Pause" : "Resume"}
						</SchematicButton>
						<SchematicButton onClick={() => setSeed((s) => s + 1)} icon={RefreshCw}>
							Reseed
						</SchematicButton>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-start">
						<div className="flex flex-col gap-1">
							<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-300">
								Pixel input
							</div>
							<div className="text-[9px] font-mono text-zinc-500">
								{GRID}×{GRID} grid · {FRAME_DIM}-d vector
							</div>
							<FrameGrid frame={liveFrame} size={220} />
							<div className="text-[10px] font-mono text-zinc-500">
								↓ shared 256→8→2 architecture, different objective
							</div>
						</div>

						<ScatterPanel
							title="Naive  ·  λ = 0"
							subtitle="Latent z_t scatter (last 200)"
							points={latNaive}
							color="#ef4444"
							status={
								naiveCollapsed ? (
									<span className="text-red-400 inline-flex items-center gap-1">
										<AlertTriangle size={11} /> COLLAPSED · tr(Σ) = {traceNaive.toExponential(2)}
									</span>
								) : (
									<span className="text-amber-400">
										training · tr(Σ) = {traceNaive.toFixed(2)}
									</span>
								)
							}
						/>

						<ScatterPanel
							title={`LeWM  ·  λ = ${lambda.toFixed(2)}`}
							subtitle="Latent z_t scatter (last 200)"
							points={latLewm}
							color="#6366f1"
							showUnitEllipse
							status={
								lewmHealthy ? (
									<span className="text-emerald-400 inline-flex items-center gap-1">
										<CheckCircle2 size={11} /> STABLE · tr(Σ) = {traceLewm.toFixed(2)}
									</span>
								) : (
									<span className="text-amber-400">
										training · tr(Σ) = {traceLewm.toFixed(2)}
									</span>
								)
							}
						/>
					</div>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
						<div className="flex flex-col gap-1">
							<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-300">
								Latent variance over training · tr(Σ)
							</div>
							<LineChart
								series={[
									{ values: traceNaiveHist, color: "#ef4444", label: "naive" },
									{ values: traceLewmHist, color: "#6366f1", label: "LeWM" },
								]}
								yLog
								height={140}
								yLabel="log tr(Σ)"
							/>
						</div>

						<div className="flex flex-col gap-1">
							<div className="flex items-baseline justify-between">
								<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-300">
									Latent rollout · decode predictor to xy
								</div>
								<button
									onClick={() => setRolloutSeed((t) => t + 1)}
									className="text-[9px] font-mono text-zinc-400 inline-flex items-center gap-1 hover:text-zinc-200"
								>
									<Shuffle size={10} /> new start
								</button>
							</div>
							<svg
								width="100%"
								viewBox={`0 0 ${W} ${H}`}
								className="block bg-zinc-950 border border-zinc-800"
							>
								<rect x={1} y={1} width={W - 2} height={H - 2} fill="none" stroke="#27272a" />
								{rollout.truth.length > 0 && (
									<>
										<path
											d={rollout.truth
												.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`)
												.join(" ")}
											fill="none"
											stroke="#10b981"
											strokeWidth={2}
										/>
										<path
											d={rollout.naive
												.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`)
												.join(" ")}
											fill="none"
											stroke="#ef4444"
											strokeWidth={1.5}
											strokeDasharray="3 3"
										/>
										<path
											d={rollout.lewm
												.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`)
												.join(" ")}
											fill="none"
											stroke="#6366f1"
											strokeWidth={1.5}
											strokeDasharray="3 3"
										/>
										<circle
											cx={sx(rollout.truth[0][0])}
											cy={sy(rollout.truth[0][1])}
											r={4}
											fill="#fbbf24"
										/>
									</>
								)}
								<text x={W - 110} y={14} fontFamily="monospace" fontSize={10} fill="#10b981">
									ground truth
								</text>
								<text x={W - 110} y={26} fontFamily="monospace" fontSize={10} fill="#ef4444">
									naive
								</text>
								<text x={W - 110} y={38} fontFamily="monospace" fontSize={10} fill="#6366f1">
									LeWM
								</text>
							</svg>
							<div className="text-[9px] font-mono text-zinc-500">
								25-step rollout · linear probe refit every 25 training steps
							</div>
						</div>
					</div>

					<div className="grid grid-cols-2 md:grid-cols-5 gap-3">
						<DataReadout label="step" value={String(step)} />
						<DataReadout label="L_pred naive" value={LpredNaive.toExponential(2)} />
						<DataReadout label="L_pred LeWM" value={LpredLewm.toExponential(2)} />
						<DataReadout label="L_gauss LeWM" value={LgaussLewm.toExponential(2)} />
						<DataReadout
							label="tr(Σ) ratio"
							value={(traceLewm / Math.max(traceNaive, 1e-6)).toFixed(1) + "×"}
						/>
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default LeWorldModelSimulation;
