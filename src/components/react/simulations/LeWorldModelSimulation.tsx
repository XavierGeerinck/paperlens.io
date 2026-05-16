import React, { useEffect, useMemo, useRef, useState } from "react";
import {
	Activity,
	AlertTriangle,
	Box,
	Compass,
	Play,
	Pause,
	RefreshCw,
	Sigma,
	Shuffle,
	Target,
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

// Render a particle to a smooth GRID×GRID occupancy frame with a small
// Gaussian blob — smoother than 0/1 pixels, which helps the encoder train.
const GRID = 12;
const FRAME_DIM = GRID * GRID; // 144

function renderFrame(p: Particle): Float64Array {
	const buf = new Float64Array(FRAME_DIM);
	const sigma = 0.06; // pixels in normalized coords
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
// All tensors are flat Float64Arrays in row-major order.
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
		// Kaiming-ish init for ReLU.
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

	// Backward returns gradients w.r.t. parameters AND w.r.t. input x.
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
	latents: Float64Array[]; // batch of z_t for visualization
	mu: Float64Array;
	sigma: Float64Array; // 2x2 covariance flattened
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

	// --- Forward: compute z_t and z_{t+1} (target, stop-grad) for the batch.
	const encFwdT: { pre: Float64Array; h: Float64Array; out: Float64Array }[] = [];
	const encFwdTp1: { out: Float64Array }[] = [];
	for (let n = 0; n < N; n++) {
		encFwdT.push(enc.forward(frames_t[n]));
		encFwdTp1.push({ out: enc.forward(frames_tp1[n]).out });
	}
	const Zt = encFwdT.map((e) => e.out);
	const Ztp1 = encFwdTp1.map((e) => e.out);

	// --- Predictor forward.
	const predFwd = Zt.map((z) => pred.forward(z));
	const Zhat = predFwd.map((f) => f.out);

	// --- Prediction loss: ||z_hat - z_tp1||^2 averaged over batch.
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

	// --- Gaussian regularizer on z_t batch.
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

	// --- Gradients of L_gauss w.r.t. z_n.
	// dLgauss/dz_n = (2/N) μ + (4/N) (Σ - I) (z_n - μ)
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

	// --- Backward through predictor → grad w.r.t. z_t (from L_pred).
	const gradZtFromPred: Float64Array[] = [];
	const accPredGrad = {
		gW1: new Float64Array(pred.W1.length),
		gb1: new Float64Array(pred.b1.length),
		gW2: new Float64Array(pred.W2.length),
		gb2: new Float64Array(pred.b2.length),
	};
	for (let n = 0; n < N; n++) {
		const bw = pred.backward(Zt[n], predFwd[n].pre, predFwd[n].h, gradZhat[n]);
		for (let i = 0; i < accPredGrad.gW1.length; i++) accPredGrad.gW1[i] += bw.gW1[i];
		for (let i = 0; i < accPredGrad.gb1.length; i++) accPredGrad.gb1[i] += bw.gb1[i];
		for (let i = 0; i < accPredGrad.gW2.length; i++) accPredGrad.gW2[i] += bw.gW2[i];
		for (let i = 0; i < accPredGrad.gb2.length; i++) accPredGrad.gb2[i] += bw.gb2[i];
		gradZtFromPred.push(bw.gX);
	}

	// --- Combined grad on z_t = pred-backward + regularizer grad.
	const gradZt: Float64Array[] = [];
	for (let n = 0; n < N; n++) {
		const g = new Float64Array(D);
		for (let i = 0; i < D; i++) g[i] = gradZtFromPred[n][i] + gradZtFromReg[n][i];
		gradZt.push(g);
	}

	// --- Backward through encoder using grad on z_t.
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

	return { Lpred, Lgauss, latents: Zt, mu, sigma };
}

// ---------------------------------------------------------------------------
// Build a random training batch of (frame_t, frame_{t+1}) pairs.
// Each sample is an independent particle, advanced by a fixed dt.
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
// UI subcomponents.
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

const ScatterPlot: React.FC<{
	points: { x: number; y: number }[];
	color?: string;
	xRange?: [number, number];
	yRange?: [number, number];
	showUnitEllipse?: boolean;
	height?: number;
}> = ({
	points,
	color = "#6366f1",
	xRange = [-3, 3],
	yRange = [-3, 3],
	showUnitEllipse = false,
	height = 200,
}) => {
	const W = 280;
	const H = height;
	const sx = (x: number) => ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
	const sy = (y: number) => H - ((y - yRange[0]) / (yRange[1] - yRange[0])) * H;
	return (
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
						opacity={0.7}
					/>
					<text x={W / 2 + W / 6 + 4} y={H / 2 - 4} fontFamily="monospace" fontSize={9} fill="#f59e0b">
						N(0, I)
					</text>
				</>
			)}
			{points.map((p, i) => (
				<circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2} fill={color} opacity={0.6} />
			))}
		</svg>
	);
};

const LineChart: React.FC<{
	series: { values: number[]; color: string; label: string }[];
	yLog?: boolean;
	height?: number;
}> = ({ series, yLog = false, height = 100 }) => {
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
	const sy = (v: number) => H - 10 - ((tr(v) - lo) / span) * (H - 20);
	return (
		<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
			<line x1={25} y1={H - 10} x2={W} y2={H - 10} stroke="#3f3f46" />
			{series.map((s, k) => (
				<g key={k}>
					<path
						d={s.values.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i)},${sy(v)}`).join(" ")}
						fill="none"
						stroke={s.color}
						strokeWidth={1.5}
					/>
					<text x={W - 80} y={14 + k * 12} fontFamily="monospace" fontSize={10} fill={s.color}>
						{s.label}
					</text>
				</g>
			))}
		</svg>
	);
};

// ---------------------------------------------------------------------------
// Training hook — runs a tick loop, calls trainStep per tick on a fresh batch.
// ---------------------------------------------------------------------------
function useTrainer(opts: {
	seed: number;
	lambda: number;
	lr: number;
	speed: number;
	dt: number;
	batch: number;
	tickMs: number;
	running: boolean;
}) {
	const { seed, lambda, lr, speed, dt, batch, tickMs, running } = opts;
	const encRef = useRef<TinyMLP | null>(null);
	const predRef = useRef<TinyMLP | null>(null);
	const [step, setStep] = useState(0);
	const [Lpred, setLpred] = useState(0);
	const [Lgauss, setLgauss] = useState(0);
	const [latents, setLatents] = useState<{ x: number; y: number }[]>([]);
	const [mu, setMu] = useState<[number, number]>([0, 0]);
	const [sigma, setSigma] = useState<number[]>([1, 0, 0, 1]);
	const [LpredHist, setLpredHist] = useState<number[]>([]);
	const [LgaussHist, setLgaussHist] = useState<number[]>([]);
	const [varHist, setVarHist] = useState<number[]>([]);

	useEffect(() => {
		encRef.current = new TinyMLP(FRAME_DIM, 8, 2, seed);
		predRef.current = new TinyMLP(2, 8, 2, seed + 1);
		setStep(0);
		setLatents([]);
		setLpredHist([]);
		setLgaussHist([]);
		setVarHist([]);
	}, [seed]);

	useEffect(() => {
		if (!running) return;
		const id = window.setInterval(() => {
			const enc = encRef.current!;
			const pred = predRef.current!;
			const { frames_t, frames_tp1 } = buildBatch(batch, speed, dt, seed * 13 + step * 911);
			const res = trainStep(enc, pred, frames_t, frames_tp1, lambda, lr);
			setStep((s) => s + 1);
			setLpred(res.Lpred);
			setLgauss(res.Lgauss);
			setMu([res.mu[0], res.mu[1]]);
			setSigma([res.sigma[0], res.sigma[1], res.sigma[2], res.sigma[3]]);
			setLatents((prev) => {
				const next = prev.slice(-180);
				for (const z of res.latents) next.push({ x: z[0], y: z[1] });
				return next.slice(-200);
			});
			setLpredHist((h) => [...h.slice(-79), res.Lpred]);
			setLgaussHist((h) => [...h.slice(-79), res.Lgauss]);
			setVarHist((h) => [...h.slice(-79), res.sigma[0] + res.sigma[3]]);
		}, tickMs);
		return () => clearInterval(id);
	}, [running, lambda, lr, speed, dt, batch, tickMs, seed, step]);

	const reset = () => {
		encRef.current = new TinyMLP(FRAME_DIM, 8, 2, seed);
		predRef.current = new TinyMLP(2, 8, 2, seed + 1);
		setStep(0);
		setLatents([]);
		setLpredHist([]);
		setLgaussHist([]);
		setVarHist([]);
	};

	return {
		enc: encRef,
		pred: predRef,
		step,
		Lpred,
		Lgauss,
		latents,
		mu,
		sigma,
		LpredHist,
		LgaussHist,
		varHist,
		reset,
	};
}

// ---------------------------------------------------------------------------
// Stage 1 — pixel environment.
// ---------------------------------------------------------------------------
const Stage1: React.FC<{ seed: number; speed: number; particles: number }> = ({
	seed,
	speed,
	particles,
}) => {
	const [tick, setTick] = useState(0);
	const psRef = useRef<Particle[]>([]);
	useEffect(() => {
		const r = rng(seed * 31 + 1);
		psRef.current = Array.from({ length: particles }, () => spawnParticle(r, speed));
	}, [seed, particles, speed]);

	useEffect(() => {
		const id = window.setInterval(() => {
			psRef.current = psRef.current.map((p) => stepParticle(p, 0.05));
			setTick((t) => t + 1);
		}, 80);
		return () => clearInterval(id);
	}, []);

	const composite = useMemo(() => {
		const buf = new Float64Array(FRAME_DIM);
		for (const p of psRef.current) {
			const f = renderFrame(p);
			for (let i = 0; i < FRAME_DIM; i++) buf[i] = Math.max(buf[i], f[i]);
		}
		return buf;
	}, [tick]);

	const filmstrip = useMemo(() => {
		const out: Float64Array[] = [];
		let copies = psRef.current.map((p) => ({ ...p }));
		for (let k = 0; k < 6; k++) {
			const buf = new Float64Array(FRAME_DIM);
			for (const p of copies) {
				const f = renderFrame(p);
				for (let i = 0; i < FRAME_DIM; i++) buf[i] = Math.max(buf[i], f[i]);
			}
			out.push(buf);
			copies = copies.map((p) => stepParticle(p, 0.05));
		}
		return out;
	}, [tick]);

	return (
		<SchematicCard title="STAGE 1 · THE PIXEL ENVIRONMENT" status="INPUT">
			<div className="flex flex-col gap-4">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					A 2D particle bouncing inside a unit square, rendered onto a {GRID}×{GRID} grid as a Gaussian blob.
					This is what the encoder consumes — a {FRAME_DIM}-dimensional pixel vector with one moving feature.
					The encoder will be asked to compress this to a 2D latent that the predictor can roll forward.
				</p>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Live frame · t = {tick}
						</div>
						<FrameGrid frame={composite} size={192} />
					</div>
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Filmstrip · next 6 steps
						</div>
						<div className="flex flex-wrap gap-1">
							{filmstrip.map((f, i) => (
								<FrameGrid key={i} frame={f} size={56} />
							))}
						</div>
					</div>
				</div>
				<div className="grid grid-cols-3 gap-3">
					<DataReadout label="grid" value={`${GRID}×${GRID}`} />
					<DataReadout label="pixel dim" value={String(FRAME_DIM)} />
					<DataReadout label="particles" value={String(particles)} />
				</div>
			</div>
		</SchematicCard>
	);
};

// ---------------------------------------------------------------------------
// Stage 2 — naive end-to-end JEPA collapses.
// ---------------------------------------------------------------------------
const Stage2: React.FC<{ seed: number; speed: number; running: boolean }> = ({
	seed,
	speed,
	running,
}) => {
	const t = useTrainer({
		seed,
		lambda: 0,
		lr: 0.05,
		speed,
		dt: 0.1,
		batch: 16,
		tickMs: 80,
		running,
	});
	const trace = t.sigma[0] + t.sigma[3];
	const collapsed = trace < 0.05 && t.step > 30;

	return (
		<SchematicCard title="STAGE 2 · NAIVE END-TO-END JEPA" status={collapsed ? "COLLAPSED" : "TRAINING"}>
			<div className="flex flex-col gap-4">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					Train the encoder and predictor jointly with only the prediction loss <code>‖p(z_t) - sg(z_{`{t+1}`})‖²</code>.
					Watch the latent scatter shrink. The loss falls to ~0 because the encoder learns the constant
					function — every frame maps to the same point, the predictor's job becomes trivial, and the
					representation carries zero information about the world.
				</p>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Latent scatter (last 200)
						</div>
						<ScatterPlot points={t.latents} color="#ef4444" />
					</div>
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Prediction loss · latent variance
						</div>
						<LineChart
							series={[
								{ values: t.LpredHist, color: "#ef4444", label: "L_pred" },
								{ values: t.varHist, color: "#10b981", label: "tr(Σ)" },
							]}
							yLog
							height={140}
						/>
					</div>
				</div>
				<div className="grid grid-cols-4 gap-3">
					<DataReadout label="step" value={String(t.step)} />
					<DataReadout label="L_pred" value={t.Lpred.toExponential(2)} />
					<DataReadout label="tr(Σ)" value={trace.toExponential(2)} />
					<DataReadout
						label="status"
						value={
							collapsed ? (
								<span className="text-red-400 inline-flex items-center gap-1">
									<AlertTriangle size={12} /> COLLAPSED
								</span>
							) : (
								<span className="text-amber-400">TRAINING</span>
							)
						}
					/>
				</div>
			</div>
		</SchematicCard>
	);
};

// ---------------------------------------------------------------------------
// Stage 3 — LeWM with Gaussian regularizer. A/B against naive.
// ---------------------------------------------------------------------------
const Stage3: React.FC<{ seed: number; speed: number; running: boolean }> = ({
	seed,
	speed,
	running,
}) => {
	const [lambda, setLambda] = useState(1.0);
	const [mode, setMode] = useState<"lewm" | "naive">("lewm");
	const effectiveLambda = mode === "lewm" ? lambda : 0;
	const t = useTrainer({
		seed,
		lambda: effectiveLambda,
		lr: 0.05,
		speed,
		dt: 0.1,
		batch: 16,
		tickMs: 80,
		running,
	});
	const trace = t.sigma[0] + t.sigma[3];

	return (
		<SchematicCard title="STAGE 3 · + GAUSSIAN REGULARIZER = LEWM" status={mode.toUpperCase()}>
			<div className="flex flex-col gap-4">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					Add the Gaussian moment-matching regularizer <code>‖μ‖² + ‖Σ − I‖²_F</code> with weight λ. The
					latent cloud now stays anchored on the unit-variance ellipse; collapse becomes impossible because
					Σ=0 is far from Σ=I. Toggle to <strong>Naive</strong> to retrain from the same seed without the
					regularizer and see the difference side-by-side.
				</p>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<label className="flex flex-col gap-1">
						<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							λ (regularizer weight): {lambda.toFixed(2)}
						</span>
						<input
							type="range"
							min={0}
							max={2}
							step={0.05}
							value={lambda}
							onChange={(e) => setLambda(Number(e.target.value))}
							disabled={mode === "naive"}
						/>
					</label>
					<div className="flex gap-2">
						<SchematicButton onClick={() => setMode("lewm")} active={mode === "lewm"} icon={Sigma}>
							LeWM
						</SchematicButton>
						<SchematicButton onClick={() => setMode("naive")} active={mode === "naive"} icon={AlertTriangle}>
							Naive
						</SchematicButton>
					</div>
				</div>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Latent scatter · {mode === "lewm" ? "stable" : "collapsing"}
						</div>
						<ScatterPlot
							points={t.latents}
							color={mode === "lewm" ? "#6366f1" : "#ef4444"}
							showUnitEllipse={mode === "lewm"}
						/>
					</div>
					<div className="flex flex-col gap-2">
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Loss curves
						</div>
						<LineChart
							series={[
								{ values: t.LpredHist, color: "#ef4444", label: "L_pred" },
								{ values: t.LgaussHist, color: "#f59e0b", label: "L_gauss" },
							]}
							yLog
							height={140}
						/>
					</div>
				</div>
				<div className="grid grid-cols-4 gap-3">
					<DataReadout label="step" value={String(t.step)} />
					<DataReadout label="L_pred" value={t.Lpred.toExponential(2)} />
					<DataReadout label="L_gauss" value={t.Lgauss.toExponential(2)} />
					<DataReadout label="tr(Σ)" value={trace.toFixed(2)} />
				</div>
			</div>
		</SchematicCard>
	);
};

// ---------------------------------------------------------------------------
// Stage 4 — latent rollout + linear probe. Trains LeWM offline, then plans.
// ---------------------------------------------------------------------------
function fitLinearProbe(
	latents: Float64Array[],
	targets: number[][], // each [x, y] in [0,1]
): { W: number[][]; b: number[] } {
	// Augment latents with a bias column, solve normal equations W = (Z^T Z)^-1 Z^T Y.
	const N = latents.length;
	const D = latents[0].length;
	// Build Z (N x (D+1)) and Y (N x 2).
	const Z: number[][] = [];
	for (let n = 0; n < N; n++) {
		const row = [...Array.from(latents[n]), 1];
		Z.push(row);
	}
	// Compute Z^T Z ((D+1) x (D+1)) and Z^T Y ((D+1) x 2).
	const M = D + 1;
	const ZtZ: number[][] = Array.from({ length: M }, () => new Array(M).fill(0));
	const ZtY: number[][] = Array.from({ length: M }, () => [0, 0]);
	for (let n = 0; n < N; n++) {
		for (let i = 0; i < M; i++) {
			for (let j = 0; j < M; j++) ZtZ[i][j] += Z[n][i] * Z[n][j];
			ZtY[i][0] += Z[n][i] * targets[n][0];
			ZtY[i][1] += Z[n][i] * targets[n][1];
		}
	}
	// Ridge for stability.
	for (let i = 0; i < M; i++) ZtZ[i][i] += 1e-4;
	// Solve via Gauss–Jordan.
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
	// After Gauss–Jordan, A's first M cols are I and the last 2 are the solution
	// for the two probe outputs (x and y). Extract a 2×D weight matrix and 2-vector bias.
	const wMat: number[][] = [[], []];
	for (let d = 0; d < D; d++) {
		wMat[0][d] = A[d][M];
		wMat[1][d] = A[d][M + 1];
	}
	const bVec: number[] = [A[D][M], A[D][M + 1]];
	return { W: wMat, b: bVec };
}

const Stage4: React.FC<{ seed: number; speed: number }> = ({ seed, speed }) => {
	const [horizon, setHorizon] = useState(20);
	const [trainSteps, setTrainSteps] = useState(0);
	const [trained, setTrained] = useState(false);
	const encRef = useRef<TinyMLP | null>(null);
	const predRef = useRef<TinyMLP | null>(null);
	const probeRef = useRef<{ W: number[][]; b: number[] } | null>(null);
	const [rollout, setRollout] = useState<{ pred: [number, number][]; gt: [number, number][] }>({
		pred: [],
		gt: [],
	});
	const [rolloutTick, setRolloutTick] = useState(0);

	// Offline pretraining loop: run a fixed number of steps, then fit the probe.
	useEffect(() => {
		setTrained(false);
		setTrainSteps(0);
		setRollout({ pred: [], gt: [] });
		const enc = new TinyMLP(FRAME_DIM, 8, 2, seed + 100);
		const pred = new TinyMLP(2, 8, 2, seed + 101);
		encRef.current = enc;
		predRef.current = pred;
		const TOTAL = 200;
		let step = 0;
		const id = window.setInterval(() => {
			const { frames_t, frames_tp1 } = buildBatch(16, speed, 0.1, seed * 41 + step * 503);
			trainStep(enc, pred, frames_t, frames_tp1, 1.0, 0.05);
			step++;
			setTrainSteps(step);
			if (step >= TOTAL) {
				// Fit linear probe on a fresh batch.
				const batch = buildBatch(128, speed, 0.1, seed * 17 + 999);
				const lats = batch.frames_t.map((f) => enc.forward(f).out);
				const tgts = batch.states_t.map((s) => [s.x, s.y]);
				probeRef.current = fitLinearProbe(lats, tgts);
				setTrained(true);
				window.clearInterval(id);
			}
		}, 16);
		return () => window.clearInterval(id);
	}, [seed, speed]);

	// Rollout (recomputed when horizon or trained changes, also re-animates).
	useEffect(() => {
		if (!trained || !encRef.current || !predRef.current || !probeRef.current) return;
		const enc = encRef.current;
		const pred = predRef.current;
		const probe = probeRef.current;
		const r = rng(seed * 7 + 11 + rolloutTick);
		let truth = spawnParticle(r, speed);
		const gt: [number, number][] = [[truth.x, truth.y]];
		const predPts: [number, number][] = [];

		let z = enc.forward(renderFrame(truth)).out;
		const decode = (zz: Float64Array): [number, number] => {
			let x = probe.b[0];
			let y = probe.b[1];
			for (let i = 0; i < zz.length; i++) {
				x += probe.W[0][i] * zz[i];
				y += probe.W[1][i] * zz[i];
			}
			return [x, y];
		};
		predPts.push(decode(z));
		for (let k = 0; k < horizon; k++) {
			truth = stepParticle(truth, 0.1);
			gt.push([truth.x, truth.y]);
			z = pred.forward(z).out;
			predPts.push(decode(z));
		}
		setRollout({ pred: predPts, gt });
	}, [trained, horizon, seed, speed, rolloutTick]);

	const W = 320;
	const H = 220;
	const sx = (v: number) => v * W;
	const sy = (v: number) => H - v * H;

	const finalErr = useMemo(() => {
		if (!rollout.pred.length) return 0;
		const k = rollout.pred.length - 1;
		const dx = rollout.pred[k][0] - rollout.gt[k][0];
		const dy = rollout.pred[k][1] - rollout.gt[k][1];
		return Math.sqrt(dx * dx + dy * dy);
	}, [rollout]);

	return (
		<SchematicCard
			title="STAGE 4 · LATENT ROLLOUT + LINEAR PROBE"
			status={trained ? "PLANNING" : `PRETRAIN ${trainSteps}/200`}
		>
			<div className="flex flex-col gap-4">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					Train LeWM for 200 steps with λ=1, then fit a linear probe (z → position) on a fresh batch. Roll
					the predictor forward in latent space for k steps and decode each latent to a 2D position. If the
					latent really encodes the physics, the predicted trajectory should track the ground truth without
					ever generating a pixel.
				</p>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
					<label className="flex flex-col gap-1">
						<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Rollout horizon: {horizon} steps
						</span>
						<input
							type="range"
							min={1}
							max={60}
							step={1}
							value={horizon}
							onChange={(e) => setHorizon(Number(e.target.value))}
							disabled={!trained}
						/>
					</label>
					<SchematicButton
						onClick={() => setRolloutTick((t) => t + 1)}
						icon={Shuffle}
						active={false}
					>
						{trained ? "New start state" : `Pretraining ${trainSteps}/200`}
					</SchematicButton>
				</div>
				<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block bg-zinc-950 border border-zinc-800">
					<rect x={1} y={1} width={W - 2} height={H - 2} fill="none" stroke="#27272a" />
					{/* Ground truth */}
					<path
						d={rollout.gt.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`).join(" ")}
						fill="none"
						stroke="#10b981"
						strokeWidth={2}
					/>
					{/* Predicted */}
					<path
						d={rollout.pred.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0])},${sy(p[1])}`).join(" ")}
						fill="none"
						stroke="#6366f1"
						strokeWidth={2}
						strokeDasharray="3 3"
					/>
					{rollout.gt[0] && (
						<circle cx={sx(rollout.gt[0][0])} cy={sy(rollout.gt[0][1])} r={4} fill="#fbbf24" />
					)}
					<text x={W - 110} y={16} fontFamily="monospace" fontSize={10} fill="#10b981">
						ground truth
					</text>
					<text x={W - 110} y={30} fontFamily="monospace" fontSize={10} fill="#6366f1">
						latent rollout
					</text>
				</svg>
				<div className="grid grid-cols-3 gap-3">
					<DataReadout label="train steps" value={`${trainSteps} / 200`} />
					<DataReadout label="horizon" value={`${horizon}`} />
					<DataReadout label="final error" value={finalErr.toFixed(3)} />
				</div>
			</div>
		</SchematicCard>
	);
};

// ---------------------------------------------------------------------------
// Top-level component.
// ---------------------------------------------------------------------------
type Stage = 1 | 2 | 3 | 4;

const LeWorldModelSimulation: React.FC = () => {
	const [stage, setStage] = useState<Stage>(1);
	const [seed, setSeed] = useState<number>(7);
	const [speed, setSpeed] = useState<number>(1.0);
	const [particles, setParticles] = useState<number>(1);
	const [running, setRunning] = useState<boolean>(true);

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<div className="flex flex-wrap gap-2">
				<SchematicButton onClick={() => setStage(1)} icon={Box} active={stage === 1}>
					1 · Environment
				</SchematicButton>
				<SchematicButton onClick={() => setStage(2)} icon={AlertTriangle} active={stage === 2}>
					2 · Collapse
				</SchematicButton>
				<SchematicButton onClick={() => setStage(3)} icon={Sigma} active={stage === 3}>
					3 · LeWM
				</SchematicButton>
				<SchematicButton onClick={() => setStage(4)} icon={Compass} active={stage === 4}>
					4 · Rollout
				</SchematicButton>
			</div>

			<SchematicCard title="CONTROLS">
				<div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
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
					{stage === 1 && (
						<label className="flex flex-col gap-1">
							<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
								Particles: {particles}
							</span>
							<input
								type="range"
								min={1}
								max={4}
								step={1}
								value={particles}
								onChange={(e) => setParticles(Number(e.target.value))}
							/>
						</label>
					)}
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
			</SchematicCard>

			{stage === 1 && <Stage1 seed={seed} speed={speed} particles={particles} />}
			{stage === 2 && <Stage2 seed={seed} speed={speed} running={running} />}
			{stage === 3 && <Stage3 seed={seed} speed={speed} running={running} />}
			{stage === 4 && <Stage4 seed={seed} speed={speed} />}
		</div>
	);
};

export default LeWorldModelSimulation;
