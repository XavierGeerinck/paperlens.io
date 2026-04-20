import React, { useMemo, useState } from "react";
import { Activity, Shuffle, Sliders, Target } from "lucide-react";
import { SchematicCard, SchematicButton, DataReadout } from "../SketchElements";

// --- Deterministic RNG (mulberry32) — keeps visualizations stable across renders. ---
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

// Box–Muller from uniform RNG.
function gauss(r: () => number): number {
	const u1 = Math.max(r(), 1e-12);
	const u2 = r();
	return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Synthetic KV-like embedding: mostly Gaussian, with a heavy-tailed outlier channel.
function synthVector(d: number, outlierIntensity: number, seed: number): number[] {
	const r = rng(seed);
	const v: number[] = new Array(d);
	for (let i = 0; i < d; i++) {
		let x = gauss(r);
		if (r() < 0.05) x *= 1 + 8 * outlierIntensity;
		v[i] = x;
	}
	return v;
}

function histogram(values: number[], binCount: number, range: [number, number]) {
	const [lo, hi] = range;
	const bins = new Array(binCount).fill(0);
	const w = (hi - lo) / binCount;
	for (const v of values) {
		if (v < lo || v > hi) continue;
		const i = Math.min(binCount - 1, Math.max(0, Math.floor((v - lo) / w)));
		bins[i]++;
	}
	return bins;
}

// Random sign flip + permutation — a cheap, orthogonal near-approximation of a random
// rotation, sufficient for the visual point we're making. Real implementations use a
// Hadamard butterfly for O(d log d) cost; we use a 2-point mixing step of the same spirit.
function randomRotation(d: number, seed: number): { apply: (x: number[]) => number[] } {
	const r = rng(seed);
	const signs = Array.from({ length: d }, () => (r() < 0.5 ? -1 : 1));
	const perm = Array.from({ length: d }, (_, i) => i);
	for (let i = d - 1; i > 0; i--) {
		const j = Math.floor(r() * (i + 1));
		[perm[i], perm[j]] = [perm[j], perm[i]];
	}
	return {
		apply(x: number[]): number[] {
			const y = new Array(d);
			for (let i = 0; i < d; i++) {
				const a = signs[i] * x[perm[i]];
				const b = signs[(i + 1) % d] * x[perm[(i + 1) % d]];
				y[i] = (a + b) / Math.SQRT2;
			}
			return y;
		},
	};
}

interface HistogramProps {
	bins: number[];
	color: string;
	overlayBins?: number[];
	overlayColor?: string;
	xRange: [number, number];
	height?: number;
}
const Histogram: React.FC<HistogramProps> = ({ bins, color, overlayBins, overlayColor, xRange, height = 140 }) => {
	const maxY = Math.max(1, ...bins, ...(overlayBins ?? []));
	const W = 400;
	const H = height;
	const barW = W / bins.length;
	return (
		<svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
			{bins.map((b, i) => (
				<rect
					key={i}
					x={i * barW}
					y={H - (b / maxY) * (H - 20)}
					width={barW - 1}
					height={(b / maxY) * (H - 20)}
					fill={color}
					opacity={0.85}
				/>
			))}
			{overlayBins &&
				overlayBins.map((b, i) => (
					<rect
						key={`o${i}`}
						x={i * barW}
						y={H - (b / maxY) * (H - 20)}
						width={barW - 1}
						height={(b / maxY) * (H - 20)}
						fill={overlayColor ?? "#f59e0b"}
						opacity={0.5}
					/>
				))}
			<line x1={0} y1={H - 1} x2={W} y2={H - 1} stroke="#3f3f46" strokeWidth={1} />
			<text x={4} y={H - 4} fontFamily="monospace" fontSize={9} fill="#71717a">
				{xRange[0].toFixed(1)}
			</text>
			<text x={W - 20} y={H - 4} fontFamily="monospace" fontSize={9} fill="#71717a">
				{xRange[1].toFixed(1)}
			</text>
		</svg>
	);
};

type Stage = 1 | 2 | 3 | 4;

const Stage2: React.FC<{ raw: number[]; seed: number }> = ({ raw, seed }) => {
	const rotated = useMemo(() => randomRotation(raw.length, seed).apply(raw), [raw, seed]);
	const rawBins = useMemo(() => histogram(raw, 50, [-12, 12]), [raw]);
	const rotBins = useMemo(() => histogram(rotated, 50, [-12, 12]), [rotated]);
	const maxAbsRaw = Math.max(...raw.map(Math.abs));
	const maxAbsRot = Math.max(...rotated.map(Math.abs));
	return (
		<SchematicCard title="STAGE 2 · RANDOM ROTATION" status="TURBOQUANT">
			<div className="flex flex-col gap-3">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					Apply a random orthogonal rotation R. Norms and inner products are preserved, but the coordinate
					distribution is homogenised — outliers smeared across all channels, marginals converging toward a
					known Beta-like shape. The quantizer now has an <em>analytic</em> target distribution.
				</p>
				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div>
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1">
							Before rotation
						</div>
						<Histogram bins={rawBins} color="#ef4444" xRange={[-12, 12]} />
					</div>
					<div>
						<div className="text-[10px] font-mono uppercase tracking-widest text-zinc-500 mb-1">
							After rotation
						</div>
						<Histogram bins={rotBins} color="#6366f1" xRange={[-12, 12]} />
					</div>
				</div>
				<div className="grid grid-cols-2 gap-3">
					<DataReadout label="max |x| raw" value={maxAbsRaw.toFixed(2)} />
					<DataReadout label="max |x| rotated" value={maxAbsRot.toFixed(2)} />
				</div>
			</div>
		</SchematicCard>
	);
};

const Stage1: React.FC<{ raw: number[] }> = ({ raw }) => {
	const bins = useMemo(() => histogram(raw, 50, [-12, 12]), [raw]);
	const maxAbs = useMemo(() => Math.max(...raw.map(Math.abs)), [raw]);
	return (
		<SchematicCard title="STAGE 1 · RAW COORDINATE DISTRIBUTION" status="PROBLEM">
			<div className="flex flex-col gap-3">
				<p className="text-xs font-mono text-zinc-400 leading-relaxed">
					A synthetic KV-cache-like embedding. A small fraction of channels carry outsized magnitudes — the
					classic "outlier" problem. Uniform quantization has to stretch its bins to cover them, wasting bits
					on the common small values.
				</p>
				<Histogram bins={bins} color="#ef4444" xRange={[-12, 12]} />
				<div className="grid grid-cols-3 gap-3">
					<DataReadout label="dim" value={String(raw.length)} />
					<DataReadout label="max |x|" value={maxAbs.toFixed(2)} />
					<DataReadout
						label="std"
						value={Math.sqrt(raw.reduce((a, b) => a + b * b, 0) / raw.length).toFixed(2)}
					/>
				</div>
			</div>
		</SchematicCard>
	);
};

const TurboQuantSimulation: React.FC = () => {
	const [stage, setStage] = useState<Stage>(1);
	const [dim, setDim] = useState<number>(128);
	const [outlier, setOutlier] = useState<number>(1);
	const [seed, setSeed] = useState<number>(42);

	const raw = useMemo(() => synthVector(dim, outlier, seed), [dim, outlier, seed]);

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<div className="flex flex-wrap gap-2">
				<SchematicButton onClick={() => setStage(1)} icon={Activity} active={stage === 1}>
					1 · Raw
				</SchematicButton>
				<SchematicButton onClick={() => setStage(2)} icon={Shuffle} active={stage === 2}>
					2 · Rotate
				</SchematicButton>
				<SchematicButton onClick={() => setStage(3)} icon={Sliders} active={stage === 3}>
					3 · Quantize
				</SchematicButton>
				<SchematicButton onClick={() => setStage(4)} icon={Target} active={stage === 4}>
					4 · Polar
				</SchematicButton>
			</div>

			<SchematicCard title="CONTROLS">
				<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
					<label className="flex flex-col gap-1">
						<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">Dimension</span>
						<select
							className="bg-zinc-900 border border-zinc-700 text-zinc-200 font-mono text-xs px-2 py-1"
							value={dim}
							onChange={(e) => setDim(Number(e.target.value))}
						>
							<option value={64}>64</option>
							<option value={128}>128</option>
							<option value={256}>256</option>
						</select>
					</label>
					<label className="flex flex-col gap-1">
						<span className="text-[10px] font-mono uppercase tracking-widest text-zinc-500">
							Outlier Intensity: {outlier.toFixed(1)}
						</span>
						<input
							type="range"
							min={0}
							max={2}
							step={0.1}
							value={outlier}
							onChange={(e) => setOutlier(Number(e.target.value))}
						/>
					</label>
					<SchematicButton onClick={() => setSeed((s) => s + 1)} icon={Shuffle}>
						Re-sample
					</SchematicButton>
				</div>
			</SchematicCard>

			{stage === 1 && <Stage1 raw={raw} />}
			{stage === 2 && <Stage2 raw={raw} seed={seed} />}
			{stage === 3 && <div className="text-zinc-500 font-mono text-sm">Stage 3 — implemented in Task 5.</div>}
			{stage === 4 && <div className="text-zinc-500 font-mono text-sm">Stage 4 — implemented in Task 6.</div>}
		</div>
	);
};

export default TurboQuantSimulation;
