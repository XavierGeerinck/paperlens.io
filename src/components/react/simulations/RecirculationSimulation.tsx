import React, { useEffect, useMemo, useState } from "react";
import { Play, Pause, RotateCcw, CornerLeftUp } from "lucide-react";
import { SchematicCard, SchematicButton, DataReadout, TechBadge } from "../SketchElements";

// ---------------------------------------------------------------------------
// A toy of the mechanism, not of the network.
//
// One ambiguous token carries a belief across three senses. Two forces act on
// it as it climbs the stack:
//
//   * the lexical prior, which dominates early and fades as context lands;
//   * a commitment pressure that grows with depth — later layers sharpen toward
//     an output rather than reconsidering.
//
// The disambiguating context arrives on a sigmoid. If it lands after commitment
// pressure has taken hold, the representation never catches up: it finishes the
// stack still lagging what the context implies. That gap is the depth bound the
// paper describes — "a race in which the model's response generation can
// outpace the model's internal semantic convergence."
//
// Recirculation leaks a deep layer's activation back to a shallow layer and
// reruns that span, buying a second climb through the region where the layers
// can still move.
//
// The mixing rule and the layer pairs are the paper's. The belief dynamics are
// invented to make the lag visible, and the effect sizes here are the toy's own
// — the paper's measured numbers are in the entry above.
// ---------------------------------------------------------------------------

const SENSES = [
	{ key: "finance", label: "money / ATM", color: "var(--green)" },
	{ key: "river", label: "river bank", color: "var(--orange)" },
	{ key: "tilt", label: "to bank / tilt", color: "var(--purple)" },
] as const;

const TRUE_SENSE = 0;
const PRIOR = [0.22, 0.62, 0.16];

/** Gemma 3 depths, with the source→destination pairs the paper reports. */
const PRESETS = {
	"1B": { depth: 26, source: 11, dest: 4 },
	"4B": { depth: 34, source: 18, dest: 9 },
	"12B": { depth: 48, source: 35, dest: 16 },
} as const;

type PresetKey = keyof typeof PRESETS;

// Dynamics constants, chosen so the lag is legible at every model size.
const SHARP = 4.0; // commitment pressure at the top of the stack
const SHARP_POW = 2.5; // how late that pressure ramps in
const EVIDENCE = 0.8; // pull toward the context-appropriate sense
const PRIOR_DRIVE = 2.8; // pull toward the lexical prior
const DECAY = 0.3;
const ETA = 0.25;

function softmax(v: number[]): number[] {
	const m = Math.max(...v);
	const e = v.map((x) => Math.exp(x - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((x) => x / s);
}

function l2(v: number[]): number {
	return Math.sqrt(v.reduce((a, b) => a + b * b, 0));
}

interface Params {
	depth: number;
	source: number;
	dest: number;
	alpha: number;
	evidenceFrac: number;
}

interface Trace {
	/** belief after each layer on the first pass */
	pass1: number[][];
	/** belief after each layer of the rerun, indexed from dest (null if no leak) */
	pass2: number[][] | null;
	/** what the context implies at each layer */
	context: number[];
	final: number[];
	confidence: number;
	/** the belief the destination layer was handed after mixing */
	mixedConfidence: number | null;
}

function simulate(p: Params, recirculate: boolean): Trace {
	const center = p.evidenceFrac * p.depth;
	const spread = Math.max(1.5, p.depth * 0.1);
	const ctx = (l: number) => 1 / (1 + Math.exp(-(l - center) / spread));
	const sharp = (l: number) => SHARP * Math.pow(l / p.depth, SHARP_POW);

	const logits: number[][] = Array.from({ length: p.depth }, () => [0, 0, 0]);

	const span = (start: number[], from: number, to: number) => {
		let b = [...start];
		const out: number[][] = [];
		for (let l = from; l <= to; l++) {
			const cur = softmax(b);
			const c = ctx(l);
			const drive = cur.map(
				(ci, i) =>
					sharp(l) * (ci - 1 / 3) + // sharpen whichever reading is ahead
					EVIDENCE * c * (i === TRUE_SENSE ? 1 : 0) + // context, arriving with depth
					PRIOR_DRIVE * (1 - c) * PRIOR[i], // lexical prior, fading
			);
			b = b.map((x, i) => x + ETA * (drive[i] - DECAY * x));
			logits[l - 1] = [...b];
			out.push(softmax(b));
		}
		return { b, out };
	};

	const first = span([0, 0, 0], 1, p.depth);
	const pass1 = first.out;
	let final = softmax(first.b);
	let pass2: number[][] | null = null;
	let mixedConfidence: number | null = null;

	if (recirculate && p.alpha > 0) {
		// z_d <- alpha * f(z_s) + beta * z_d, alpha + beta = 1,
		// where f() rescales the source to the destination's L2 norm.
		const zs = logits[p.source - 1];
		const zd = logits[p.dest - 1];
		const scale = l2(zd) / Math.max(l2(zs), 1e-6);
		const mixed = zs.map((x, i) => p.alpha * (x * scale) + (1 - p.alpha) * zd[i]);
		mixedConfidence = softmax(mixed)[TRUE_SENSE];

		const second = span(mixed, p.dest + 1, p.depth);
		pass2 = second.out;
		final = softmax(second.b);
	}

	return {
		pass1,
		pass2,
		context: Array.from({ length: p.depth }, (_, i) => ctx(i + 1)),
		final,
		confidence: final[TRUE_SENSE],
		mixedConfidence,
	};
}

// ---------------------------------------------------------------------------

const Stack: React.FC<{
	rows: number[][];
	depth: number;
	upTo: number;
	title: string;
	accent: string;
	source?: number;
	dest?: number;
}> = ({ rows, depth, upTo, title, accent, source, dest }) => {
	const rowH = Math.max(5, Math.min(11, Math.round(300 / depth)));

	return (
		<div className="flex-1 min-w-0">
			<div className="flex items-baseline justify-between mb-2 gap-2">
				<span className="text-[11.5px] font-mono font-semibold truncate" style={{ color: accent }}>
					{title}
				</span>
				<span className="text-[10.5px] font-mono text-mute tabular-nums whitespace-nowrap">{depth}L</span>
			</div>

			<div className="relative border border-bg2 rounded bg-bg0h p-1.5">
				{source !== undefined && dest !== undefined && (
					<svg
						className="absolute left-0 top-0 h-full pointer-events-none"
						width="18"
						aria-hidden="true"
						style={{ overflow: "visible" }}
					>
						{(() => {
							const pad = 6;
							const ys = pad + (source - 0.5) * rowH;
							const yd = pad + (dest - 0.5) * rowH;
							return (
								<g stroke="var(--aqua)" fill="none" strokeWidth="1.2">
									<path d={`M 2 ${ys} L -7 ${ys} L -7 ${yd} L 2 ${yd}`} />
									<path d={`M 2 ${yd} l -4 -3 M 2 ${yd} l -4 3`} strokeWidth="1.4" />
									<circle cx="2" cy={ys} r="1.8" fill="var(--aqua)" stroke="none" />
								</g>
							);
						})()}
					</svg>
				)}

				<div className="flex flex-col" style={{ gap: 1 }}>
					{rows.slice(0, depth).map((dist, i) => {
						const active = i < upTo;
						const marked = (source !== undefined && i + 1 === source) || (dest !== undefined && i + 1 === dest);
						return (
							<div
								key={i}
								className="flex w-full overflow-hidden rounded-[1px]"
								style={{
									height: rowH,
									opacity: active ? 1 : 0.12,
									outline: marked ? "1px solid var(--aqua)" : undefined,
									outlineOffset: 1,
								}}
								title={`layer ${i + 1}`}
							>
								{dist.map((share, si) => (
									<div
										key={si}
										style={{
											width: `${share * 100}%`,
											background: SENSES[si].color,
											opacity: si === TRUE_SENSE ? 0.95 : 0.5,
										}}
									/>
								))}
							</div>
						);
					})}
				</div>
			</div>
		</div>
	);
};

/**
 * The teaching device: what the context implies, against what each stack's
 * representation actually encodes, layer by layer. The vertical distance at the
 * right-hand edge is the depth bound.
 */
const LagChart: React.FC<{ plain: Trace; recirc: Trace; p: Params }> = ({ plain, recirc, p }) => {
	const W = 300;
	const H = 120;
	const x = (l: number) => ((l - 1) / (p.depth - 1)) * W;
	const y = (v: number) => H - v * H;

	const line = (vals: number[], from = 1) =>
		vals.map((v, i) => `${i === 0 ? "M" : "L"} ${x(from + i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

	const ctxLine = line(recirc.context);
	const plainLine = line(plain.pass1.map((d) => d[TRUE_SENSE]));
	const rerun = recirc.pass2 ? line(recirc.pass2.map((d) => d[TRUE_SENSE]), p.dest + 1) : null;
	const leakY = recirc.mixedConfidence;

	return (
		<div>
			<svg viewBox={`0 0 ${W} ${H + 18}`} className="w-full" role="img"
				aria-label="Context-implied reading against what each stack encodes, by layer">
				<line x1="0" y1={H} x2={W} y2={H} stroke="var(--bg3)" strokeWidth="1" />
				<line x1="0" y1="0" x2="0" y2={H} stroke="var(--bg3)" strokeWidth="1" />

				{/* the lag at the output */}
				<line
					x1={W - 1}
					y1={y(plain.confidence)}
					x2={W - 1}
					y2={y(recirc.context[p.depth - 1])}
					stroke="var(--red)"
					strokeWidth="4"
					opacity="0.35"
				/>

				<path d={ctxLine} fill="none" stroke="var(--orange)" strokeWidth="1.4" strokeDasharray="3 3" />
				<path d={plainLine} fill="none" stroke="var(--fg2)" strokeWidth="1.6" />
				{rerun && <path d={rerun} fill="none" stroke="var(--aqua)" strokeWidth="1.8" />}
				{rerun && leakY !== null && (
					<>
						<line
							x1={x(p.source)}
							y1={y(plain.pass1[p.source - 1][TRUE_SENSE])}
							x2={x(p.dest)}
							y2={y(leakY)}
							stroke="var(--aqua)"
							strokeWidth="1"
							strokeDasharray="2 2"
							opacity="0.7"
						/>
						<circle cx={x(p.dest)} cy={y(leakY)} r="2.5" fill="var(--aqua)" />
					</>
				)}

				<text x="2" y={H + 13} fontSize="8.5" fill="var(--fg4)" fontFamily="monospace">
					layer 1
				</text>
				<text x={W} y={H + 13} fontSize="8.5" fill="var(--fg4)" fontFamily="monospace" textAnchor="end">
					layer {p.depth}
				</text>
			</svg>
			<div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10.5px] font-mono mt-1">
				<span style={{ color: "var(--orange)" }}>--- context implies</span>
				<span style={{ color: "var(--fg2)" }}>— frozen</span>
				<span style={{ color: "var(--aqua)" }}>— rerun span</span>
				<span style={{ color: "var(--red)" }}>▮ residual lag</span>
			</div>
		</div>
	);
};

// ---------------------------------------------------------------------------

const RecirculationSimulation: React.FC = () => {
	const [preset, setPreset] = useState<PresetKey>("4B");
	const [alpha, setAlpha] = useState(0.15);
	const [evidenceFrac, setEvidenceFrac] = useState(0.4);
	const [playing, setPlaying] = useState(true);
	const [tick, setTick] = useState(0);

	const { depth, source, dest } = PRESETS[preset];
	const p: Params = { depth, source, dest, alpha, evidenceFrac };

	const plain = useMemo(() => simulate(p, false), [depth, source, dest, evidenceFrac]);
	const recirc = useMemo(() => simulate(p, true), [depth, source, dest, evidenceFrac, alpha]);

	useEffect(() => {
		if (typeof window === "undefined") return;
		if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
			setPlaying(false);
			setTick(depth);
		}
	}, [depth]);

	useEffect(() => {
		if (!playing) return;
		const id = window.setInterval(() => setTick((t) => (t >= depth ? 0 : t + 1)), 90);
		return () => window.clearInterval(id);
	}, [playing, depth]);

	useEffect(() => setTick(depth), [preset]);

	const upTo = playing ? tick : depth;

	const implied = recirc.context[depth - 1];
	const gain = recirc.confidence - plain.confidence;
	const lagBefore = implied - plain.confidence;
	const lagAfter = implied - recirc.confidence;
	const closed = lagBefore > 1e-6 ? (1 - lagAfter / lagBefore) * 100 : 0;

	// The rows shown for the recirculated stack: first pass below the destination,
	// the rerun above it.
	const recircRows = useMemo(() => {
		const rows = recirc.pass1.map((r) => [...r]);
		if (recirc.pass2) recirc.pass2.forEach((r, i) => (rows[dest + i] = r));
		return rows;
	}, [recirc, dest]);

	return (
		<div className="flex flex-col gap-4">
			<SchematicCard
				title="RECIRCULATION · frozen stack, one extra iteration"
				status={`gemma3 ${preset} · source L${source} → dest L${dest}`}
			>
				<p className="text-[12.5px] text-ink2 leading-relaxed mb-4">
					One ambiguous token — <span className="text-ink font-semibold">“bank”</span> in{" "}
					<span className="text-ink1">“he went to the bank to withdraw…”</span>. Each bar is one layer's belief
					across the three readings. The disambiguating word lands late, and by then the upper layers are
					sharpening toward an answer rather than reconsidering.
				</p>

				<div className="flex gap-5 items-start pl-3">
					<Stack rows={plain.pass1} depth={depth} upTo={upTo} title="frozen forward pass" accent="var(--fg2)" />
					<Stack
						rows={recircRows}
						depth={depth}
						upTo={upTo}
						title="+ recirculation"
						accent="var(--aqua)"
						source={source}
						dest={dest}
					/>
				</div>

				<div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[11px] font-mono">
					{SENSES.map((s, i) => (
						<span key={s.key} className="flex items-center gap-1.5">
							<span
								className="inline-block w-2.5 h-2.5 rounded-[1px]"
								style={{ background: s.color, opacity: i === TRUE_SENSE ? 0.95 : 0.5 }}
							/>
							<span className={i === TRUE_SENSE ? "text-ink1" : "text-mute"}>{s.label}</span>
						</span>
					))}
					<span className="flex items-center gap-1.5 text-mute">
						<CornerLeftUp size={12} style={{ color: "var(--aqua)" }} /> feedback path
					</span>
				</div>
			</SchematicCard>

			<div className="grid gap-4 md:grid-cols-2">
				<SchematicCard title="CONTROLS">
					<div className="flex flex-col gap-4">
						<div>
							<span className="text-[11px] font-mono text-mute block mb-1.5">
								model · layer pair from the paper
							</span>
							<div className="flex gap-2 flex-wrap">
								{(Object.keys(PRESETS) as PresetKey[]).map((k) => (
									<SchematicButton key={k} active={preset === k} onClick={() => setPreset(k)}>
										{k}
									</SchematicButton>
								))}
							</div>
						</div>

						<label className="block">
							<span className="text-[11px] font-mono text-mute flex justify-between">
								<span>α · leak fraction</span>
								<span className="text-ink1 tabular-nums">{alpha.toFixed(2)}</span>
							</span>
							<input
								type="range"
								min={0}
								max={0.4}
								step={0.01}
								value={alpha}
								onChange={(e) => setAlpha(parseFloat(e.target.value))}
								className="w-full accent-mint-400 mt-1"
							/>
							<span className="text-[10.5px] font-mono text-mute">
								paper sweeps 0.04 – 0.16, default 0.15
							</span>
						</label>

						<label className="block">
							<span className="text-[11px] font-mono text-mute flex justify-between">
								<span>context lands at layer</span>
								<span className="text-ink1 tabular-nums">
									{Math.round(evidenceFrac * depth)} / {depth}
								</span>
							</span>
							<input
								type="range"
								min={0.2}
								max={0.8}
								step={0.02}
								value={evidenceFrac}
								onChange={(e) => setEvidenceFrac(parseFloat(e.target.value))}
								className="w-full accent-mint-400 mt-1"
							/>
						</label>

						<div className="flex gap-2">
							<SchematicButton onClick={() => setPlaying((v) => !v)} icon={playing ? Pause : Play}>
								{playing ? "pause" : "play"}
							</SchematicButton>
							<SchematicButton
								onClick={() => {
									setAlpha(0.15);
									setEvidenceFrac(0.4);
									setPreset("4B");
									setTick(0);
								}}
								icon={RotateCcw}
							>
								reset
							</SchematicButton>
						</div>
					</div>
				</SchematicCard>

				<SchematicCard title="CONTEXTUALIZATION LAG" status="toy model">
					<LagChart plain={plain} recirc={recirc} p={p} />

					<div className="grid grid-cols-2 gap-x-4 gap-y-3 mt-4 pt-3 border-t border-bg2">
						<DataReadout
							label="context implies"
							value={<span className="text-amber-400">{(implied * 100).toFixed(1)}%</span>}
						/>
						<DataReadout label="frozen reached" value={`${(plain.confidence * 100).toFixed(1)}%`} />
						<DataReadout
							label="recirculated reached"
							value={<span style={{ color: "var(--aqua)" }}>{(recirc.confidence * 100).toFixed(1)}%</span>}
						/>
						<DataReadout
							label="lag closed"
							value={
								<span style={{ color: closed >= 0 ? "var(--green)" : "var(--red)" }}>
									{closed >= 0 ? "+" : ""}
									{closed.toFixed(0)}%
								</span>
							}
						/>
						<DataReadout label="generation cost" value={<span className="text-mint-400">1.00×</span>} />
						<DataReadout
							label="prefill cost"
							value={
								<span className="text-amber-400">
									{(1 + (depth - dest) / depth).toFixed(2)}× · serial
								</span>
							}
						/>
					</div>
					<p className="text-[11px] font-mono text-mute mt-3">
						gain {gain >= 0 ? "+" : ""}
						{(gain * 100).toFixed(1)} pts — modest by construction, as it is in the paper
					</p>
				</SchematicCard>
			</div>

			<SchematicCard title="WHAT TO TRY">
				<ul className="text-[12.5px] text-ink2 leading-relaxed flex flex-col gap-2">
					<li>
						<TechBadge label="α = 0" /> — turn the leak off and the two stacks are identical. The dashed line
						shows what the context implies; the gap to the grey line at the right edge is the depth bound.
					</li>
					<li>
						<TechBadge label="context → late" /> — push the disambiguating word past the commitment ramp. The
						frozen stack finishes further behind, and recirculation has more to recover.
					</li>
					<li>
						<TechBadge label="1B → 12B" /> — deeper stacks put the source further from the destination, so one
						leak carries back more integrated context. The paper's largest perplexity win is also on 12B.
					</li>
					<li>
						<TechBadge label="watch the aqua line" /> — it drops at the destination layer, then climbs again.
						That second climb through the movable layers is the whole mechanism.
					</li>
				</ul>
				<p className="text-[11.5px] text-mute mt-4 leading-relaxed">
					A three-sense abstraction, not a transformer. The mixing rule and the layer pairs are the paper's; the
					belief dynamics are invented to make the depth bound visible, and the effect sizes are the toy's own.
					The measured results — 23% perplexity reduction, 20.9% GSM8k error reduction at pass@128 — are in the
					entry above. The toy does not model the cost of leaking too hard, which is why the paper's sweep stops
					at α = 0.16 and this slider does not show a penalty beyond it.
				</p>
			</SchematicCard>
		</div>
	);
};

export default RecirculationSimulation;
