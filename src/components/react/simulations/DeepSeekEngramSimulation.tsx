import { useMemo } from "react";
import type { FC } from "react";
import { Database, Hash, ShieldCheck, SlidersHorizontal } from "lucide-react";
import { useSimulation } from "../../../hooks/useSimulation";
import { SchematicCard, SchematicButton } from "../SketchElements";

type EngramSimState = {
	// Controls
	distinctNgrams: number;
	tableSize: number;
	hashHeads: number;
	contextAlignment: number; // [-1, 1]
	temperature: number;

	// Metrics
	avgHeadCollisionRate: number; // [0, 1]
	ambiguityRate: number; // [0, 1] collisions across all heads (K-tuple duplicates)
	gate: number; // [0, 1]
	effectiveMemory: number; // [0, 1]
};

function clamp(value: number, min: number, max: number) {
	return Math.max(min, Math.min(max, value));
}

function sigmoid(x: number) {
	return 1 / (1 + Math.exp(-x));
}

function randomU32() {
	// 32-bit unsigned int
	return (Math.random() * 0xffffffff) >>> 0;
}

function mix32(x: number) {
	// A tiny integer mixer (not crypto; just for simulation)
	x = Math.imul(x ^ (x >>> 16), 0x7feb352d);
	x = Math.imul(x ^ (x >>> 15), 0x846ca68b);
	x = x ^ (x >>> 16);
	return x >>> 0;
}

function hashToIndex(id: number, seed: number, tableSize: number) {
	const mixed = mix32(id ^ seed);
	return mixed % tableSize;
}

function renderMiniGraph(history: number[] | undefined, color: string) {
	const values = history && history.length > 1 ? history : [0, 0];
	const width = 220;
	const height = 64;
	const paddingX = 4;
	const paddingY = 4;
	const min = 0;
	const max = 1;

	const points = values
		.map((val, i) => {
			const x = paddingX + (i / (values.length - 1)) * (width - 2 * paddingX);
			const y =
				height -
				paddingY -
				((clamp(val, min, max) - min) / (max - min)) * (height - 2 * paddingY);
			return `${x},${y}`;
		})
		.join(" ");

	return (
		<svg
			viewBox={`0 0 ${width} ${height}`}
			className="w-full h-16 overflow-visible"
			preserveAspectRatio="none"
		>
			<title>Metric sparkline</title>
			<polyline
				points={points}
				fill="none"
				stroke={color}
				strokeWidth="2"
				strokeLinecap="round"
				strokeLinejoin="round"
			/>
		</svg>
	);
}

const DeepSeekEngramSimulation: FC = () => {
	const { isRunning, state, history, start, stop, reset, update } =
		useSimulation<EngramSimState>({
			initialState: {
				distinctNgrams: 4096,
				tableSize: 131071,
				hashHeads: 8,
				contextAlignment: 0.6,
				temperature: 0.9,
				avgHeadCollisionRate: 0,
				ambiguityRate: 0,
				gate: 0,
				effectiveMemory: 0,
			},
			onTick: (prev) => {
				const distinctNgrams = clamp(
					Math.floor(prev.distinctNgrams),
					64,
					20000,
				);
				const tableSize = clamp(Math.floor(prev.tableSize), 1024, 2000000);
				const hashHeads = clamp(Math.floor(prev.hashHeads), 1, 32);
				const contextAlignment = clamp(prev.contextAlignment, -1, 1);
				const temperature = clamp(prev.temperature, 0.1, 2.5);

				// Build a set of distinct "ngrams" (just unique integer IDs for the simulation)
				const ids = new Set<number>();
				while (ids.size < distinctNgrams) ids.add(randomU32());
				const idList = Array.from(ids);

				// Per-head collision estimate and K-tuple ambiguity estimate
				const uniquePerHead: Array<Set<number>> = Array.from(
					{ length: hashHeads },
					() => new Set<number>(),
				);
				const tupleCounts = new Map<string, number>();

				for (const id of idList) {
					const tuple: number[] = [];
					for (let k = 0; k < hashHeads; k++) {
						const idx = hashToIndex(id, 1337 + 97 * k, tableSize);
						uniquePerHead[k].add(idx);
						tuple.push(idx);
					}
					const key = tuple.join("|");
					tupleCounts.set(key, (tupleCounts.get(key) ?? 0) + 1);
				}

				let avgHeadCollisionRate = 0;
				for (let k = 0; k < hashHeads; k++) {
					const unique = uniquePerHead[k].size;
					avgHeadCollisionRate += 1 - unique / distinctNgrams;
				}
				avgHeadCollisionRate /= hashHeads;

				let ambiguous = 0;
				for (const count of tupleCounts.values()) {
					if (count > 1) ambiguous += count;
				}
				const ambiguityRate = ambiguous / distinctNgrams;

				// Gate models whether the current hidden state agrees with memory.
				// Add small noise so you can see dynamics while running.
				const noise = (Math.random() * 2 - 1) * 0.15;
				const gate = sigmoid((contextAlignment + noise) / temperature);

				// Effective memory signal: gated and discounted by ambiguous lookups.
				const effectiveMemory = clamp(gate * (1 - ambiguityRate), 0, 1);

				return {
					distinctNgrams,
					tableSize,
					hashHeads,
					contextAlignment,
					temperature,
					avgHeadCollisionRate,
					ambiguityRate,
					gate,
					effectiveMemory,
				};
			},
			tickRate: 180,
		});

	const tableSizeLabel = useMemo(() => {
		if (state.tableSize >= 1_000_000)
			return `${(state.tableSize / 1_000_000).toFixed(2)}M`;
		if (state.tableSize >= 1_000)
			return `${(state.tableSize / 1_000).toFixed(1)}k`;
		return `${state.tableSize}`;
	}, [state.tableSize]);

	return (
		<div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl">
			<SchematicCard title="ENGRAM_CONDITIONAL_MEMORY">
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<div className="space-y-4">
						<div className="flex items-center gap-2 text-indigo-300 font-mono text-xs uppercase border-b border-slate-800 pb-2">
							<SlidersHorizontal size={16} /> Controls
						</div>

						<div className="space-y-3">
							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
									<span>Distinct N-grams sampled</span>
									<span className="text-slate-300">{state.distinctNgrams}</span>
								</div>
								<input
									type="range"
									min={256}
									max={20000}
									step={256}
									value={state.distinctNgrams}
									onChange={(e) =>
										update({ distinctNgrams: Number(e.target.value) })
									}
									className="w-full accent-indigo-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
									<span>Embedding table slots (M)</span>
									<span className="text-slate-300">{tableSizeLabel}</span>
								</div>
								<input
									type="range"
									min={2048}
									max={2000000}
									step={2048}
									value={state.tableSize}
									onChange={(e) =>
										update({ tableSize: Number(e.target.value) })
									}
									className="w-full accent-indigo-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
									<span>Hash heads (K)</span>
									<span className="text-slate-300">{state.hashHeads}</span>
								</div>
								<input
									type="range"
									min={1}
									max={16}
									step={1}
									value={state.hashHeads}
									onChange={(e) =>
										update({ hashHeads: Number(e.target.value) })
									}
									className="w-full accent-indigo-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
									<span>Context alignment (Query · Key)</span>
									<span className="text-slate-300">
										{state.contextAlignment.toFixed(2)}
									</span>
								</div>
								<input
									type="range"
									min={-1}
									max={1}
									step={0.02}
									value={state.contextAlignment}
									onChange={(e) =>
										update({ contextAlignment: Number(e.target.value) })
									}
									className="w-full accent-indigo-500"
								/>
							</label>

							<label className="block">
								<div className="flex justify-between text-[10px] font-mono text-slate-500 uppercase mb-1">
									<span>Gate temperature</span>
									<span className="text-slate-300">
										{state.temperature.toFixed(2)}
									</span>
								</div>
								<input
									type="range"
									min={0.1}
									max={2.5}
									step={0.05}
									value={state.temperature}
									onChange={(e) =>
										update({ temperature: Number(e.target.value) })
									}
									className="w-full accent-indigo-500"
								/>
							</label>
						</div>

						<div className="grid grid-cols-2 gap-3 pt-2 border-t border-slate-800">
							<div className="p-3 bg-black/30 border border-slate-800 rounded">
								<div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 uppercase mb-1">
									<Hash size={14} /> Avg per-head collisions
								</div>
								<div className="text-xl font-mono text-slate-200">
									{(state.avgHeadCollisionRate * 100).toFixed(2)}%
								</div>
							</div>
							<div className="p-3 bg-black/30 border border-slate-800 rounded">
								<div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 uppercase mb-1">
									<Database size={14} /> K-tuple ambiguity
								</div>
								<div className="text-xl font-mono text-slate-200">
									{(state.ambiguityRate * 100).toFixed(2)}%
								</div>
							</div>
						</div>

						<div className="mt-3 p-3 bg-emerald-900/10 border border-emerald-500/20 rounded-lg">
							<h4 className="text-[10px] font-bold text-emerald-400 uppercase mb-1 flex items-center gap-2">
								<ShieldCheck size={14} /> What to try
							</h4>
							<p className="text-[11px] text-slate-400 leading-relaxed">
								Increase <span className="text-slate-200 font-bold">K</span>{" "}
								from 1→8: K-tuple ambiguity should fall fast.
							</p>
							<p className="text-[11px] text-slate-400 leading-relaxed">
								Decrease <span className="text-slate-200 font-bold">M</span>:
								per-head collisions rise; ambiguity rises.
							</p>
							<p className="text-[11px] text-slate-400 leading-relaxed">
								Lower{" "}
								<span className="text-slate-200 font-bold">alignment</span>: the
								gate suppresses memory.
							</p>
						</div>

						<div className="flex gap-4 mt-4">
							<SchematicButton onClick={isRunning ? stop : start}>
								{isRunning ? "HALT" : "START"}
							</SchematicButton>
							<SchematicButton onClick={reset} variant="secondary">
								RESET
							</SchematicButton>
						</div>
					</div>

					<div className="space-y-4">
						<div className="flex items-center gap-2 text-indigo-300 font-mono text-xs uppercase border-b border-slate-800 pb-2">
							<Database size={16} /> Metrics
						</div>

						<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
							<div className="bg-black/40 p-4 rounded border border-slate-800">
								<div className="text-[10px] font-mono text-slate-500 uppercase mb-2">
									Per-head collision rate
								</div>
								{renderMiniGraph(history.avgHeadCollisionRate, "#7bb7f7")}
							</div>
							<div className="bg-black/40 p-4 rounded border border-slate-800">
								<div className="text-[10px] font-mono text-slate-500 uppercase mb-2">
									K-tuple ambiguity rate
								</div>
								{renderMiniGraph(history.ambiguityRate, "#f5a623")}
							</div>
							<div className="bg-black/40 p-4 rounded border border-slate-800">
								<div className="text-[10px] font-mono text-slate-500 uppercase mb-2">
									Gate α (sigmoid)
								</div>
								{renderMiniGraph(history.gate, "#35d492")}
							</div>
							<div className="bg-black/40 p-4 rounded border border-slate-800">
								<div className="text-[10px] font-mono text-slate-500 uppercase mb-2">
									Effective memory signal
								</div>
								{renderMiniGraph(history.effectiveMemory, "#a992f6")}
							</div>
						</div>

						<div className="p-3 bg-blue-900/10 border border-blue-500/20 rounded-lg">
							<p className="text-[11px] text-slate-400 leading-relaxed">
								Engram reduces lookup noise by (1) using multiple hash heads
								(collisions become less likely to match across all heads) and
								(2) gating the retrieved memory by context.
							</p>
						</div>
					</div>
				</div>
			</SchematicCard>
		</div>
	);
};

export default DeepSeekEngramSimulation;
