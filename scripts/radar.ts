#!/usr/bin/env bun
/**
 * Weekly paper radar.
 *
 * Harvests candidates, drops what the catalogue already covers, ranks the rest
 * against docs/scan-rubric.md, and prints a markdown report. Writes no content
 * and opens no PR — the output is a decision, not a page.
 *
 *   bun scripts/radar.ts                 # last 7 days, report to stdout
 *   bun scripts/radar.ts --days 14
 *   bun scripts/radar.ts --top 8
 *   bun scripts/radar.ts --out report.md
 *
 * Needs OPENAI_API_KEY; OPENAI_API_URL and PAPERLENS_MODEL are optional.
 */

import { catalogue, similarity, type Entry } from "./catalogue";
import { fromHuggingFace, fromArxiv, merge, type Candidate } from "./lib/sources";
import { askJson, assertModelAvailable, model } from "./lib/llm";

const ARXIV_CATEGORIES = ["cs.LG", "cs.AI", "cs.CL"];

interface Ranked {
	id: string;
	title: string;
	searchDemand: number;
	simulatable: number;
	oneMechanism: number;
	overturns: number;
	fillsGap: number;
	total: number;
	reason: string;
	targetQuery: string;
	excluded?: string;
}

interface Verdict {
	shortlist: Ranked[];
	duplicates: { id: string; title: string; coveredBy: string; why: string }[];
	pick: { id: string; why: string; beatOut: string; targetQuery: string; shortName?: string; oneLine?: string };
}

const SCHEMA = {
	name: "paper_shortlist",
	schema: {
		type: "object",
		required: ["shortlist", "duplicates", "pick"],
		properties: {
			shortlist: {
				type: "array",
				items: {
					type: "object",
					required: [
						"id",
						"title",
						"searchDemand",
						"simulatable",
						"oneMechanism",
						"overturns",
						"fillsGap",
						"total",
						"reason",
						"targetQuery",
					],
					properties: {
						id: { type: "string" },
						title: { type: "string" },
						searchDemand: { type: "integer", minimum: 0, maximum: 3 },
						simulatable: { type: "integer", minimum: 0, maximum: 3 },
						oneMechanism: { type: "integer", minimum: 0, maximum: 3 },
						overturns: { type: "integer", minimum: 0, maximum: 3 },
						fillsGap: { type: "integer", minimum: 0, maximum: 3 },
						total: { type: "integer", minimum: 0, maximum: 18 },
						reason: { type: "string", description: "the single biggest reason this did not beat the pick" },
						targetQuery: { type: "string" },
					},
				},
			},
			duplicates: {
				type: "array",
				items: {
					type: "object",
					required: ["id", "title", "coveredBy", "why"],
					properties: {
						id: { type: "string" },
						title: { type: "string" },
						coveredBy: { type: "string" },
						why: { type: "string" },
					},
				},
			},
			pick: {
				type: "object",
				required: ["id", "why", "beatOut", "targetQuery"],
				properties: {
					id: { type: "string" },
					why: { type: "string" },
					beatOut: { type: "string" },
					targetQuery: { type: "string" },
				},
			},
		},
	},
};

function arg(name: string, fallback?: string): string | undefined {
	const i = Bun.argv.indexOf(`--${name}`);
	return i === -1 ? fallback : Bun.argv[i + 1];
}

/** ISO week, e.g. 2026-W35 — the same label the HuggingFace weekly pages use. */
export function isoWeek(d = new Date()): string {
	const t = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
	// Thursday of the current week determines the ISO year.
	t.setUTCDate(t.getUTCDate() + 4 - (t.getUTCDay() || 7));
	const yearStart = new Date(Date.UTC(t.getUTCFullYear(), 0, 1));
	const week = Math.ceil(((t.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
	return `${t.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
}

/** GitHub shows roughly 90 characters before truncating; keep the suffix if it fits. */
export function clampTitle(head: string, suffix: string, max = 90): string {
	if (head.length + suffix.length <= max) return head + suffix;
	if (head.length <= max) return head;
	return head.slice(0, max - 1).trimEnd() + "…";
}

function catalogueDigest(entries: Entry[]): string {
	return entries
		.map((e) => `- ${e.slug} | ${e.title} | tags: ${e.tags.join(", ") || "—"} | arxiv: ${e.arxiv.join(", ") || "—"}`)
		.join("\n");
}

async function main() {
	const days = Number(arg("days", "7"));
	const top = Number(arg("top", "6"));
	const out = arg("out");
	const titleOut = arg("title-out");

	await assertModelAvailable();
	console.error(`# radar · last ${days} days · model ${model()}`);

	const entries = await catalogue();
	const covered = new Set(entries.flatMap((e) => e.arxiv));

	console.error("harvesting…");
	const [hf, ax] = await Promise.all([fromHuggingFace(days), fromArxiv(ARXIV_CATEGORIES)]);
	const all = merge(hf, ax);
	console.error(`  huggingface: ${hf.length}  arxiv: ${ax.length}  merged: ${all.length}`);

	// Exact ID matches are a hard drop and never reach the model.
	const hardDropped = all.filter((c) => covered.has(c.id));
	let candidates = all.filter((c) => !covered.has(c.id));

	// Keep the prompt affordable: everything that trended, plus the most
	// title-similar of the rest (those are where near-duplicates hide).
	const scored = candidates.map((c) => ({
		c,
		best: entries.reduce((m, e) => Math.max(m, similarity(c.title, e)), 0),
	}));
	candidates = scored
		.sort((a, b) => (b.c.upvotes ?? 0) - (a.c.upvotes ?? 0) || b.best - a.best)
		.slice(0, 40)
		.map((s) => s.c);

	console.error(`  ${hardDropped.length} already covered by ID, ${candidates.length} sent for ranking`);
	if (candidates.length === 0) {
		console.log("_No new candidates this week._");
		return;
	}

	const rubric = await Bun.file("docs/scan-rubric.md").text();

	const verdict = await askJson<Verdict>({
		system:
			"You select research papers for PaperLens, a site of interactive explainers. " +
			"You apply the given rubric literally and you never invent facts about a paper " +
			"beyond the abstract you are shown.",
		user: [
			"# Rubric",
			rubric,
			"",
			"# The catalogue already covers these entries",
			catalogueDigest(entries),
			"",
			"# Candidates",
			candidates
				.map(
					(c) =>
						`## ${c.id} — ${c.title}\n` +
						`upvotes: ${c.upvotes ?? 0} | published: ${c.published} | sources: ${c.sources.join(", ")}\n` +
						`${c.abstract.slice(0, 600)}`,
				)
				.join("\n\n"),
			"",
			"# Task",
			`Apply the rubric. Return the top ${top} candidates by total score in "shortlist",`,
			'ordered best first. Put anything whose core mechanism an existing entry already',
			'covers into "duplicates" instead, naming the entry slug in coveredBy. Apply the',
			"hard exclusions. Then name the single best paper to publish in \"pick\".",
			"",
			'For "targetQuery", give the actual search query the entry would be trying to win.',
			'For "shortName", give the artefact name on its own — "EchoWM", not the full title.',
			'For "oneLine", say what it does in under 60 lowercase characters, no trailing period.',
			"Score search demand on the evidence you can see: a named artefact, upvote count,",
			"a lab people follow. Be honest when a candidate is a system report.",
			"",
			'"reason" is the single biggest reason that paper did NOT beat your pick —',
			"a weakness, not a summary. For the pick itself, say why it won instead.",
			"",
			"Keep every \"reason\" to one sentence under 140 characters, and \"why\" under 400.",
			"Cap \"duplicates\" at 6 entries. Brevity matters: the reply must be complete,",
			"valid JSON, and a truncated reply is worse than a terse one.",
		].join("\n"),
		schema: SCHEMA,
		maxTokens: 14000,
	});

	// The model is asked for an entry slug in `coveredBy`, but it sometimes returns
	// a candidate's arXiv ID instead — which would claim an unpublished paper covers
	// another. Only keep duplicate claims that point at a real entry.
	const slugs = new Set(entries.map((e) => e.slug));
	const unverified = verdict.duplicates.filter((d) => !slugs.has(d.coveredBy));
	verdict.duplicates = verdict.duplicates.filter((d) => slugs.has(d.coveredBy));
	if (unverified.length) {
		console.error(`  ! dropped ${unverified.length} duplicate claim(s) naming something that is not an entry:`);
		for (const d of unverified) console.error(`      ${d.id} -> "${d.coveredBy}"`);
	}

	const picked = verdict.shortlist.find((s) => s.id === verdict.pick.id) ?? verdict.shortlist[0];
	const iso = new Date().toISOString().slice(0, 10);

	const lines: string[] = [
		`## Paper radar · ${iso}`,
		"",
		`Scanned ${all.length} papers from the last ${days} days ` +
			`(HuggingFace daily + arXiv ${ARXIV_CATEGORIES.join(", ")}). ` +
			`${hardDropped.length} already covered by arXiv ID.`,
		"",
		"| # | arXiv | Paper | Search×2 | Sim | Mech | Overturns | Gap | Total |",
		"|---|---|---|---|---|---|---|---|---|",
		...verdict.shortlist.map(
			(s, i) =>
				`| ${i + 1} | [${s.id}](https://arxiv.org/abs/${s.id}) | ${s.title} | ${s.searchDemand * 2} | ` +
				`${s.simulatable} | ${s.oneMechanism} | ${s.overturns} | ${s.fillsGap} | **${s.total}** |`,
		),
		"",
		"### Recommended",
		"",
		`**[${picked?.title ?? verdict.pick.id}](https://arxiv.org/abs/${verdict.pick.id})** (\`${verdict.pick.id}\`)`,
		"",
		verdict.pick.why,
		"",
		`*Target query:* ${verdict.pick.targetQuery}`,
		`*Beat out:* ${verdict.pick.beatOut}`,
		"",
	];

	if (verdict.duplicates.length) {
		lines.push(
			"### Already covered",
			"",
			...verdict.duplicates.map((d) => `- \`${d.id}\` ${d.title} — covered by \`${d.coveredBy}\`: ${d.why}`),
			"",
		);
	}

	if (unverified.length) {
		lines.push(
			"### Flagged, not verified",
			"",
			"These were called duplicates of something that is not an entry in the catalogue, so the claim was dropped:",
			"",
			...unverified.map((d) => `- \`${d.id}\` ${d.title} — claimed covered by \`${d.coveredBy}\``),
			"",
		);
	}

	lines.push(
		"### Why the others lost",
		"",
		...verdict.shortlist.filter((s) => s.id !== verdict.pick.id).map((s) => `- \`${s.id}\` — ${s.reason}`),
		"",
		"---",
		"",
		"Reply `/build <arxiv-id>` on this issue to publish one.",
	);

	// "Paper radar" tells a reader nothing. Lead with the recommendation.
	const shortName =
		verdict.pick.shortName?.trim() ||
		(picked?.title ?? "").split(/[:—-]/)[0].trim() ||
		verdict.pick.id;
	const oneLine = verdict.pick.oneLine?.trim().replace(/\.$/, "") ?? "";
	const head = oneLine ? `${shortName} — ${oneLine}` : shortName;
	const title = clampTitle(
		`Radar ${isoWeek()} · ${head}`,
		` · ${verdict.shortlist.length} of ${all.length}`,
	);

	const report = lines.join("\n");
	if (out) {
		await Bun.write(out, report);
		console.error(`wrote ${out}`);
	}
	if (titleOut) {
		await Bun.write(titleOut, title);
		console.error(`wrote ${titleOut}: ${title}`);
	}
	console.error(`\ntitle: ${title}`);
	console.log(report);
}

if (import.meta.main) {
	main().catch((err) => {
		console.error(err instanceof Error ? err.message : String(err));
		process.exit(1);
	});
}
