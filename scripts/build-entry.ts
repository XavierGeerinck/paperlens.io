#!/usr/bin/env bun
/**
 * Turn one arXiv ID into a PaperLens entry plus its simulation.
 *
 *   bun scripts/build-entry.ts 2608.17981
 *   bun scripts/build-entry.ts 2608.17981 --retries 4
 *
 * The paper record comes from the arXiv API, never from a title or a trending
 * blurb. Generated files are written, `bun run build` runs, and any failure is
 * fed back for repair until the build is clean or the retry budget runs out.
 *
 * Git and pull requests are deliberately left to the caller: this script writes
 * files and reports what it wrote on stdout as JSON, so the workflow stays
 * debuggable and a local run does not touch your branches.
 */

import { catalogue } from "./catalogue";
import { fetchPaper } from "./lib/sources";
import { ask, askJsonChecked, assertModelAvailable, model } from "./lib/llm";
import { $ } from "bun";

const CONTENT_DIR = "src/content/ideas";
const SIM_DIR = "src/components/react/simulations";

interface EntryPlan {
	slug: string;
	title: string;
	subtitle: string;
	impact: string;
	readTime: string;
	category: "idea" | "paper" | "deep-dive" | "tutorial" | "concept";
	status: "RESEARCH" | "CONCEPT" | "PROTOTYPE" | "ALPHA" | "ARCHIVED";
	tags: string[];
	simulationName: string;
	simulationBrief: string;
	relatedSlugs: string[];
	body: string;
}

const PLAN_SCHEMA = {
	name: "entry_plan",
	schema: {
		type: "object",
		required: [
			"slug",
			"title",
			"subtitle",
			"impact",
			"readTime",
			"category",
			"status",
			"tags",
			"simulationName",
			"simulationBrief",
			"relatedSlugs",
			"body",
		],
		properties: {
			slug: { type: "string" },
			title: { type: "string" },
			subtitle: { type: "string" },
			impact: { type: "string" },
			readTime: { type: "string" },
			category: { type: "string", enum: ["idea", "paper", "deep-dive", "tutorial", "concept"] },
			status: { type: "string", enum: ["RESEARCH", "CONCEPT", "PROTOTYPE", "ALPHA", "ARCHIVED"] },
			tags: { type: "array", items: { type: "string" } },
			simulationName: { type: "string" },
			simulationBrief: { type: "string" },
			relatedSlugs: { type: "array", items: { type: "string" } },
			body: { type: "string" },
		},
	},
};

const MIN_WORDS = 600;
const MIN_HEADINGS = 4;
const MIN_SIM_LINES = 150;

/**
 * Reject output that is structurally hollow.
 *
 * The build and the browser check both pass on a truncated stub — they verify
 * that nothing crashes, not that anything was said. These are the cheap
 * structural facts that separate a real entry from a placeholder.
 */
export function planProblems(p: EntryPlan, truncated: boolean): string[] {
	const bad: string[] = [];
	const body = p.body ?? "";
	const words = body.trim().split(/\s+/).filter(Boolean).length;

	if (truncated) bad.push("the reply hit the token cap, so the entry is cut off mid-generation");
	if (words < MIN_WORDS) bad.push(`body is ${words} words, needs at least ${MIN_WORDS}`);

	const headings = (body.match(/^##\s+\S/gm) ?? []).length;
	if (headings < MIN_HEADINGS) bad.push(`only ${headings} '##' headings, needs at least ${MIN_HEADINGS}`);

	// A heading run into its own paragraph means the model lost its newlines.
	if (/^#{1,3}\s+\S[^\n]{120,}/m.test(body)) bad.push("a heading runs into body text on the same line");

	if (!/```mermaid/.test(body)) bad.push("no mermaid diagram");
	if (!/```(python|ts|tsx|javascript)/.test(body)) bad.push("no code block");

	// Truncation usually shows up as a final line with no terminal punctuation.
	const tail = body.trimEnd().slice(-1);
	if (tail && !".!?`)]|\"".includes(tail)) bad.push(`body ends mid-sentence ("…${body.trimEnd().slice(-40)}")`);

	if (!p.simulationName || !/^[A-Z][A-Za-z0-9]*$/.test(p.simulationName))
		bad.push(`simulationName "${p.simulationName}" is not PascalCase`);

	return bad;
}

/**
 * Numbers in the prose that the abstract does not contain.
 *
 * This is the check that matters for unattended publishing: structure and a
 * clean build say nothing about whether a benchmark figure was invented. Code
 * blocks and mermaid are stripped first, since illustrative constants there are
 * not claims about the paper.
 */
export function unsupportedNumbers(body: string, abstract: string): string[] {
	const prose = body
		.replace(/```[\s\S]*?```/g, " ") // fenced code and diagrams
		.replace(/`[^`]*`/g, " ") // inline code
		.replace(/\$\$[\s\S]*?\$\$/g, " ") // display math
		.replace(/\$[^$\n]*\$/g, " "); // inline math

	const haystack = abstract.replace(/[,\s]/g, "");
	const found = new Set<string>();

	for (const m of prose.matchAll(/\d[\d,]*\.?\d*\s*(?:%|×|x\b|B\b|M\b|K\b|p\b|-DoF\b)?/g)) {
		const raw = m[0].trim();
		const digits = raw.replace(/[^\d.]/g, "").replace(/\.$/, "");
		if (!digits) continue;

		// Years, small ordinals and the paper's own identifier are not claims.
		if (digits.length < 2) continue;
		if (/^(19|20|21)\d{2}$/.test(digits)) continue;
		// arXiv identifiers are references, not claims.
		if (/^\d{4}\.\d{4,5}$/.test(digits)) continue;
		if (abstract.includes(digits)) continue;
		if (haystack.includes(digits.replace(/[,\s]/g, ""))) continue;

		found.add(raw.replace(/\s+/g, ""));
	}

	return [...found];
}

/** Structural checks on the generated component, before the build ever runs. */
export function simProblems(src: string): string[] {
	const bad: string[] = [];
	const lines = src.split("\n").length;
	if (lines < MIN_SIM_LINES) bad.push(`simulation is ${lines} lines, needs at least ${MIN_SIM_LINES}`);
	if (!/export default/.test(src)) bad.push("no default export");
	// The canonical mulberry32 mixes with 0x6d2b79f5; other constants signal a
	// hallucinated generator that will not produce a usable distribution.
	if (/mulberry32|rng\s*\(/.test(src) && !/0x6d2b79f5/.test(src))
		bad.push("the seeded RNG is not mulberry32 — check the constants");

	// Canvas silently ignores an unresolvable colour, leaving whatever was set
	// before. This drew three grey lines where three coloured ones were intended.
	// Only the literal form is detectable with certainty: passing the token
	// through a resolver, as `resolve("var(--purple)")`, is the correct pattern.
	if (/(?:strokeStyle|fillStyle)\s*=\s*["'`]\s*var\(--/.test(src))
		bad.push("a canvas colour is assigned a literal var(--token); canvas cannot resolve CSS variables");

	if (/getContext\(["']2d["']\)/.test(src)) {
		// A canvas that uses tokens anywhere but never calls getPropertyValue has
		// no way to have resolved them. This catches the indirect form, where the
		// token is held in a data structure and assigned later.
		if (/var\(--/.test(src) && !/getPropertyValue/.test(src))
			bad.push("canvas uses var(--token) colours but never resolves them via getPropertyValue");
		if (!/devicePixelRatio/.test(src))
			bad.push("canvas does not scale for devicePixelRatio, so it will render blurry");
		if (!/ResizeObserver|clientWidth|offsetWidth/.test(src))
			bad.push("canvas is not sized from its container, so it will not fit or be responsive");
	}

	return bad;
}

function arg(name: string, fallback?: string): string | undefined {
	const i = Bun.argv.indexOf(`--${name}`);
	return i === -1 ? fallback : Bun.argv[i + 1];
}

function frontmatter(p: EntryPlan, pdfUrl: string, date: string): string {
	const esc = (s: string) => `"${s.replace(/"/g, '\\"')}"`;
	return [
		"---",
		`title: ${esc(p.title)}`,
		`subtitle: ${esc(p.subtitle)}`,
		`date: ${date}`,
		`status: ${p.status}`,
		`category: ${p.category}`,
		`impact: ${esc(p.impact)}`,
		`readTime: ${esc(p.readTime)}`,
		"tags:",
		...p.tags.map((t) => `  - ${t}`),
		`coverImage: https://picsum.photos/seed/${p.slug}/800/600?grayscale`,
		`simulation: ${p.simulationName}`,
		`pdfUrl: ${pdfUrl}`,
		"featured: true",
		"---",
		"",
	].join("\n");
}

async function main() {
	const id = Bun.argv[2];
	if (!id || !/^\d{4}\.\d{4,5}(v\d+)?$/.test(id)) {
		console.error("usage: bun scripts/build-entry.ts <arxiv-id> [--retries N]");
		process.exit(2);
	}
	const retries = Number(arg("retries", "3"));

	await assertModelAvailable();
	console.error(`# build-entry ${id} · model ${model()}`);

	// --- 1. the authoritative record -------------------------------------------
	const paper = await fetchPaper(id);
	console.error(`  "${paper.title}"`);
	console.error(`  ${paper.authors.slice(0, 4).join(", ")}${paper.authors.length > 4 ? " et al." : ""} · ${paper.published}`);

	const entries = await catalogue();
	const clash = entries.find((e) => e.arxiv.includes(paper.id));
	if (clash) {
		console.error(`\nAlready covered by ${clash.slug} ("${clash.title}"). Nothing to do.`);
		process.exit(3);
	}

	const rubric = await Bun.file("docs/scan-rubric.md").text();
	const sketch = await Bun.file(`src/components/react/SketchElements.tsx`).text();
	const agents = await Bun.file("AGENTS.md").text();

	const cataloguePlain = entries
		.map((e) => `- /idea/${e.slug}/ — ${e.title} (tags: ${e.tags.join(", ") || "—"})`)
		.join("\n");

	// --- 2. the written entry ---------------------------------------------------
	console.error("\ngenerating entry…");
	const { value: plan, truncated } = await askJsonChecked<EntryPlan>({
		system:
			"You write for PaperLens: technical explainers of AI research for working engineers. " +
			"Objective third person, no first-person pronouns. You never state a number or claim " +
			"the provided abstract does not support. Plain language, define terms on first use.",
		user: [
			"# Rubric and house style",
			rubric,
			"",
			"# Content conventions (excerpt from AGENTS.md)",
			agents.slice(0, 6000),
			"",
			"# Existing entries you may cross-link to",
			cataloguePlain,
			"",
			"# The paper",
			`arXiv: ${paper.id}`,
			`Title: ${paper.title}`,
			`Authors: ${paper.authors.join(", ")}`,
			`Submitted: ${paper.published}`,
			"",
			"Abstract:",
			paper.abstract,
			"",
			"# Task",
			"Write the entry. Return JSON.",
			"",
			"- `slug`: kebab-case, contains the searchable name of the artefact.",
			"- `body`: the markdown body only, no frontmatter. Start at `# Executive Summary`.",
			"  Use `##` headings that answer the obvious follow-up questions. Include one",
			"  mermaid diagram in a ```mermaid fence, KaTeX math with $...$ and $$...$$ where",
			"  it adds precision, a markdown table of results, and a Python implementation",
			"  sketch in a ```python fence.",
			"- Quote ONLY numbers present in the abstract above. If the abstract does not give",
			"  a figure, describe the result qualitatively instead of inventing one.",
			"- Cross-link to relevant existing entries using plain paths like /idea/some-slug/.",
			"  Never use [[wikilink]] syntax; this project has no plugin for it.",
			"- `relatedSlugs`: existing entry slugs that should gain a reverse link to this one.",
			"- `simulationName`: PascalCase, no 'Simulation' suffix (the file adds it).",
			"- `simulationBrief`: 3-5 sentences describing the interactive toy to build — what",
			"  state it holds, what the controls change, and what the reader should notice.",
			"  It must demonstrate the mechanism honestly. If the paper's real effect is modest,",
			"  the toy shows a modest effect.",
			"- Mermaid node labels containing braces, brackets or parentheses must be quoted,",
			'  e.g. A["Suffix N-grams g_{t,n}"], or the diagram fails to parse.',
		].join("\n"),
		schema: PLAN_SCHEMA,
		maxTokens: 16000,
	});

	const problems = planProblems(plan, truncated);
	if (problems.length) {
		console.error("\nThe generated entry is not publishable:");
		for (const p of problems) console.error(`  - ${p}`);
		console.error(
			"\nThis is a model-capability problem, not a build error, so no repair loop will fix it.\n" +
				`Try a stronger PAPERLENS_MODEL (currently ${model()}).`,
		);
		console.log(JSON.stringify({ ok: false, reason: "entry failed the quality gate", problems, arxiv: paper.id }));
		process.exit(4);
	}

	// Numeric claims the abstract does not support. Not fatal — the entry may
	// legitimately cite a related work — but it decides whether this is safe to
	// merge without a human reading it.
	const claims = unsupportedNumbers(plan.body, paper.abstract);
	if (claims.length) {
		console.error(`\n  ! ${claims.length} number(s) not found in the abstract: ${claims.join(", ")}`);
		console.error("    a human should check these before this is published");
	} else {
		console.error("\n  · every number in the body appears in the abstract");
	}

	const ext = "md";
	const entryPath = `${CONTENT_DIR}/${plan.slug}.${ext}`;
	const simPath = `${SIM_DIR}/${plan.simulationName}Simulation.tsx`;
	const pdfUrl = `https://arxiv.org/pdf/${paper.id}`;

	await Bun.write(entryPath, frontmatter(plan, pdfUrl, paper.published) + plan.body.trim() + "\n");
	console.error(`  wrote ${entryPath}`);

	// --- 3. the simulation ------------------------------------------------------
	console.error("generating simulation…");
	const simRules = [
		"# Shared components you must build on (src/components/react/SketchElements.tsx)",
		sketch,
		"",
		"# What makes one of these good",
		"The reader should come away understanding the mechanism better than the prose",
		"alone would leave them. That means a control they move must change an outcome",
		"they can read as a number, and the change must be the paper's claim.",
		"",
		"Before writing, decide: what is the single quantity this paper improves, and",
		"what would it look like with the mechanism off? Build the toy around that",
		"contrast. A side-by-side of with/without, or a curve that visibly fails to",
		"converge, teaches; a panel of sliders wired to a decorative drawing does not.",
		"",
		"Be honest about size. If the paper's real effect is modest, show a modest",
		"effect and say so. Never invent a dramatic result or a penalty the paper does",
		"not report.",
		"",
		"# Hard requirements",
		"- Default-export a single React functional component. TypeScript, React 19.",
		"- Import from '../SketchElements' — SchematicCard, SchematicButton, DataReadout, TechBadge.",
		"- Icons from 'lucide-react'.",
		"- At least three controls, and EVERY piece of state must be read by something",
		"  that renders. No state or effect dependency that nothing consumes.",
		"- At least one numeric readout that visibly changes when a control changes.",
		"  State the units.",
		"- No decorative parameters. If a control only rescales the drawing without",
		"  changing what is being claimed, it is a zoom, not a demonstration — cut it.",
		"- Never fabricate structure the paper contradicts. If the paper says things",
		"  happen simultaneously, do not render them round-robin.",
		"",
		"# Colour",
		"- Use the CSS tokens: var(--green) primary, var(--orange), var(--aqua),",
		"  var(--purple), var(--red), var(--fg), var(--fg2), var(--fg4), var(--bg0),",
		"  var(--bg1), var(--bg2), var(--bg3). Tailwind semantic classes are available",
		"  too: bg-bg0, text-ink, text-ink1, text-ink2, text-ink4, text-mute, border-bg2,",
		"  text-mint-400, text-amber-400. No raw hex, no light theme.",
		"- CRITICAL: a canvas cannot resolve CSS variables. `ctx.strokeStyle =",
		"  'var(--purple)'` is invalid and silently keeps the previous colour. Resolve",
		"  tokens first with getComputedStyle(document.documentElement).getPropertyValue,",
		"  and pass the resolved string to the canvas.",
		"",
		"# If you draw on a canvas",
		"- Size it to its container with a ResizeObserver on a wrapper div, and set",
		"  canvas.width = cssWidth * devicePixelRatio with a matching setTransform,",
		"  or it will be blurry and a fixed width will overflow on mobile.",
		"- Fill the space. A mostly-empty canvas with a handful of points reads as",
		"  broken. Fit the drawing to the extent of the data with padding.",
		"- Label it: axis labels or a scale bar, and a legend naming every series with",
		"  its colour. A line with no units is decoration.",
		"",
		"# Correctness",
		"- Any randomness must be seeded with a mulberry32 rng(seed) helper defined",
		"  in-file, mixing with 0x6d2b79f5, so the figure is identical on every load.",
		"- Never touch window or document during render; guard effects with a typeof check.",
		"- Respect prefers-reduced-motion: no autoplaying animation when it is set.",
		"- Must not scroll horizontally at 390px wide. No fixed pixel widths on containers.",
		"- Every labelled section must contain something. Never render a heading over",
		"  an empty region.",
		"- Include a short 'What to try' block naming 3-4 specific things to change and",
		"  what each reveals, and a closing note stating plainly that this is a toy",
		"  abstraction, with the paper's real measured numbers left to the entry.",
		"- No external libraries beyond react and lucide-react. No network calls.",
		"- Aim for 250-450 lines. Below that it will not demonstrate enough.",
	].join("\n");

	let sim = await ask({
		system:
			"You write small, self-contained, mathematically honest React simulations. " +
			"You never fabricate an effect the underlying paper does not support.",
		user: [
			simRules,
			"",
			"# The paper",
			`${paper.title} (arXiv:${paper.id})`,
			paper.abstract,
			"",
			"# The simulation to build",
			plan.simulationBrief,
			"",
			"# Task",
			`Write ${plan.simulationName}Simulation.tsx in full.`,
			"Output ONLY the TypeScript source. No markdown fences, no commentary.",
		].join("\n"),
		maxTokens: 16000,
		temperature: 0.2,
	});

	const strip = (s: string) => s.replace(/^\s*```(?:tsx?|typescript)?\s*\n/, "").replace(/\n```\s*$/, "").trim();
	const simSource = strip(sim);
	const simIssues = simProblems(simSource);
	if (simIssues.length) {
		console.error("\nThe generated simulation is not publishable:");
		for (const p of simIssues) console.error(`  - ${p}`);
		console.error(`\nTry a stronger PAPERLENS_MODEL (currently ${model()}).`);
		console.log(JSON.stringify({ ok: false, reason: "simulation failed the quality gate", problems: simIssues, arxiv: paper.id }));
		process.exit(4);
	}

	await Bun.write(simPath, simSource + "\n");
	console.error(`  wrote ${simPath}`);

	// --- 4. reverse links -------------------------------------------------------
	for (const slug of plan.relatedSlugs.slice(0, 3)) {
		const target = entries.find((e) => e.slug === slug);
		if (!target) continue;
		const existing = await Bun.file(target.file).text();
		if (existing.includes(`/idea/${plan.slug}/`)) continue;
		await Bun.write(
			target.file,
			`${existing.trimEnd()}\n\n## Related\n\n[${plan.title}](/idea/${plan.slug}/) covers a closely related mechanism.\n`,
		);
		console.error(`  linked back from ${slug}`);
	}

	// --- 5. build until clean ---------------------------------------------------
	for (let attempt = 0; attempt <= retries; attempt++) {
		console.error(`\nbuilding (attempt ${attempt + 1}/${retries + 1})…`);
		const result = await $`bun run build`.nothrow().quiet();
		if (result.exitCode === 0) {
			console.error("  build clean");
			console.log(
				JSON.stringify({
					ok: true,
					slug: plan.slug,
					entryPath,
					simPath,
					title: plan.title,
					arxiv: paper.id,
					simulation: plan.simulationName,
					relatedSlugs: plan.relatedSlugs,
					unsupportedNumbers: claims,
				}),
			);
			return;
		}

		const log = (result.stderr.toString() + result.stdout.toString()).slice(-4000);
		console.error(`  build failed:\n${log.split("\n").slice(-12).join("\n")}`);
		if (attempt === retries) break;

		console.error("  asking for a repair…");
		const current = await Bun.file(simPath).text();
		const entryText = await Bun.file(entryPath).text();
		const fixed = await ask({
			system: "You fix build errors in Astro/React/TypeScript projects. You return complete files, never diffs.",
			user: [
				"The build failed. Here is the tail of the output:",
				"```",
				log,
				"```",
				"",
				`# ${simPath}`,
				"```tsx",
				current,
				"```",
				"",
				`# ${entryPath}`,
				"```markdown",
				entryText.slice(0, 8000),
				"```",
				"",
				"Identify the cause and return ONLY the corrected content of whichever file is at",
				"fault, prefixed by a single line naming it exactly:",
				`FILE: ${simPath}`,
				"or",
				`FILE: ${entryPath}`,
				"Then the full corrected file. No fences, no commentary.",
			].join("\n"),
			maxTokens: 16000,
			temperature: 0.1,
		});

		const match = fixed.match(/^FILE:\s*(\S+)\s*\n([\s\S]*)$/);
		if (!match) {
			console.error("  repair reply was not in the expected form; retrying build as-is");
			continue;
		}
		const [, which, content] = match;
		const path = which.trim() === entryPath ? entryPath : simPath;
		await Bun.write(path, strip(content) + "\n");
		console.error(`  rewrote ${path}`);
	}

	console.log(JSON.stringify({ ok: false, slug: plan.slug, entryPath, simPath, arxiv: paper.id }));
	process.exit(1);
}

if (import.meta.main) {
	main().catch((err) => {
		console.error(err instanceof Error ? err.message : String(err));
		process.exit(1);
	});
}
