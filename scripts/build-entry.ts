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
import { ask, askJson, assertModelAvailable, model } from "./lib/llm";
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
	const plan = await askJson<EntryPlan>({
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
		"# House rules",
		"- Default-export a single React functional component. TypeScript, React 19.",
		"- Import from '../SketchElements' — SchematicCard, SchematicButton, DataReadout, TechBadge.",
		"- Icons from 'lucide-react'.",
		"- Colour comes from CSS custom properties: var(--green) primary, var(--orange),",
		"  var(--aqua), var(--purple), var(--red), var(--fg), var(--fg2), var(--fg4),",
		"  var(--bg0), var(--bg1), var(--bg2), var(--bg3). Tailwind semantic classes are",
		"  available too: bg-bg0, text-ink, text-ink1, text-ink2, text-ink4, text-mute,",
		"  border-bg2, text-mint-400, text-amber-400. No raw hex, no light theme.",
		"- Any randomness must be seeded with a mulberry32 rng(seed) helper defined in-file,",
		"  so the figure is identical on every load.",
		"- Never touch window or document during render; guard effects with a typeof check.",
		"- Respect prefers-reduced-motion: no autoplaying animation when it is set.",
		"- Must not scroll horizontally at 390px wide. No fixed pixel widths on containers.",
		"- Include a short 'What to try' block and a note stating plainly that this is a toy",
		"  abstraction, with the paper's real measured numbers left to the entry.",
		"- No external libraries beyond react and lucide-react. No network calls.",
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
	await Bun.write(simPath, strip(sim) + "\n");
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
			console.log(JSON.stringify({ ok: true, slug: plan.slug, entryPath, simPath, title: plan.title, arxiv: paper.id, simulation: plan.simulationName, relatedSlugs: plan.relatedSlugs }));
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

main().catch((err) => {
	console.error(err instanceof Error ? err.message : String(err));
	process.exit(1);
});
