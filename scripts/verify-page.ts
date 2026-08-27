#!/usr/bin/env bun
/**
 * Browser verification for a built page.
 *
 * `bun run build` passing proves the page compiles, not that it works: a
 * simulation that throws on mount builds perfectly. This starts the preview
 * server, loads the page, and fails on console/page errors, an unmounted
 * simulation, unrendered mermaid, or horizontal scroll at 390px.
 *
 *   bun scripts/verify-page.ts /idea/some-slug/
 */

import { chromium } from "playwright";
import { $ } from "bun";

const path = Bun.argv[2];
if (!path) {
	console.error("usage: bun scripts/verify-page.ts /idea/<slug>/");
	process.exit(2);
}

const PORT = 4321;
const server = Bun.spawn(["bun", "run", "preview", "--port", String(PORT)], {
	stdout: "pipe",
	stderr: "pipe",
});

async function waitForServer(timeoutMs = 30_000) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		try {
			const res = await fetch(`http://localhost:${PORT}/`);
			if (res.ok) return;
		} catch {}
		await new Promise((r) => setTimeout(r, 500));
	}
	throw new Error("preview server did not come up");
}

const problems: string[] = [];

try {
	await waitForServer();
	const browser = await chromium.launch();
	const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

	const errors: string[] = [];
	page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
	page.on("console", (m) => {
		if (m.type() === "error") errors.push(`console: ${m.text()}`);
	});

	const url = `http://localhost:${PORT}${path}`;
	const response = await page.goto(url, { waitUntil: "networkidle" });
	if (!response || response.status() >= 400) problems.push(`page returned ${response?.status()}`);
	await page.waitForTimeout(3000);

	const state = await page.evaluate(() => {
		const host = document.querySelector(".sim-host");
		return {
			hasSimHost: Boolean(host),
			simText: (host as HTMLElement | null)?.innerText.trim().length ?? 0,
			simError: /is not in the registry/.test((host as HTMLElement | null)?.innerText ?? ""),
			mermaidUnrendered: document.querySelectorAll(".mermaid-diagram:not(.mermaid-rendered)").length,
			deadWikilinks: [...document.querySelectorAll(".doc a")].filter((a) =>
				(a.getAttribute("href") ?? "").includes("[["),
			).length,
			wideScroll: document.documentElement.scrollWidth > window.innerWidth + 1,
		};
	});

	if (state.hasSimHost && state.simError) problems.push("simulation is not in the registry");
	if (state.hasSimHost && !state.simError && state.simText < 120)
		problems.push(`simulation rendered almost nothing (${state.simText} chars) — likely threw on mount`);
	if (state.mermaidUnrendered) problems.push(`${state.mermaidUnrendered} mermaid diagram(s) failed to render`);
	if (state.deadWikilinks) problems.push(`${state.deadWikilinks} dead [[wikilink]] href(s)`);
	if (state.wideScroll) problems.push("page scrolls horizontally at 1280px");

	await page.setViewportSize({ width: 390, height: 844 });
	await page.waitForTimeout(1000);
	if (await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1))
		problems.push("page scrolls horizontally at 390px");

	problems.push(...errors);
	await browser.close();
} catch (err) {
	problems.push(`verification threw: ${(err as Error).message}`);
} finally {
	server.kill();
	await $`pkill -f "astro preview" || true`.nothrow().quiet();
}

if (problems.length) {
	console.error(`FAIL ${path}`);
	for (const p of problems) console.error(`  - ${p}`);
	process.exit(1);
}
console.log(`OK ${path} — mounts clean, no console errors, no horizontal scroll`);
