#!/usr/bin/env bun
/**
 * Measures how tall each simulation renders, so the page can hold the space.
 *
 * The simulations are `client:only` islands: nothing of them exists in the HTML,
 * and they mount anywhere between 500 and 2000 pixels tall into a 260px
 * placeholder. Everything below them jumps when they arrive, at the very top of
 * the article — the shift Core Web Vitals measures.
 *
 * This runs against a built site and writes the measured heights, which
 * src/pages/idea/[slug].astro turns into a min-height on the placeholder.
 * It is not part of `bun run build` (it needs the build to already exist);
 * rerun it when a simulation's layout changes:
 *
 *   bun run build && bun scripts/sim-metrics.ts
 */

import { chromium } from "playwright";
import { readEntries } from "./lib/entries";

const OUT = "src/data/sim-heights.json";
const PORT = 4331;
/** the width the CSS switches to the narrow layout at — see src/styles/global.css */
const BREAKPOINT = 720;

const server = Bun.spawn(["bun", "run", "preview", "--port", String(PORT)], {
	stdout: "pipe",
	stderr: "pipe",
});

async function waitForServer(timeoutMs = 30_000) {
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		try {
			if ((await fetch(`http://localhost:${PORT}/`)).ok) return;
		} catch {}
		await new Promise((r) => setTimeout(r, 500));
	}
	throw new Error("preview server did not come up");
}

try {
	await waitForServer();

	const entries = readEntries().filter((e) => e.simulation);
	const browser = await chromium.launch();
	const heights: Record<string, { lg: number; sm: number }> = {};

	for (const entry of entries) {
		const measure = async (width: number) => {
			const page = await browser.newPage({ viewport: { width, height: 900 } });
			await page.goto(`http://localhost:${PORT}/idea/${entry.slug}/`, { waitUntil: "networkidle" });
			// the island is empty until React mounts; measuring before that reads the placeholder
			await page
				.waitForSelector(".sim-host astro-island:not(:empty)", { timeout: 20_000 })
				.catch(() => {});
			const height = await page.evaluate(
				() => document.querySelector(".sim-host")?.getBoundingClientRect().height ?? 0,
			);
			await page.close();
			return Math.round(height);
		};

		// the first page of a run occasionally comes back before the island exists;
		// one retry is enough, and a second zero means the simulation really is broken
		const at = async (width: number) => (await measure(width)) || (await measure(width));

		const lg = await at(1280);
		const sm = await at(390);
		if (!lg || !sm) {
			console.error(`  ${entry.slug}: no simulation rendered (lg=${lg} sm=${sm}), skipped`);
			continue;
		}

		heights[entry.slug] = { lg, sm };
		console.error(`  ${entry.slug}: ${lg}px / ${sm}px`);
	}

	await browser.close();

	const sorted = Object.fromEntries(Object.entries(heights).sort(([a], [b]) => a.localeCompare(b)));
	await Bun.write(
		OUT,
		`${JSON.stringify({ breakpoint: BREAKPOINT, heights: sorted }, null, 2)}\n`,
	);
	console.error(`# wrote ${OUT} (${Object.keys(sorted).length} simulations)`);
} finally {
	server.kill();
}
