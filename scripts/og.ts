#!/usr/bin/env bun
/**
 * Social cards, drawn at build time.
 *
 * Every entry gets a 1200x630 PNG that looks like the site does: the title bar,
 * a prompt line, the title, and the status bar carrying the real metadata.
 * The old frontmatter pointed og:image at a random picsum photo, so every share
 * of a paper about world models showed an unrelated grayscale stock image.
 *
 *   bun scripts/og.ts           # only what is missing or stale
 *   bun scripts/og.ts --force   # redraw everything
 */

import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { chromium, type Browser } from "playwright";
import { readEntries, type Entry } from "./lib/entries";

const OUT_DIR = "public/og";
const FONT = "public/fonts/geist-mono-latin.woff2";
const force = Bun.argv.includes("--force");

const palette = {
	bg0: "#0d0f13",
	bg0h: "#080a0d",
	bg1: "#161a20",
	bg2: "#22272f",
	bg3: "#323944",
	fg: "#e6eaf0",
	fg2: "#98a2ae",
	fg4: "#6b7583",
	green: "#35d492",
	orange: "#f5a623",
	blue: "#4c9ef5",
	aqua: "#5cc8ff",
	purple: "#8b6cf2",
};

/** same mapping the thumbnails use — see src/lib/thumb.ts */
const CATEGORY_COLOR: Record<string, string> = {
	paper: palette.blue,
	"deep-dive": palette.orange,
	idea: palette.purple,
	concept: palette.aqua,
	tutorial: palette.aqua,
};

const escape = (s: string) =>
	s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");

/** the longest entry title is ~75 characters; every step keeps it inside three lines */
function titleSize(title: string): number {
	if (title.length > 80) return 40;
	if (title.length > 55) return 48;
	if (title.length > 35) return 56;
	return 64;
}

function shell(accent: string, body: string): string {
	const font = readFileSync(FONT).toString("base64");
	return `<!doctype html>
<html><head><meta charset="utf-8"><style>
@font-face {
  font-family: 'Geist Mono';
  font-weight: 100 900;
  src: url(data:font/woff2;base64,${font}) format('woff2');
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  width: 1200px; height: 630px; overflow: hidden;
  background: ${palette.bg0};
  color: ${palette.fg};
  font-family: 'Geist Mono', monospace;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
  display: flex; flex-direction: column;
}
/* the faint grid the panes sit on */
body::before {
  content: ''; position: absolute; inset: 0;
  background-image:
    linear-gradient(${palette.bg1} 1px, transparent 1px),
    linear-gradient(90deg, ${palette.bg1} 1px, transparent 1px);
  background-size: 60px 60px;
  opacity: 0.35;
}
.bar {
  position: relative; flex: none; display: flex; align-items: center; gap: 28px;
  height: 62px; padding: 0 40px;
  background: ${palette.bg0h}; border-bottom: 1px solid ${palette.bg2};
  font-size: 22px; color: ${palette.fg4};
}
.brand { color: ${accent}; font-weight: 600; }
.n { color: ${palette.bg3}; }
.bar .right { margin-left: auto; color: ${accent}; }
.body { position: relative; flex: 1; padding: 46px 40px 0; display: flex; flex-direction: column; }
.status {
  position: relative; flex: none; display: flex; align-items: center;
  height: 54px; background: ${palette.bg0h}; border-top: 1px solid ${palette.bg2};
  font-size: 20px; color: ${palette.fg2};
}
.seg { padding: 0 22px; border-right: 1px solid ${palette.bg2}; height: 100%; display: flex; align-items: center; gap: 10px; }
.seg.mode { background: ${accent}; color: ${palette.bg0h}; font-weight: 700; letter-spacing: 0.08em; }
.seg.grow { flex: 1; border-right: none; }
.seg.end { border-right: none; border-left: 1px solid ${palette.bg2}; }
.prompt { font-size: 22px; color: ${palette.fg4}; margin-bottom: 30px; }
.prompt .u { color: ${palette.green}; }
.prompt .h { color: ${palette.blue}; }
.prompt .cmd { color: ${palette.fg2}; }
h1 {
  font-size: ${"var(--title-size)"}; font-weight: 600; line-height: 1.22;
  letter-spacing: -0.02em; max-width: 1050px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.sub {
  margin-top: 26px; font-size: 22px; line-height: 1.55; color: ${palette.fg2}; max-width: 1000px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.tags { margin-top: auto; margin-bottom: 34px; display: flex; gap: 18px; font-size: 20px; color: ${palette.fg4}; }
.tags span::before { content: '#'; color: ${palette.bg3}; }
.figlet { font-size: 24px; line-height: 1.3; color: ${palette.green}; white-space: pre; }
.stats { margin-top: auto; margin-bottom: 34px; display: flex; gap: 26px; font-size: 20px; color: ${palette.green}; }
.dot { color: ${accent}; }
</style></head><body>${body}</body></html>`;
}

function entryCard(entry: Entry): string {
	const accent = CATEGORY_COLOR[entry.category] ?? palette.green;
	const ext = entry.file.endsWith(".mdx") ? "mdx" : "md";
	const tags = entry.tags
		.slice(0, 4)
		.map((t) => `<span>${escape(t.toLowerCase().replace(/\s+/g, "-"))}</span>`)
		.join("");

	return shell(
		accent,
		`<div class="bar">
  <span class="brand">paperlens</span>
  <span><span class="n">0:</span>home</span>
  <span><span class="n">1:</span>papers</span>
  <span class="right">${escape(entry.category)}</span>
</div>
<div class="body">
  <p class="prompt"><span class="u">xavier</span>@<span class="h">paperlens</span> $ <span class="cmd">cat ~/papers/${escape(entry.slug)}.${ext}</span></p>
  <h1 style="--title-size: ${titleSize(entry.title)}px">${escape(entry.title)}</h1>
  <p class="sub">${escape(entry.subtitle)}</p>
  <div class="tags">${tags}</div>
</div>
<div class="status">
  <span class="seg mode">${escape(entry.status)}</span>
  <span class="seg">${escape(entry.date)}</span>
  <span class="seg">${escape(entry.readTime)} read</span>
  ${entry.simulation ? `<span class="seg"><span class="dot">●</span> interactive simulation</span>` : ""}
  <span class="seg grow"></span>
  <span class="seg end">paperlens.io</span>
</div>`,
	);
}

function defaultCard(entries: Entry[]): string {
	const sims = entries.filter((e) => e.simulation).length;
	const figlet = `                              _
  _ __   __ _ _ __   ___ _ __| | ___ _ __  ___
 | '_ \\ / _\` | '_ \\ / _ \\ '__| |/ _ \\ '_ \\/ __|
 | |_) | (_| | |_) |  __/ |  | |  __/ | | \\__ \\
 | .__/ \\__,_| .__/ \\___|_|  |_|\\___|_| |_|___/
 |_|         |_|`;

	return shell(
		palette.green,
		`<div class="bar">
  <span class="brand">paperlens</span>
  <span><span class="n">0:</span>home</span>
  <span><span class="n">1:</span>papers</span>
  <span><span class="n">2:</span>whoami</span>
  <span class="right">${entries.length} entries</span>
</div>
<div class="body">
  <p class="prompt"><span class="u">xavier</span>@<span class="h">paperlens</span> $ <span class="cmd">ls ~/papers</span></p>
  <pre class="figlet">${escape(figlet)}</pre>
  <p class="sub" style="margin-top: 44px; color: ${palette.fg}">papers and research for AI engineers, visualized.</p>
  <p class="sub" style="margin-top: 10px">every entry ships with a simulation, the math, and the code — drag a slider instead of trusting a figure.</p>
  <div class="stats">
    <span>${entries.length} entries indexed</span>
    <span>${sims} simulations loaded</span>
  </div>
</div>
<div class="status">
  <span class="seg mode">NORMAL</span>
  <span class="seg">~/papers</span>
  <span class="seg grow"></span>
  <span class="seg end">paperlens.io</span>
</div>`,
	);
}

async function draw(browser: Browser, html: string, out: string) {
	const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
	await page.setContent(html, { waitUntil: "load" });
	await page.evaluate(() => document.fonts.ready);
	await page.screenshot({ path: out, type: "png" });
	await page.close();
}

/** a card is stale when its entry has been edited since it was drawn */
function stale(out: string, source: string): boolean {
	if (force || !existsSync(out)) return true;
	return statSync(source).mtimeMs > statSync(out).mtimeMs;
}

async function main() {
	if (!existsSync(FONT)) throw new Error(`missing ${FONT} — the card needs the site typeface`);
	mkdirSync(OUT_DIR, { recursive: true });

	const entries = readEntries();
	const todo = entries.filter((e) => stale(`${OUT_DIR}/${e.slug}.png`, e.file));
	const defaultOut = "public/og-default.png";
	const needsDefault =
		force ||
		!existsSync(defaultOut) ||
		entries.some((e) => statSync(e.file).mtimeMs > statSync(defaultOut).mtimeMs);

	if (!todo.length && !needsDefault) {
		console.error(`# og: ${entries.length} cards up to date`);
		return;
	}

	const browser = await chromium.launch();
	try {
		if (needsDefault) {
			await draw(browser, defaultCard(entries), defaultOut);
			console.error("  og-default.png");
		}
		for (const entry of todo) {
			await draw(browser, entryCard(entry), `${OUT_DIR}/${entry.slug}.png`);
			console.error(`  og/${entry.slug}.png`);
		}
	} finally {
		await browser.close();
	}
	console.error(`# og: drew ${todo.length + (needsDefault ? 1 : 0)} card(s)`);
}

await main();
